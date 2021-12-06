import os
import h5py
import torch
import imageio
import numpy as np
from smplx import SMPL

from .dataset import BaseH5Dataset
from .process_spin import SMPL_JOINT_MAPPER, write_to_h5py
from .utils.skeleton_utils import *

# to align the ground plan to x-z plane
zju_to_nerf_rot = np.array([[1, 0, 0],
                            [0, 0,-1],
                            [0, 1, 0]], dtype=np.float32)

num_train_frames = {
    '313': 60,
    '315': 300,
    '377': 300,
    '386': 300,
    '387': 300,
    '390': 300, # begin ith frame == 700
    '392': 300,
    '393': 300,
    '394': 300,
    '395': 300,
    '396': 540, # begin ith frame == 810
}

def get_mask(path, img_path):
    '''
    Following NeuralBody repo
    https://github.com/zju3dv/neuralbody/blob/master/lib/datasets/light_stage/can_smpl.py#L46    '''
    mask_path = os.path.join(path, 'mask', img_path[:-4] + '.png')
    mask = imageio.imread(mask_path)
    mask = (mask != 0).astype(np.uint8)

    mask_path = os.path.join(path, 'mask_cihp', img_path[:-4] + '.png')
    mask_cihp = imageio.imread(mask_path)
    mask_cihp = (mask_cihp != 0).astype(np.uint8)
    mask = (mask | mask_cihp).astype(np.uint8)

    border = 5
    kernel = np.ones((border, border), np.uint8)
    #mask_erode = cv2.erode(mask.copy(), kernel)
    sampling_mask = cv2.dilate(mask.copy(), kernel, iterations=2)

    #sampling_mask = mask.copy()
    #sampling_mask[(mask_dilate - mask_erode) == 1] = 100

    return mask, sampling_mask

@torch.no_grad()
def get_smpls(path, kp_idxs, gender='neutral', ext_scale=1.0, scale_to_ref=True,
              ref_pose=smpl_rest_pose):
    '''
    Note: it's yet-another-smpl-coordinate system
    bodies: the vertices from ZJU dataset
    '''
    bones, betas, root_bones, root_locs = [], [], [], []
    zju_vertices = []
    for kp_idx in kp_idxs:
        param_path = os.path.join(path, 'params', f'{kp_idx}.npy')
        params = np.load(param_path, allow_pickle=True).item()
        bone = params['poses'].reshape(-1, 24, 3)
        beta = params['shapes']
        # load the provided vertices
        zju_vert = np.load(os.path.join(path, 'vertices', f'{kp_idx}.npy')).astype(np.float32)

        zju_vertices.append(zju_vert)
        bones.append(bone)
        betas.append(beta)
        root_bones.append(params['Rh'])
        root_locs.append(params['Th'])

    bones = np.concatenate(bones, axis=0).reshape(-1, 3)
    betas = np.concatenate(betas, axis=0)
    root_bones = np.concatenate(root_bones, axis=0)
    # note: this is in global space, but different in ZJU's system
    Tp = torch.FloatTensor(np.concatenate(root_locs, axis=0))

    # intentionally separate these for clarity
    Rn = torch.FloatTensor(zju_to_nerf_rot[None]) # rotation to align ground plane to x-z
    zju_global_orient = axisang_to_rot(torch.FloatTensor(root_bones))
    rots = axisang_to_rot(torch.FloatTensor(bones)).view(-1, 24, 3, 3)
    rots[..., 0, :, :] = Rn @ zju_global_orient
    root_bones = rot_to_axisang(rots[..., 0, :, :].clone()).numpy()
    betas = torch.FloatTensor(betas)

    # note that the rotation definition is different for ZJU_mocap
    # SMPL is joints = (RX + T), where the parenthesis implies the thing happened in SMPL()
    # but ZJU is joints =  R'(RX + T) + T', where R' and T' is the global rotation/translation
    # here, we directly include R' in R (and also a Rn to align the ground plane), so the vertices we get become
    # joints = (RnR'RX + T)
    # so to get the correct vertices in ZJU's system, we need to do
    # joints = (RnR'RX + T) - T + RnR'T + RnT'
    # WARNING: they define the T in R = 0 matrix!


    # 1. get T
    dummy = torch.zeros(1, 24, 3, 3).float()
    smpl = SMPL(model_path='smpl', gender=gender, joint_mapper=SMPL_JOINT_MAPPER)

    T = smpl(betas=betas.mean(0)[None],
             body_pose=dummy[:, 1:],
             global_orient=dummy[:, :1],
             pose2rot=False).joints[0, 0]
    # 2. get the rest pose
    dummy = torch.eye(3).view(1, 1, 3, 3).expand(-1, 24, -1, -1)

    rest_pose = smpl(betas=betas.mean(0)[None],
                     body_pose=dummy[:, 1:],
                     global_orient=dummy[:, :1],
                     pose2rot=False).joints[0]
    rest_pose = rest_pose.numpy()
    rest_pose -= rest_pose[0] # center rest pose

    # scale the rest pose if needed
    if scale_to_ref:
        ref_pose = ref_pose * ext_scale
        bone_len = calculate_bone_length(rest_pose).mean()
        ref_bone_len = calculate_bone_length(ref_pose).mean()
        pose_scale = ref_bone_len / bone_len
    else:
        pose_scale = 1.0
    rest_pose = rest_pose * pose_scale

    # 3. get RhR'T
    T = T.view(1, 1, 3)
    RnRpT = (T @ rots[:, 0].permute(0, 2, 1))

    # 4. get the posed joints/vertices
    smpl_output = smpl(betas=betas,
                       body_pose=rots[:, 1:],
                       global_orient=rots[:, :1],
                       pose2rot=False)
    # apply (RhR'RX + T) - T + RhR'T + RhT'
    RnTp = (Rn @ Tp.view(-1, 3, 1)).view(-1, 1, 3)
    joints = smpl_output.joints - T + RnRpT + RnTp
    vertices = smpl_output.vertices - T + RnRpT + RnTp
    joints *= pose_scale
    vertices *= pose_scale

    root_locs = joints[:, 0].numpy()
    bones = bones.reshape(-1, 24, 3)
    bones[:, 0] = root_bones
    l2ws = np.array([get_smpl_l2ws(bone, rest_pose=rest_pose) for bone in bones])
    l2ws[..., :3, -1] += root_locs[:, None]

    kp3d = l2ws[..., :3, -1]
    skts = np.array([np.linalg.inv(l2w) for l2w in l2ws])

    return betas, kp3d, bones, skts, rest_pose, vertices, pose_scale


def process_zju_data(data_path, subject='377', training_view=[0, 6, 12, 18],
                     i_intv=1, split='train',
                     ext_scale=0.001, res=None, skel_type=SMPLSkeleton):
    '''
    ni, i_intv, intv, begin_i: setting from NeuralBody
    '''
    assert ext_scale == 0.001 # TODO: deal with this later
    # TODO: might have to check if this is true for all
    H = W = 1024 # default image size.
    ni = num_train_frames[subject]
    begin_i = 0
    if subject == '390':
        begin_i = 700
    elif subject == '396':
        begin_i = 810

    if res is not None:
        H = int(H * res)
        W = int(W * res)

    subject_path = os.path.join(data_path, f"CoreView_{subject}")
    annot_path = os.path.join(subject_path, "annots.npy")
    annots = np.load(annot_path, allow_pickle=True).item()
    cams = annots['cams']
    num_cams = len(cams['K'])
    i = begin_i
    if split == 'train':
        view = training_view
        idxs = slice(i, i + ni * i_intv)
    else:
        view = [1,4,5,10,17,20]
        if subject != '392':
            idxs = np.concatenate([np.arange(1, 31), np.arange(400, 601)])
        else:
            idxs = np.concatenate([np.arange(1, 31), np.arange(400, 556)])
        i_intv = 1


    # extract image and the corresponding camera indices
    img_paths = np.array([
        np.array(imgs_data['ims'])[view]
        for imgs_data in np.array(annots['ims'])[idxs][::i_intv]
    ]).ravel()

    cam_idxs = np.array([
        np.arange(len(imgs_data['ims']))[view]
        for imgs_data in np.array(annots['ims'])[idxs][::i_intv]
    ]).ravel()

    # Extract information from the dataset.
    imgs, masks, sampling_masks = [], [], []
    kp_idxs = []
    for img_path, cam_idx in zip(img_paths, cam_idxs):

        K = np.array(cams['K'][cam_idx])
        D = np.array(cams['D'][cam_idx])

        img = imageio.imread(os.path.join(subject_path, img_path))
        mask, sampling_mask = get_mask(subject_path, img_path)

        img = cv2.undistort(img, K, D)
        mask = cv2.undistort(mask, K, D)[..., None]
        img = img * mask # + (1 - mask) * 255
        sampling_mask = cv2.undistort(sampling_mask, K, D)[..., None]

        # resize the corresponding data as needed
        if res is not None:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            sampling_mask = cv2.resize(sampling_mask, (W, H), interpolation=cv2.INTER_NEAREST)
            K[:2] = K[:2] * res

        if subject == '313' or subject == '315':
            kp_idx = int(os.path.basename(img_path).split('_')[4])
        else:
            kp_idx = int(os.path.basename(img_path)[:-4])

        imgs.append(img)
        masks.append(mask)
        sampling_masks.append(sampling_mask)
        kp_idxs.append(kp_idx)

    unique_cams = np.unique(cam_idxs)
    imgs = np.array(imgs)
    masks = np.array(masks).reshape(-1, H, W, 1)
    sampling_masks = np.array(sampling_masks).reshape(-1, H, W, 1)

    # get extrinsic data
    c2ws, focals, centers = [], [], []
    for c in range(num_cams):
        R = np.array(cams['R'][c])
        T = np.array(cams['T'][c]) / 1000. # in 1m system.
        K = np.array(cams['K'][c])

        # get camera-to-world matrix from extrinsic
        ext = np.concatenate([R, T], axis=-1)
        ext = np.concatenate([ext, np.array([[0, 0, 0., 1.]])], axis=0)
        c2w = np.linalg.inv(ext)
        c2w[:3, -1:] = zju_to_nerf_rot @ c2w[:3, -1:]
        c2w[:3, :3] = zju_to_nerf_rot @ c2w[:3, :3]
        c2ws.append(c2w)

        # save intrinsic data
        if res is not None:
            K[:2] = K[:2] * res
        focals.append([K[0, 0], K[1, 1]])
        centers.append(K[:2, -1])

    focals = np.array(focals)
    centers = np.array(centers)
    c2ws = np.array(c2ws).astype(np.float32)
    c2ws = swap_mat(c2ws) # to NeRF format

    # get pose-related data
    betas, kp3d, bones, skts, rest_pose, vertices, pose_scale = get_smpls(subject_path, np.unique(kp_idxs),
                                                                          scale_to_ref=False)
    cyls = get_kp_bounding_cylinder(kp3d,
                                    ext_scale=ext_scale,
                                    skel_type=skel_type,
                                    extend_mm=250,
                                    top_expand_ratio=1.00,
                                    bot_expand_ratio=0.25,
                                    head='-y')
    if split == 'test':
        kp_idxs = np.arange(len(kp_idxs))
    elif subject == '313' or subject == '315':
        kp_idxs = np.array(kp_idxs) - 1
    elif subject == '390':
        kp_idxs = np.array(kp_idxs) - 700

    return {'imgs': np.array(imgs),
            'masks': np.array(masks).reshape(-1, H, W, 1),
            'sampling_masks': np.array(sampling_masks).reshape(-1, H, W, 1),
            'c2ws': c2ws.astype(np.float32),
            'img_pose_indices': cam_idxs,
            'kp_idxs': np.array(kp_idxs),
            'centers': centers.astype(np.float32),
            'focals': focals.astype(np.float32),
            'kp3d': kp3d.astype(np.float32),
            'betas': betas.numpy().astype(np.float32),
            'bones': bones.astype(np.float32),
            'skts': skts.astype(np.float32),
            'cyls': cyls.astype(np.float32),
            'rest_pose': rest_pose.astype(np.float32),
            #'vertices': vertices
            }

class ZJUMocapDataset(BaseH5Dataset):

    N_render = 15
    render_skip = 63

    def __init__(self, *args, **kwargs):
        super(ZJUMocapDataset, self).__init__(*args, **kwargs)

    def init_meta(self):
        super(ZJUMocapDataset, self).init_meta()

        dataset = h5py.File(self.h5_path, 'r')
        self.kp_idxs = dataset['kp_idxs'][:]
        self.cam_idxs = dataset['img_pose_indices'][:]

        # create fake background
        self.has_bg = True
        self.bgs = np.zeros((1, *self.HW, 3), dtype=np.uint8).reshape(1, -1, 3)
        self.bg_idxs = np.arange(len(dataset['imgs'])) * 0
        print('WARNING: ZJUMocap does not support pose refinement for now (_get_subset_idxs is not implemented)')
        dataset.close()

    def get_meta(self):
        data_attrs = super(ZJUMocapDataset, self).get_meta()
        return data_attrs

    def get_kp_idx(self, idx, q_idx):
        '''
        idx: the actual index(s) for the full .h5 dataset
        q_idx: the 'queried' index(s) received from the sampler,
               may not coincide with idx.
        '''
        return self.kp_idxs[idx], q_idx

    def get_cam_idx(self, idx, q_idx):
        '''
        idx: the actual index(s) for the full .h5 dataset
        q_idx: the 'queried' index(s) received from the sampler,
               may not coincide with idx.
        '''
        return self.cam_idxs[idx], q_idx

    def _get_subset_idxs(self):
        '''
        get the part of data that you want to train on
        '''
        if self._idx_map is not None:
            i_idxs = self._idx_map
            _k_idxs = self._idx_map
            _c_idxs = self._idx_map
            _kq_idxs = np.arange(len(self._idx_map))
            _cq_idxs = np.arange(len(self._idx_map))
        else:
            i_idxs = np.arange(self._N_total_img)
            _k_idxs = _kq_idxs = np.arange(self._N_total_img)
            _c_idxs = _cq_idxs = np.arange(self._N_total_img)

        # call the dataset-dependent fns to get the true kp/cam idx
        k_idxs, kq_idxs = self.get_kp_idx(_k_idxs, _kq_idxs)
        c_idxs, cq_idxs = self.get_cam_idx(_c_idxs, _cq_idxs)

        return k_idxs, c_idxs, i_idxs, kq_idxs, cq_idxs


if __name__ == '__main__':
    #from renderer import Renderer
    import argparse
    parser = argparse.ArgumentParser(description='Arguments for masks extraction')
    parser.add_argument("-s", "--subject", type=str, default="377",
                        help='subject to extract')
    parser.add_argument("--split", type=str, default="train",
                        help='split to use')
    args = parser.parse_args()
    subject = args.subject
    split = args.split
    data_path = 'data/zju_mocap/'
    print(f"Processing {subject}_{split}...")
    data = process_zju_data(data_path, subject, split=split, res=1.0)
    #dd.io.save(os.path.join(data_path, f"{subject}_{split}.h5"), data)
    write_to_h5py(os.path.join(data_path, f"{subject}_{split}_h5py.h5"), data)

