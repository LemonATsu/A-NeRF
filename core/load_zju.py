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

def get_mask(path, img_path, erode_border=False):
    '''
    Following NeuralBody repo
    https://github.com/zju3dv/neuralbody/blob/master/lib/datasets/light_stage/can_smpl.py#L46    '''

    mask_path = os.path.join(path, 'mask', img_path[:-4] + '.png')
    mask = None
    if os.path.exists(mask_path):
        mask = imageio.imread(mask_path)
        mask = (mask != 0).astype(np.uint8)

    mask_path = os.path.join(path, 'mask_cihp', img_path[:-4] + '.png')
    mask_cihp = None
    if os.path.exists(mask_path):
        mask_cihp = imageio.imread(mask_path)
        mask_cihp = (mask_cihp != 0).astype(np.uint8)
    
    if mask is not None and mask_cihp is not None:
        mask = (mask | mask_cihp).astype(np.uint8)
    elif mask_cihp is not None:
        mask = mask_cihp
    
    border = 5
    kernel = np.ones((border, border), np.uint8)
    #mask_erode = cv2.erode(mask.copy(), kernel)
    sampling_mask = cv2.dilate(mask.copy(), kernel, iterations=3)

    #sampling_mask = mask.copy()
    #sampling_mask[(mask_dilate - mask_erode) == 1] = 100
    if erode_border:
        dilated = cv2.dilate(mask.copy(), kernel) 
        eroded = cv2.erode(mask.copy(), kernel) 
        sampling_mask[(dilated - eroded) == 1] = 0
    #eroded = cv2.erode(mask.copy(), kernel, iterations=1) 
    #mask = eroded.copy()

    return mask, sampling_mask

@torch.no_grad()
def get_smpls(path, kp_idxs, gender='neutral', ext_scale=1.0, scale_to_ref=True,
              ref_pose=smpl_rest_pose, param_path=None, model_path=None, vertices_path=None):
    '''
    Note: it's yet-another-smpl-coordinate system
    bodies: the vertices from ZJU dataset
    '''

    if param_path is None:
        param_path = 'params'
    if model_path is None:
        model_path = 'smpl'
    if vertices_path is None:
        vertices_path = 'vertices'

    bones, betas, root_bones, root_locs = [], [], [], []
    zju_vertices = []
    for kp_idx in kp_idxs:
        params = np.load(os.path.join(path, param_path, f'{kp_idx}.npy'), allow_pickle=True).item()
        bone = params['poses'].reshape(-1, 24, 3)
        beta = params['shapes']
        # load the provided vertices
        zju_vert = np.load(os.path.join(path, vertices_path, f'{kp_idx}.npy')).astype(np.float32)

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
    smpl = SMPL(model_path=model_path, gender=gender, joint_mapper=SMPL_JOINT_MAPPER)

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
    imgs = np.zeros((len(img_paths), H, W, 3), dtype=np.uint8)
    masks = np.zeros((len(img_paths), H, W, 1), dtype=np.uint8)
    sampling_masks = np.zeros((len(img_paths), H, W, 1), dtype=np.uint8)
    kp_idxs = []
    for i, (img_path, cam_idx) in enumerate(zip(img_paths, cam_idxs)):

        if i % 50 == 0:
            print(f'{i+1}/{len(img_paths)}')

        K = np.array(cams['K'][cam_idx])
        D = np.array(cams['D'][cam_idx])

        img = imageio.imread(os.path.join(subject_path, img_path))
        mask, sampling_mask = get_mask(subject_path, img_path, erode_border=True)

        img = cv2.undistort(img, K, D)
        mask = cv2.undistort(mask, K, D)[..., None]
        sampling_mask = cv2.undistort(sampling_mask, K, D)[..., None]
        mask[mask > 1] = 1

        # resize the corresponding data as needed
        if res is not None and res != 1.0:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            sampling_mask = cv2.resize(sampling_mask, (W, H), interpolation=cv2.INTER_NEAREST)
            K[:2] = K[:2] * res

        if subject == '313' or subject == '315':
            kp_idx = int(os.path.basename(img_path).split('_')[4])
        else:
            kp_idx = int(os.path.basename(img_path)[:-4])

        imgs[i] = img
        masks[i] = mask
        sampling_masks[i] = sampling_mask
        kp_idxs.append(kp_idx)

    unique_cams = np.unique(cam_idxs)
    bkgds = np.zeros((num_cams, H, W, 3), dtype=np.uint8)
    # get background
    for c in unique_cams:
        cam_imgs = imgs[cam_idxs==c].reshape(-1, H, W, 3)
        cam_masks = masks[cam_idxs==c].reshape(-1, H, W, 1)
        N_cam_imgs = len(cam_imgs)
        for h_ in range(H):
            for w_ in range(W):
                vals = []
                is_bg = np.where(cam_masks[:, h_, w_] < 1)[0]
                if len(is_bg) == 0:
                    med = np.array([0, 0, 0]).astype(np.uint8)
                else:
                    med = np.median(cam_imgs[is_bg, h_, w_], axis=0)
                bkgds[c, h_, w_] = med.astype(np.uint8) 

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
            'bkgds': np.array(bkgds),
            'bkgd_idxs': cam_idxs,
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


def set_h36m_zju_config(ann_file, num_train_frame, num_eval_frame, begin_ith_frame=0, frame_interval=5,
                        smpl='new_smpl', params='new_params', vertices='new_vertices', erode_border=True, 
                        smpl_path='smplx'):
    return {'ann_file': ann_file, 
            'num_train_frame': num_train_frame,
            'num_eval_frame': num_eval_frame,
            'begin_ith_frame': begin_ith_frame, 
            'frame_interval': frame_interval,
            'smpl': smpl,
            'params': params,
            'vertices': vertices,
            'erode_border': erode_border,
            'smpl_path': smpl_path
           }

h36m_zju_configs = {
    'S1': set_h36m_zju_config('Posing/annots.npy', 150, 49),
    'S5': set_h36m_zju_config('Posing/annots.npy', 250, 127),
    'S6': set_h36m_zju_config('Posing/annots.npy', 150, 83),
    'S7': set_h36m_zju_config('Posing/annots.npy', 300, 200),
    'S8': set_h36m_zju_config('Posing/annots.npy', 250, 87),
    'S9': set_h36m_zju_config('Posing/annots.npy', 260, 133),
    'S11': set_h36m_zju_config('Posing/annots.npy', 200, 82),
}

def process_h36m_zju_data(data_path, subject='S1', training_view=[0,1,2], split='train', res=None,
                          ext_scale=0.001, skel_type=SMPLSkeleton):
    '''
    Note: they only use the Posing sequence for training
    '''
    H = 1000
    W = 1000 
    assert ext_scale == 0.001
    
    if res is not None and res != 1.0:
        H = int(H * res)
        W = int(W * res)
    
    config = h36m_zju_configs[subject]
    subject_path = os.path.join(data_path, f"{subject}")
    annot_path = os.path.join(subject_path, config['ann_file'])
    annots = np.load(annot_path, allow_pickle=True).item()
    subject_path = os.path.join(subject_path, 'Posing') # h36m-zju only use this sequence
    
    cams = annots['cams']
    num_cams = len(cams['K'])

    # following animatable NeRF's format
    i = config['begin_ith_frame']
    i_intv = config['frame_interval']
    ni = config['num_train_frame']
    
    if split == 'train':
        view = training_view
    else:
        
        view = np.array([i for i in range(num_cams) if i not in training_view])
        if len(view) == 0:
            view = [0]

        i = config['begin_ith_frame'] + config['num_train_frame'] * i_intv
        ni = config['num_eval_frame']
        
    # extract image and the corresponding camera indices
    img_paths = np.array([
            np.array(imgs_data['ims'])[view]
            for imgs_data in annots['ims'][i:i + ni * i_intv][::i_intv]
    ]).ravel()
    cam_idxs = np.array([
            np.arange(len(imgs_data['ims']))[view]
            for imgs_data in annots['ims'][i:i + ni * i_intv][::i_intv]
    ]).ravel()
    
    kp_ids = []
    imgs = np.zeros((len(img_paths), H, W, 3), dtype=np.uint8)
    masks = np.zeros((len(img_paths), H, W, 1), dtype=np.uint8)
    sampling_masks = np.zeros((len(img_paths), H, W, 1), dtype=np.uint8)

    for i, (img_path, cam_idx) in enumerate(zip(img_paths, cam_idxs)):
        
        if i % 50 == 0:
            print(f'{i+1}/{len(img_paths)}')
        
        # get camera parameters
        K = np.array(cams['K'][cam_idx])
        D = np.array(cams['D'][cam_idx])
        
        # retrieve images
        img = imageio.imread(os.path.join(subject_path, img_path))
        mask, sampling_mask = get_mask(subject_path, img_path, erode_border=config['erode_border'])
        
        # process image and mask
        img = cv2.undistort(img, K, D)
        mask = cv2.undistort(mask, K, D)[..., None]
        sampling_mask = cv2.undistort(sampling_mask, K, D)[..., None]
        mask[mask > 1] = 1

        # resize the corresponding data as needed
        if res is not None and res != 1.0:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            sampling_mask = cv2.resize(sampling_mask, (W, H), interpolation=cv2.INTER_NEAREST)
            K[:2] = K[:2] * res

        # grab img_id for creating kp_idxs
        kp_id = img_path.split('/')[-1][:-4]
        imgs[i] = img
        masks[i] = mask
        sampling_masks[i] = sampling_mask
        kp_ids.append(kp_id)
    kp_ids = [int(kp_id) for kp_id in kp_ids]
    kp_ids, kp_idxs = np.unique(kp_ids, return_inverse=True)
    unique_cams = np.unique(cam_idxs)
    bkgds = np.zeros((num_cams, H, W, 3), dtype=np.uint8)

    # get background
    for c in unique_cams:
        cam_imgs = imgs[cam_idxs==c].reshape(-1, H, W, 3)
        cam_masks = masks[cam_idxs==c].reshape(-1, H, W, 1)
        N_cam_imgs = len(cam_imgs)
        for h_ in range(H):
            for w_ in range(W):
                vals = []
                is_bg = np.where(cam_masks[:, h_, w_] < 1)[0]
                if len(is_bg) == 0:
                    med = np.array([0, 0, 0]).astype(np.uint8)
                else:
                    med = np.median(cam_imgs[is_bg, h_, w_], axis=0)
                bkgds[c, h_, w_] = med.astype(np.uint8) 

    # get camera data
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
    betas, kp3d, bones, skts, rest_pose, vertices, pose_scale = get_smpls(
                                                                    subject_path, kp_ids,
                                                                    scale_to_ref=False,
                                                                    model_path=os.path.join(data_path, config['smpl_path'], 'smpl'),
                                                                    param_path=config['params'],
                                                                    vertices_path=config['vertices'],
                                                                  )
    cyls = get_kp_bounding_cylinder(kp3d,
                                    ext_scale=ext_scale,
                                    skel_type=skel_type,
                                    extend_mm=250,
                                    top_expand_ratio=1.00,
                                    bot_expand_ratio=0.25,
                                    head='-y')

    return {'imgs': np.array(imgs),
            'bkgds': np.array(bkgds),
            'bkgd_idxs': cam_idxs,
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
            }

class ZJUMocapDataset(BaseH5Dataset):

    N_render = 15
    render_skip = 63

    def __init__(self, *args, **kwargs):
        super(ZJUMocapDataset, self).__init__(*args, **kwargs)

    def init_meta(self):
        if self.split == 'test':
            self.h5_path = self.h5_path.replace('train', 'test')
        super(ZJUMocapDataset, self).init_meta()

        dataset = h5py.File(self.h5_path, 'r')
        self.kp_idxs = dataset['kp_idxs'][:]
        self.cam_idxs = dataset['img_pose_indices'][:]

        if self.split == 'test':
            n_unique_cam = len(np.unique(self.cam_idxs))
            self.kp_idxs = self.kp_idxs // n_unique_cam

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

    def _get_subset_idxs(self, render=False):
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

class ZJUH36MDataset(ZJUMocapDataset):

    N_render = 30
    render_skip = 1

    def init_meta(self):
        if self.split == 'test':
            self.h5_path = self.h5_path.replace('train', 'test')
        super(ZJUH36MDataset, self).init_meta()

        dataset = h5py.File(self.h5_path, 'r')
        self.kp_idxs = dataset['kp_idxs'][:]
        self.cam_idxs = dataset['img_pose_indices'][:]


        idxs = np.arange(len(self.kp_idxs))
        train_idxs, val_idxs = idxs[:-30], idxs[-30:]
        if self.split == 'train':
            self._idx_map = train_idxs
        elif self.split == 'val':
            self._idx_map = val_idxs

        dataset.close()

    def _get_subset_idxs(self, render=False):
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
    parser.add_argument("-d", "--dataset", type=str, default='mocap')
    parser.add_argument("-s", "--subject", type=str, default="377",
                        help='subject to extract')
    parser.add_argument("--split", type=str, default="train",
                        help='split to use')
    args = parser.parse_args()
    dataset = args.dataset
    subject = args.subject
    split = args.split

    if dataset == 'mocap':
        data_path = 'data/zju_mocap/'
        print(f"Processing {subject}_{split}...")
        data = process_zju_data(data_path, subject, split=split, res=1.0)
    elif dataset == 'h36m':
        data_path = 'data/h36m_zju'
        print(f"Processing {subject}_{split}...")
        data = process_h36m_zju_data(data_path, subject, split=split, res=1.0)
    else:
        raise NotImplementedError(f'Unknown dataset {dataset}')

    write_to_h5py(os.path.join(data_path, f"{subject}_{split}.h5"), data)

