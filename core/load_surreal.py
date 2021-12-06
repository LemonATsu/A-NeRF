import os
import cv2
import glob
import math
import torch
import pickle
import imageio
import numpy as np
from scipy.io import loadmat

from .process_spin import write_to_h5py
from .dataset import BaseH5Dataset
from .utils.skeleton_utils import *
from .utils.ray_utils import get_rays_np

# a fixed scale that roughly make skeleton of all datasets
# to be of similar range
dataset_ext_scale = 0.25 / 0.00035

def generate_camera_path(amp_wubble=15, amp_dist=0.5, dist=3.0, center=np.array([0., 0., 0.]), n_cams=60):

    y_angles = np.linspace(0, math.radians(360), n_cams+1)[:-1]
    wave = np.sin(np.linspace(0, math.radians(360*4), n_cams+1)[:-1])
    dist_offset = amp_dist * wave

    # base camera
    loc_camera = np.array([dist, 0., 0.], dtype=np.float32)
    forward = np.array([1.0, 0., 0.], dtype=np.float32)
    right = np.array([0., 0., -1.], dtype=np.float32)
    up = np.cross(forward, right)

    cam = np.stack([right, up, forward, loc_camera], axis=1)
    cam = np.concatenate([cam, np.array([[0., 0., 0., 1.]])], axis=0)

    poses = []
    center_trans = translate(*center)
    # zoom-in and out
    for a, d in zip(y_angles, dist_offset):
        pose = center_trans @ rotate_y(a) @ translate(d, 0., 0.) @ cam
        poses.append(pose)

    # wubble up and down
    wubble_offset = math.radians(amp_wubble) * wave + math.radians(30)
    for a, w in zip(y_angles, wubble_offset):
        pose = rotate_y(a) @ rotate_z(w)  @ cam
        poses.append(pose)

    return np.array(poses).astype(np.float32)

def dilate_masks(masks, extend_iter=1):
    d_kernel = np.ones((5, 5))
    dilated_masks = []

    for mask in masks:
        dilated = cv2.dilate(mask, kernel=d_kernel,
                             iterations=extend_iter)
        dilated_masks.append(dilated)

    return np.array(dilated_masks)

def get_img_cam_indices(N_imgs, N_seqs, N_kps, N_kp_per_seq, seq_cam_type):
    N_cams_per_type = int(N_imgs / (N_seqs * N_kp_per_seq))
    N_cam_type  = len(np.unique(seq_cam_type))

    """
    # from idx to the answer
    img_cam_idx = np.arange(N_imgs).reshape(-1, N_seqs, N_kp_per_seq)
    img_cam_base = img_cam_idx // N_kps
    img_cam_seq = (img_cam_idx % N_kps) // N_kp_per_seq
    img_cam_idx = img_cam_base + N_cams_per_type * seq_cam_type[img_cam_seq]
    """

    # we know how many cams for each type
    # and then find where to add the offset to (and how much)
    indices = np.arange(N_cams_per_type).reshape(-1, 1, 1).repeat(N_seqs, 1).repeat(N_kp_per_seq, 2)
    indices = indices + seq_cam_type[None, :, None] * N_cams_per_type
    return indices.reshape(-1)

def extract_seq_indices(seqs, N_imgs, N_kps, N_kp_per_seq=35, val_skip=1):

    selected_kp = []
    for seq in seqs:
        selected_kp.extend(np.arange(N_kp_per_seq * seq, N_kp_per_seq * (seq + 1)).tolist())
    selected_kp = np.array(selected_kp)

    all_indices = []
    for i in range(0, N_imgs, N_kps):
        i_sub = []
        for seq in seqs:
            idx = i + np.arange(seq * N_kp_per_seq, (seq + 1) * N_kp_per_seq)
            i_sub.extend(idx)
        all_indices.append(i_sub)
    selected_indices = np.array(all_indices[::val_skip]).reshape(-1)

    return selected_indices, selected_kp, np.concatenate(all_indices)


def process_surreal_data(h5_file, data_path, extend_iter=2, ext_scale=0.001,
                         exclude_seq=[]):

    print("START PROCESSING SURREAL")

    rot_rootbone = np.array([[1., 0., 0., 0.],
                         [0., 0.,-1., 0.],
                         [0., 1., 0., 0.],
                         [0., 0., 0., 1.]]).astype(np.float32)

    rot_glob = np.array([[1., 0., 0., 0.],
                         [0.,-1., 0., 0.],
                         [0., 0.,-1., 0.],
                         [0., 0., 0., 1.]]).astype(np.float32)
    # From surreal generation code
    beta = np.array([-0.8010307 ,  0.6838105 ,  0.7480726 , -1.1379223 , -0.32415348,
                     -0.8404733 , -0.4795286 , -0.63125765, -0.13453396,  1.4934114 ]).astype(np.float32)
    beta = beta.reshape(-1, 10)

    ext_scale = dataset_ext_scale * ext_scale
    sorted_glob = lambda path: sorted(glob.glob(path))

    data_dirs = sorted_glob(os.path.join(data_path, "*_*/"))

    # load metadata
    cams, kp_3d, bone_poses, render_types, seq_cam_type, focals = [], [], [], [], [], []
    fg_masks, imgs = None, None
    for i, data_dir in enumerate(data_dirs):
        with open(os.path.join(data_dir, "metadata.pkl"), "rb") as meta_f:
            metadata = pickle.load(meta_f)
        focals.append(metadata["focal"] * metadata["int_scale"])

        # set up camera
        render_type = metadata["render_type"]
        cam = metadata["cams"]
        if render_type not in render_types:
            render_types.append(render_type)

            cam[..., :3, -1] *= ext_scale
            cams.append(cam)

        seq_cam_type.append(render_types.index(render_type))

        # setup kp
        # assume all dataset have the same N_kp
        N_kp_per_seq = metadata["N_kp"]
        # subset: all the subfolders for this motion sequence
        N_cam_this_seq = metadata["N_cams"]
        N_cam_per_subdir = metadata["N_cam_per_subdir"]
        kp_3d.append(metadata["joints3D"] * ext_scale)
        bone_poses.append(metadata["poses"].reshape(N_kp_per_seq, -1, 3))

        #assert (metadata["N_cam_per_subdir"] % view_skip) == 0

        # collect all the data for this set from the subset
        fg_seq = []
        fg_paths = sorted_glob(os.path.join(data_dir, "*-*/", "*segm.mat"))
        for fg_path in fg_paths:
            fg_mask = loadmat(fg_path)["data"]
            fg_mask = fg_mask.reshape(N_cam_per_subdir, N_kp_per_seq, *fg_mask.shape[-2:])
            fg_mask[fg_mask > 0] = 1
            fg_seq.append(fg_mask)
        fg_seq = np.concatenate(fg_seq, axis=0)

        # pre-allocate to save memory:
        if fg_masks is None:
            pre_alloc_shape = (min(N_cam_this_seq, fg_seq.shape[0]),
                               len(data_dirs)*fg_seq.shape[1], *fg_seq.shape[-2:])
            fg_masks = np.zeros(pre_alloc_shape, dtype=np.uint8)
        fg_masks[:, i*fg_seq.shape[1]:(i+1)*fg_seq.shape[1]] = fg_seq[:len(fg_masks)]

        # get image path and read
        img_seq_paths = sorted_glob(os.path.join(data_dir, "*-*/", "imageSequences/*.png"))
        img_seq_paths = np.array(img_seq_paths).reshape(N_cam_this_seq, N_kp_per_seq)

        # TODO: read it here
        img_seq = []
        for img_seq_path in img_seq_paths.reshape(-1):
            img_seq.append(imageio.imread(img_seq_path)[..., :3])
        img_shape = img_seq[-1].shape

        # pre-allocate to save memory:
        if imgs is None:
            imgs = np.zeros((*fg_masks.shape, 3), dtype=np.uint8)
        img_seq = np.array(img_seq).reshape(-1, N_kp_per_seq, *img_shape)
        imgs[:, i*N_kp_per_seq:(i+1)*N_kp_per_seq] = img_seq[:len(imgs)]

    ####################################
    # Organized data that we just read #
    ####################################
    kp_3d = np.array(kp_3d)
    bone_poses = np.array(bone_poses)
    N_joints = kp_3d.shape[-2]
    kp_3d = kp_3d.reshape(-1, N_joints, 3)
    bone_poses = bone_poses.reshape(-1, N_joints, 3)
    N_kps = kp_3d.shape[0]

    focal = np.mean(focals)
    H, W, C = imgs.shape[-3:]
    N_seqs, N_cam_per_type = len(data_dirs), imgs.shape[0]

    # of shape (#cams_per_type, #seqs * #poses, H, W, ...)
    imgs = imgs.reshape(-1, H, W, C)
    fg_masks = fg_masks.reshape(-1, H, W)

    # Note: we have two different type of cameras,
    # each type has the same #cams but different config
    # to save memory, we order them in (#cams_per_type, #seqs * #poses, ...)
    # (compares to having #cam_type * #cams_per_type in the first dim.)
    seq_cam_type = np.array(seq_cam_type)
    img_cam_indices = get_img_cam_indices(imgs.shape[0], N_seqs, N_kps,
                                          N_kp_per_seq, seq_cam_type)


    # create sampling masks
    sampling_masks = fg_masks if extend_iter == 0 else dilate_masks(fg_masks, extend_iter)

    # make it (-1, H, W, 1)
    fg_masks = fg_masks[..., None]#.astype(np.float32)
    sampling_masks = sampling_masks[..., None]

    # ok, deal with c2ws.
    c2ws = np.array(cams).reshape(-1, 4, 4)
    c2ws = rot_glob[None] @ c2ws.copy()
    N_types = c2ws.shape[0]

    ############################
    # Process bones and things #
    ############################
    skel_type = SMPLSkeleton
    skts, transformed_kps = None, None

    # bones already have a rotation, just need to correct it instead of using a global one
    root_bones = torch.FloatTensor(bone_poses[:, :1]).view(-1, 3)
    root_rots = torch.FloatTensor(rot_rootbone[None, :3, :3]) @ axisang_to_rot(root_bones)
    rotated_root = rot_to_axisang(root_rots).view(-1, 1, 3).cpu().numpy()
    bone_poses[:, :1] = rotated_root

    kp_3d = kp_3d @ rot_glob[:3, :3].T
    # This is local-to-world matrix, and we need world-to-local
    # Note: the pose has pelvis centered at (0, 0, 0), so it's offseted from
    # kp3d with a translation.
    skts, l2ws = skt_from_smpl(bone_poses, scale=ext_scale, kp_3d=kp_3d)
    print("spit back")

    cylinder_params = None
    # note: we change ext_scale from arg scale to surreal in previously
    # now we need to scale it back to arg scale
    cylinder_params = get_kp_bounding_cylinder(kp_3d,
                                               ext_scale=ext_scale / dataset_ext_scale,
                                               skel_type=skel_type, extend_mm=250,
                                               head='-y')

    rays_od = [get_rays_np(H, W, focal, c2w) for c2w in c2ws]
    for i, (mask, cam_idx) in enumerate(zip(sampling_masks, img_cam_indices)):
        cyl = cylinder_params[i % N_kps]
        rays_o, rays_d = rays_od[cam_idx]
        # consider 2D projection only
        rays_o = rays_o.reshape(-1, 3)[..., [0, -1]]
        rays_d = rays_d.reshape(-1, 3)[..., [0, -1]]

        # some very far away vector
        far = rays_o + rays_d * 100.
        near = rays_o #+ rays_d * 0.35
        of = far - near#rays_o
        od = cyl[:2] - near#rays_o # first 2-dim is the center
        dist = np.abs(np.cross(of, od)) / np.linalg.norm(of, axis=-1)

        radius = cyl[2]
        mask[..., 0] *= (1 * (dist < radius).reshape(H, W)).astype(np.uint8)



    # expand #c2ws to match number of images
    c2ws = c2ws[img_cam_indices]
    root_id = skel_type.root_id
    nonroot_id = skel_type.nonroot_id

    fake_bkgd = (np.ones((1,)+imgs.shape[1:]) * 255).astype(np.uint8)
    bkgd_idxs = np.zeros((len(imgs),)).astype(np.int64)
    focals = np.ones((len(imgs),)) * focal
    rest_pose = smpl_rest_pose * ext_scale

    surreal_dict = {'imgs': imgs,
                    'masks': fg_masks,
                    'sampling_masks': sampling_masks,
                    'bkgds': fake_bkgd,
                    'bkgd_idxs': bkgd_idxs,
                    # body data
                    'kp3d': kp_3d,
                    'gt_kp3d': kp_3d,
                    'bones': bone_poses,
                    'skts': skts,
                    'cyls': cylinder_params,
                    'rest_pose': rest_pose,
                    #'skts': test,
                    'betas': beta,
                    # camera
                    'c2ws': c2ws,
                    'focals': focals,
                    'ext_scale': ext_scale}

    write_to_h5py(h5_file, surreal_dict)

class SurrealDataset(BaseH5Dataset):
    '''
    Images/cameras are organized as the following:
    imgs: shape (N_cams, N_kps, ...)
    c2ws: shape (N_cams, N_kps, ...)

    To get the camera/view id, simply divide the index by N_kps.
    And to get the kp id, take modulo of the index
    '''
    #render_skip = 1
    #N_render = 9
    render_skip = 1
    N_render = 15

    rand_kps = {'230': 'data/surreal/surreal_rand_230.npy',
                '400': 'data/surreal/surreal_rand_400.npy',
                }

    def __init__(self, *args, N_rand_kps=None, N_cams=None, **kwargs):
        # TODO: handle rand kps cases
        self._N_rand_kps = N_rand_kps

        self._N_kps = None
        if N_rand_kps is not None:
            self._N_kps = int(N_rand_kps.split('_')[-1])
        self._N_cams = N_cams

        super(SurrealDataset, self).__init__(*args, **kwargs)

    def init_meta(self):

        if self.split == 'val':
            self.h5_path = self.h5_path.replace('train_h5py', 'val_h5py')

        super(SurrealDataset, self).init_meta()
        N_total_cams = len(self.c2ws) // len(self.kp3d)
        N_total_kps = len(self.kp3d)


        # get the right numbers of kps/cameras
        # NOTE: assume that we use the same set of kps for all views,
        #       and the cameras data are arranged as (N_cams, N_kps)
        if self._N_kps is None:
            self._N_kps = N_total_kps
        if self._N_cams is None:
            self._N_cams = N_total_cams

        #TODO: temporarily this
        if self.split == 'val':
            self._idx_map = np.load('data/surreal/surreal_val_idxs.npy')[0::2]
            return

        if self._N_kps == N_total_kps and self._N_cams == N_total_cams:
            return

        # create _idx_map if otherwise
        # TODO: hard-coded for now
        if self._N_rand_kps is None:
            selected_kps = np.arange(N_total_kps)
        else:
            selected_kps = np.unique(np.load(self.rand_kps[self._N_rand_kps]))
        selected_cams = np.array([0, 3, 6])
        self._idx_map = np.concatenate([selected_kps + N_total_kps * c for c in selected_cams])

    def get_kp_idx(self, idx, q_idx):
        '''
        idx: the actual index(s) for the full .h5 dataset
        q_idx: the 'queried' index(s) received from the sampler,
               may not coincide with idx.
        '''
        return idx % len(self.kp3d), q_idx % self._N_kps

    def get_cam_idx(self, idx, q_idx):
        '''
        idx: the actual index(s) for the full .h5 dataset
        q_idx: the 'queried' index(s) received from the sampler,
               may not coincide with idx.
        '''
        # NOTE: we already expand cameras data to the same shape of images,
        #       so no special treament needed on idx
        return idx, q_idx // self._N_kps

    def get_meta(self):
        data_attrs = super(SurrealDataset, self).get_meta()
        data_attrs['n_views'] = self._N_cams
        return data_attrs

if __name__ == "__main__":
    print("process data")

    path_to_surreal = "data/surreal/"
    process_surreal_data(os.path.join(path_to_surreal, "surreal_val_h5py.h5"),
                         os.path.join(path_to_surreal, "surreal_val"))
