import os
import cv2
import glob
import h5py
import math
import torch
import imageio
import numpy as np

from .dataset import PoseRefinedDataset
from .process_spin import process_spin_data, write_to_h5py, read_spin_data
from .load_surreal import dilate_masks
from .utils.skeleton_utils import get_smpl_l2ws, rotate_y, skeleton3d_to_2d, create_kp_mask

from collections import OrderedDict

def extract_background(data_path, subject="S9"):
    import deepdish as dd
    mask_h5 = os.path.join(data_path, f"{subject}_mask_fixed.h5")

    cameras = ['54138969',  '55011271',  '58860488',  '60457274']
    chair_seqs = ['Sitting-', 'Eating-', 'Phoning-', 'Smoking-']

    # handling mask
    mask_data = dd.io.load(mask_h5)
    mask_img_path = mask_data['index']
    H = W = mask_data['masks'].shape[-2]

    bkgds = np.zeros((len(cameras), H, W, 3), dtype=np.float32)
    mask_cnts = np.zeros((len(cameras), H, W, 1), dtype=np.float32)

    for i in range(len(mask_img_path)):
        img_path = mask_img_path[i]

        has_chair = False
        for chair_seq in chair_seqs:
            if chair_seq in img_path:
                has_chair = True

        if has_chair:
            continue

        # read and append img
        img = imageio.imread(os.path.join(data_path, img_path))
        if img.shape[0] != H:
            # this camera has resolution 1002x1000, while others are 1000x1000
            img = img[1:-1, ...]

        cam_idx = None
        for e, camera in enumerate(cameras):
            if camera in img_path:
                cam_idx = e

        if cam_idx is None:
            raise ValueError('Camera not found!')

        # Calculate bkgd
        mask = mask_data['masks'][i]
        bkgds[cam_idx] = bkgds[cam_idx] + (img / 255.) * (1 - mask)
        mask_cnts[cam_idx] = mask_cnts[cam_idx] + (1 - mask)

    bkgds = ((bkgds / np.maximum(mask_cnts, 1)) * 255.).astype(np.uint8)
    np.save(os.path.join(data_path, f"{subject}_clean_bkgds_.npy"), bkgds)
    return bkgds

def extract_chairs_background(data_path, subject="S9"):
    import deepdish as dd
    mask_h5 = os.path.join(data_path, f"{subject}_mask_fixed.h5")

    cameras = ['54138969',  '55011271',  '58860488',  '60457274']
    chair_seqs = ['Sitting-', 'Eating-', 'Phoning-', 'Smoking-']

    # handling mask
    mask_data = dd.io.load(mask_h5)
    mask_img_path = mask_data['index']
    H = W = mask_data['masks'].shape[-2]

    bkgds = [[] for i in range(len(cameras))]

    for i in range(len(mask_img_path)):
        img_path = mask_img_path[i]

        has_chair = False
        for chair_seq in chair_seqs:
            if chair_seq in img_path:
                has_chair = True

        if not has_chair:
            continue

        # read and append img
        img = imageio.imread(os.path.join(data_path, img_path))
        if img.shape[0] != H:
            # this camera has resolution 1002x1000, while others are 1000x1000
            img = img[1:-1, ...]

        cam_idx = None
        for e, camera in enumerate(cameras):
            if camera in img_path:
                cam_idx = e

        if cam_idx is None:
            raise ValueError('Camera not found!')

        # Calculate bkgd
        mask = mask_data['masks'][i].astype(np.uint8)
        bkgd = img
        bkgds[cam_idx].append(bkgd)

    bkgds = np.array([np.median(bkgd, axis=0) for bkgd in bkgds]).astype(np.uint8)
    np.save(os.path.join(data_path, f"{subject}_chair_bkgds_.npy"), bkgds)
    return bkgds

def process_h36m_data(data_path, subject="S9", ext_scale=0.001,
                      res=1.0, bbox_res=224, extend_iter=2, val_cam='54138969', camera_name=None):
    '''
    :param data_path: path to h3.6m dataset root
    :param subject: subject directory
    :param ext_scale: to scale human poses and camera location
    :param bbox_res: resolution of bounding box when running the pose estimator
    :param extend_iter: extend mask to obtain sampling mask
    :param val_cam: use the data from this camera for validation
    '''

    import deepdish as dd
    if camera_name is None:
        spin_pickle = os.path.join(data_path, f"{subject}_SPIN_rect_output-maxmin.h5")
    elif subject != 'S1':
        spin_pickle = os.path.join(data_path, f"{subject}-camera=[{camera_name}]-subsample=5.h5")
    else:
        spin_pickle = os.path.join(data_path, f"{subject}-camera=[{camera_name}]-subsample=1.h5")


    bkgds = np.load(os.path.join(data_path, f"{subject.replace('s', '')}_clean_bkgds.npy"))
    chair_bkgds = np.load(os.path.join(data_path, f"{subject.replace('s', '')}_chair_bkgds.npy"))
    bkgds = np.concatenate([bkgds, chair_bkgds], axis=0)

    # handling mask if camera is not specified (since it's more manageable)
    if camera_name is None:
        mask_h5 = os.path.join(data_path, f"{subject}_mask_deeplab_crop.h5")
    else:
        mask_h5 = os.path.join(data_path, f"{subject}_{camera_name}_mask_deeplab_crop.h5")

    print(f"Load mask from {mask_h5}")
    mask_data = dd.io.load(mask_h5)
    mask_img_path = mask_data['index']
    mask_data['masks'] = mask_data['masks'].astype(np.uint8)

    if len(mask_data["masks"].shape) <= 3:
        mask_data['masks'] = mask_data['masks'][..., None]

    max_val = mask_data['masks'].max()
    if max_val > 1:
        mask_data['masks'][mask_data['masks'] < 2] = 0
        mask_data['masks'][mask_data['masks'] >= 2] = 1

    # Do not check H because one of the camera has weird H
    H = W = mask_data['masks'].shape[-2]

    if 'res' in mask_data:
        res = mask_data['res']

    if res != 1.0:
        H = int(H / res)
        W = int(W / res)
        new_W, new_H = int(res * W), int(res * H)
        resized_bkgds = []
        for bkgd in bkgds:
            bkgd = cv2.resize(bkgd, (new_W, new_H), interpolation=cv2.INTER_AREA)
            resized_bkgds.append(bkgd)
        resized_bkgds = np.array(resized_bkgds)
        bkgds = resized_bkgds

    processed_est = read_spin_data(spin_pickle, ext_scale,
                                   img_res=H,
                                   bbox_res=bbox_res)

    if res != 1.0:
        processed_est['focals'] = processed_est['focals'] * res
        print(f"resolution {res}")


    # expand mask to regions that cover the estimated skeleton.
    mask_H, mask_W = mask_data['masks'][0].shape[:2]
    kp2ds = skeleton3d_to_2d(processed_est['kp3d'], processed_est['c2ws'],
                             mask_H, mask_W, processed_est['focals'])
    sampling_masks = dilate_masks(mask_data['masks'][..., 0], extend_iter)[..., None]
    mask_data['sampling_masks'] = sampling_masks

    imgs = []
    train_idxs, val_idxs = [] , []
    cam_idxs = []
    # These are the camera used in h36m
    if subject != 'S1':
        cameras = ['54138969',  '55011271',  '58860488',  '60457274']
    else:
        cameras = ['60457274']
    chair_seqs = ['Sitting-', 'Eating-', 'Phoning-', 'Smoking-']
    # * for views with chairs ...
    cam_idxs = []

    img_paths = processed_est['img_path']
    for i in range(len(img_paths)):
        img_path = img_paths[i]
        if (i + 1) % 100 == 0:
            print(f"{i}/{len(img_paths)}")

        # find background
        offset = 0
        for chair_seq in chair_seqs:
            if chair_seq in img_path:
                offset = offset + len(cameras)

        cam_idx = None
        for e, camera in enumerate(cameras):
            if camera in img_path:
                cam_idx = e + offset
                break
        cam_idxs.append(cam_idx)

        # read and append img
        img = imageio.imread(os.path.join(data_path, img_path))
        if img.shape[0] != H:
            # this camera has resolution 1002x1000, while others are 1000x1000
            img = img[1:-1, ...]
        # TODO: verify nm flag
        # mask it out because we can have nice mask now
        #img = img * mask_data['masks'][i] + (1 - mask_data['masks'][i]) * bkgds[cam_idx]
        if res != 1.0:
            new_W, new_H = int(res * W), int(res * H)
            img = cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_AREA)
        imgs.append(img)

    data = {'imgs': np.array(imgs), #data["imgs"],
            'bkgd_idxs': np.array(cam_idxs),
            'train_idxs': np.array(train_idxs),
            'val_idxs': np.array(val_idxs),
            'bkgds': bkgds,
            'img_paths': mask_img_path,
            **mask_data, **processed_est}

    if camera_name is None:
        h5_name = f"{subject}_processed_h5py.h5"
    else:
        h5_name = f"{subject}_{camera_name}_processed_h5py.h5"
    print(f"WRITING H5 FILE TO {h5_name}, image size: {imgs[-1].shape}")
    write_to_h5py(os.path.join(data_path, h5_name), data)

    return data

def find_motion_set(img_paths):
    set_dict = OrderedDict()
    set_cnt = OrderedDict()
    set_idxs = []
    for p in img_paths:
        set_name = p.split(b'/')[1]
        if set_name not in set_dict.keys():
            set_dict[set_name] = len(set_dict)
            set_cnt[set_name] = 1
        else:
            set_cnt[set_name] += 1
        set_idxs.append(set_dict[set_name])

    set_idxs = np.array(set_idxs)
    return set_dict, set_cnt, set_idxs

def create_kp_mapping(set_dict, set_cnt, set_idxs, n_views=2):
    """
    map from multiple to a single view
    """
    assert n_views % 2 == 0
    kp_map = []
    unique_indices = []
    acc_idx = 0
    acc_unique = 0
    for set_name in set_dict.keys():
        num_kp_original = set_cnt[set_name]
        num_kps = num_kp_original // n_views
        kp_off = np.arange(num_kp_original) % num_kps
        k_map = kp_off + acc_idx
        unique_idx = kp_off + acc_unique

        kp_map.append(k_map)
        unique_indices.append(unique_idx)

        acc_idx += num_kps
        acc_unique += num_kp_original
    return np.concatenate(kp_map), np.unique(np.concatenate(unique_indices))

def get_temporal_validity(img_paths):
    valid = np.ones((len(img_paths),))
    seq_map = np.zeros((len(img_paths),), dtype=np.int32)
    seq_cnt = 0
    for i, img_path in enumerate(img_paths):
        if i == 0:
            valid[i] = 0
            continue
        prev_path = img_paths[i-1]
        if os.path.dirname(prev_path) != os.path.dirname(img_path):
            #print(f"invalid: prev {prev_path}, cur {img_path}")
            valid[i] = 0
            seq_cnt += 1
        seq_map[i] = seq_cnt
    return valid, seq_map

def map_data_to_n_views(img_paths, kp3d, bones, rest_pose,
                        n_views=4, avg_kps=True):

    def set_root(k, k_unique, k_map, root_id=0):
        root = k[:, root_id:root_id+1]
        if not avg_kps:
            other_parts = k_unique[k_map, root_id+1:]
        else:
            print("avg kps")
            other_parts = np.zeros_like(k_unique[:, root_id+1:])
            for i, k_idx in enumerate(k_map):
                other_parts[k_idx] = other_parts[k_idx] + k[i, root_id+1:]
            other_parts = other_parts / float(n_views)
            other_parts = other_parts[k_map]
        return np.concatenate([root, other_parts], axis=1)

    set_dict, set_cnt, set_idxs = find_motion_set(img_paths)
    kp_map, kp_uidxs = create_kp_mapping(set_dict, set_cnt, set_idxs, n_views=n_views)


    unique_bones = bones[kp_uidxs]
    unique_kp3d = kp3d[kp_uidxs]

    bones = set_root(bones, unique_bones, kp_map)
    kp3d = set_root(kp3d, unique_kp3d, kp_map)

    # set skts properly for rendering
    l2ws = np.array([get_smpl_l2ws(bone, rest_pose, scale=1.) for bone in bones])
    # assume root is at 0
    l2ws[..., :3, -1] = l2ws[..., :3, -1] + kp3d[:, 0:1].copy()
    skts = np.array([np.linalg.inv(l2w) for l2w in l2ws])

    print(f"Data remapped to {n_views}. Note that root-rotation is not with the kp3d anymore.")

    return kp_map, kp_uidxs, kp3d, bones, skts

def generate_bullet_time(c2w, n_views=20):

    y_angles = np.linspace(0, math.radians(360), n_views+1)[:-1]
    c2ws = []
    for a in y_angles:
        c = rotate_y(a) @ c2w
        c2ws.append(c)
    return np.array(c2ws)

def save_masks(data_path, subject='S9'):
    import deepdish as dd
    data_h5 = os.path.join(data_path, f"{subject}_processed.h5")
    #data_h5 = os.path.join(data_path, f"S11_processed_.h5")
    data = dd.io.load(data_h5)
    img_paths = data["img_paths"]
    masks = data["masks"]

    for img_path, mask in zip(img_paths, masks):

        # S11/Directions-1/imageSequence/54138969/img_000001.jpg
        name_parts = img_path.split("/")
        mask_dir = os.path.join(*name_parts[:-1]).replace('imageSequence', 'Mask')
        os.makedirs(os.path.join(data_path, mask_dir), exist_ok=True)
        #mask[]
        print(np.unique(mask))
        imageio.imwrite(os.path.join(data_path, mask_dir, name_parts[-1]), (mask * 255).astype(np.uint8))

class H36MDataset(PoseRefinedDataset):

    # define the attribute for rendering data
    render_skip = 80
    N_render = 15

    refined_paths = {
        'S9': ('data/h36m/S9_refined_64.tar', True),
        'S11': ('data/h36m/S11_refined_64.tar', True),
    }

    def init_meta(self):
        dataset = h5py.File(self.h5_path, 'r', swmr=True)
        img_paths = dataset['img_paths'][:]

        val_sets = ['Greeting-', 'Walking-', 'Posing-']

        self._idx_map = None
        if self.subject.endswith('c'):
            idxs = []
            for i, p in enumerate(img_paths):
                seq = p.decode().split('/')[1]
                if seq.endswith('-1'):
                    idxs.append(i)
            self._idx_map = np.array(idxs)
        elif self.split != 'full':
            train_idxs, val_idxs = [], []

            # check if an image blongs to validation set
            for i, p in enumerate(img_paths):
                seq = p.decode().split('/')[1]
                is_val = False
                for v in val_sets:
                    if seq.startswith(v):
                        is_val = True
                        break

                if is_val:
                    val_idxs.append(i)
                else:
                    train_idxs.append(i)
            train_idxs = np.array(train_idxs)
            val_idxs = np.array(val_idxs)
            if self.split == 'train':
                self._idx_map = train_idxs
            elif self.split == 'val':
                self._idx_map = val_idxs
            else:
                raise NotImplementedError(f'Split {self.split} is undefined!')

        dataset.close()
        super(H36MDataset, self).init_meta()

    def _load_multiview_pose(self, dataset, kp3d, bones, skts, cyls):
        # cyls usually doesn't change too much
        rest_pose = dataset['rest_pose'][:]
        img_paths = dataset['img_paths'][:]
        # set self.kp_map and self.kp_uidxs
        kp_map, kp_uidxs, kp3d, bones, skts = map_data_to_n_views(img_paths, kp3d, bones, rest_pose)
        self.kp_map = kp_map
        self.kp_uidxs = kp_uidxs

        return kp3d, bones, skts, cyls

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Arguments for masks extraction')
    parser.add_argument("-s", "--subject", type=str, default="S9",
                        help='subject to extract')
    parser.add_argument("-b", "--base_path", type=str, default="data/h36m/",
                        help='path to dataset')
    parser.add_argument("-c", "--camera_id", type=int, default=None,
                        help='camera to extract')
    parser.add_argument("-i", "--extend_iter", type=int, default=2,
                        help='extend foreground mask a bit to get sampling masks')
    args = parser.parse_args()
    data_path = args.base_path
    subject = args.subject
    extend_iter = args.extend_iter

    camera = None
    cameras = ["54138969", "55011271", "58860488", "60457274"]
    if args.camera_id is not None:
        camera = cameras[args.camera_id]
    data = process_h36m_data(data_path, subject, camera_name=camera, extend_iter=extend_iter)
