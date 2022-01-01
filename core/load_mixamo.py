import os
import h5py
import imageio
import numpy as np
import pickle as pkl
from collections import OrderedDict

from .dataset import PoseRefinedDataset
from .load_h36m import read_spin_data
from .load_surreal import dilate_masks
from .process_spin import write_to_h5py
from .utils.skeleton_utils import skeleton3d_to_2d, create_kp_mask

def process_mixamo_data(data_path, subject='Kachujin', ext_scale=0.001,
                        bbox_res=224, extend_iter=2):

    '''
    Assume that we have 4 cameras for each case
    '''
    n_cam = 4
    spin_data = read_spin_data(os.path.join(data_path, subject, f'{subject}.h5'),
                               ext_scale=ext_scale, img_res=1000,
                               bbox_res=bbox_res)
    img_paths = spin_data['img_path']
    mask_path = []

    imgs, masks, kp_idxs, cam_idxs = [], [], [], []
    seq_dict = OrderedDict()
    for i, p in enumerate(img_paths):
        if (i+1) % 500 == 0:
            print(f"{i+1}/{len(img_paths)}")
        name_parts = p.split('/')
        d = '/'.join(name_parts[:3])
        seq_name = name_parts[1]
        img_name = name_parts[-1]
        cam_idx = int(name_parts[2].split('_')[-1])
        kp_idx = int(img_name[5:-4])-1
        if seq_name not in seq_dict:
            seq_dict[seq_name] = []
        seq_dict[seq_name].append(i)
        m = f'{d}/Masks/{img_name}'
        img = imageio.imread(os.path.join(data_path, p))[..., :3]
        mask = imageio.imread(os.path.join(data_path, m))[..., :1]
        mask[mask < 2] = 0
        mask[mask >= 2] = 1

        imgs.append(img*mask + (1-mask)* np.ones_like(img)*255)
        masks.append(mask)
        kp_idxs.append(kp_idx)
        cam_idxs.append(cam_idx)

    gt_kps = []
    joint_names = None
    # extract ground truth poses
    for k in seq_dict.keys():
        metadata = pkl.load(open(os.path.join(data_path, subject, k, "Camera_0", "metadata.pickle"), "rb"))
        gt_poses = metadata['gt_pose']
        if joint_names is None: # initialize this to ensure the keypoints are read in the same way
            joint_names = [k for k in gt_poses[0].item().keys()]
        for pose in gt_poses:
            kp = np.array([pose.item()[j] for j in joint_names])
            gt_kps.append(kp)

    kp_idxs = np.array(kp_idxs)
    cam_idxs = np.array(cam_idxs)
    # now, remap kp_idxs.
    i = 0
    for k in seq_dict.keys():
        seq_len = len(seq_dict[k]) # note: we have only // n_cam poses.
        # Offset the kp_idxs by #kps of previous sequences
        kp_idxs[i*n_cam:i*n_cam+seq_len] += i
        print(f"{i*n_cam}:{i*n_cam+seq_len}")
        i += seq_len // n_cam


    # expand mask to regions that cover the estimated skeleton.
    H, W = imgs[0].shape[:2]
    kp2ds = skeleton3d_to_2d(spin_data["kp3d"], spin_data["c2ws"], H, W, spin_data["focals"])
    """
    sampling_masks = []
    for i, mask in enumerate(masks):
        sampling_mask = create_kp_mask(mask, kp2ds[i], H, W)
        sampling_masks.append(sampling_mask)
    sampling_masks = np.array(sampling_masks)
    sampling_masks = dilate_masks(sampling_masks, extend_iter=extend_iter)
    """
    # and then expand the sampling region!
    sampling_masks = dilate_masks(masks, extend_iter=extend_iter)

    #print(f"Writing to path {subject}_processed_i2.h5")
    print(f"Writing to path {subject}_processed_h5py.h5")
    data = {
        'img_paths': np.array(img_paths),
        'imgs': np.array(imgs),
        'masks': np.array(masks),
        'sampling_masks': np.array(sampling_masks)[..., None],
        'kp_idxs': kp_idxs,
        'cam_idxs': cam_idxs,
        #'seq_dict': seq_dict,
        #'joint_names': joint_names,
        'gt_poses': np.array(gt_kps).astype(np.float32),
        **spin_data
    }
    write_to_h5py(os.path.join(data_path, f'{subject}_processed_h5py.h5'), data)

    return data

def extract_data_from_camera(data, camera_id, val_camera_id):

    img_paths = data['img_path']
    selected_idx = []
    val_selected_idx = []
    for i, img_path in enumerate(img_paths):
        if camera_id in img_path:
            selected_idx.append(i)
        elif val_camera_id in img_path:
            val_selected_idx.append(i)

    full_length = len(img_paths)
    N_train, N_val = len(selected_idx), len(val_selected_idx)
    # pull out training and validation set at once.
    selected_idx = selected_idx + val_selected_idx

    img_paths = np.array(img_paths)[selected_idx]
    data['img_path'] = img_paths

    for k in data:
        if hasattr(data[k], '__len__') and len(data[k]) == full_length:
            print(f"Get {k} subset")
            data[k] = data[k][selected_idx]
        else:
            print(f"{k} stay the same.")

    return data, N_train, N_val

def get_temporal_validity(img_paths):
    valid = np.ones((len(img_paths),))
    seq_map = np.zeros((len(img_paths),), dtype=np.int32)
    seq_cnt = 0

    def get_num(name):
        base = os.path.splitext(os.path.basename(name))[0]
        return int(base.split(b'Image')[-1])

    for i, img_path in enumerate(img_paths):
        if i == 0:
            valid[i] = 0
            continue
        prev_path = img_paths[i-1]
        prev_num = get_num(prev_path)#os.path.basename(prev_path)
        curr_num = get_num(img_path)
        diff = abs(curr_num - prev_num) # invalid when image is not consecutive
        #print(f"current num: {curr_num}, previous num: {prev_num}")
        if (os.path.dirname(prev_path) != os.path.dirname(img_path)) or (diff > 1):
            #print(f"invalid: prev {prev_path}, cur {img_path}")
            valid[i] = 0
            seq_cnt += 1
        seq_map[i] = seq_cnt
    return valid, seq_map

class MixamoDataset(PoseRefinedDataset):

    render_skip = 40
    N_render = 15
    refined_paths = {
        'james': ('data/mixamo/james_refined.tar', True),
        'archer': ('data/mixamo/archer_refined.tar', True),
    }

    def _load_pose_data(self, dataset):
        kp3d, bones, skts, cyls = dataset['kp3d'][:], dataset['bones'][:], dataset['skts'][:], dataset['cyls'][:]
        if not self.load_refined:
            return kp3d, bones, skts, cyls
        # refinement is only done on selected poses
        r_kp3d, r_bones, r_skts, r_cyls = super(MixamoDataset, self)._load_pose_data(dataset)

        kp3d[self._idx_map] = r_kp3d
        bones[self._idx_map] = r_bones
        skts[self._idx_map] = r_skts
        cyls[self._idx_map] = r_cyls
        return kp3d, bones, skts, cyls



    def init_meta(self):
        dataset = h5py.File(self.h5_path, 'r', swmr=True)
        selected_idxs_path = self.h5_path.replace('processed_h5py.h5', 'selected.npy')
        self._idx_map = np.load(selected_idxs_path)
        self._idx_map = np.array(sorted(self._idx_map))

        super(MixamoDataset, self).init_meta()


        # set white bkgd manually
        self.bgs = np.ones((1, np.prod(self.HW), 3), dtype=np.uint8) * 255
        self.bg_idxs = np.zeros((dataset['imgs'].shape[0],), dtype=np.int64)
        self.has_bg = True
        self.temp_validity = get_temporal_validity(dataset['img_paths'][self._idx_map])[0]
        dataset.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Arguments for masks extraction')
    parser.add_argument("-s", "--subject", type=str, default="Kachujin",
                        help='subject to extract')
    args = parser.parse_args()
    subject = args.subject
    data_path = 'data/mixamo/'
    data = process_mixamo_data(data_path, extend_iter=2, subject=subject)
