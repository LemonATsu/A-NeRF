import os
import cv2
import h5py
import imageio
import numpy as np

from .dataset import PoseRefinedDataset
from .load_surreal import dilate_masks
from .process_spin import process_spin_data, write_to_h5py, read_spin_data
from .utils.skeleton_utils import *

def process_perfcap_data(data_path, subject='Weipeng_outdoor', ext_scale=0.001,
                         img_res=(1080, 1920), bbox_res=224, extend_iter=2):

    spin_data = read_spin_data(os.path.join(data_path, 'MonoPerfCap', f'MonoPerfCap-{subject}.h5'),
                               img_res=img_res, bbox_res=bbox_res)
    img_paths = spin_data['img_path']

    bkgd = imageio.imread(os.path.join(data_path, 'MonoPerfCap', f'{subject}/bkgd.png'))
    imgs, masks = [], []
    for i, img_path in enumerate(img_paths):
        img_path = os.path.join(data_path, img_path)
        mask_path = img_path.replace('/images/', '/masks/')

        img = imageio.imread(img_path)
        mask = imageio.imread(mask_path)[..., None]

        mask[mask < 2] = 0
        mask[mask >= 2]= 1

        #imgs.append(mask * img + (1 - mask) * bkgd)
        imgs.append(img)
        masks.append(mask)

    masks = np.array(masks)
    sampling_masks = dilate_masks(masks, extend_iter=extend_iter)[..., None]
    cam_idxs = np.arange(len(masks))
    kp_idxs = np.arange(len(masks))

    data = {
        'imgs': np.array(imgs),
        'masks': np.array(masks),
        'sampling_masks': sampling_masks,
        'kp_idxs': kp_idxs,
        'cam_idxs': cam_idxs,
        'bkgds': bkgd[None],
        'bkgd_idxs': np.zeros((len(masks),)),
        **spin_data,
    }
    h5_name = os.path.join(data_path, 'MonoPerfCap', f'{subject}/{subject}_processed_h5py.h5')
    print(f"Writing h5 file to {h5_name}")
    write_to_h5py(h5_name ,data, img_chunk_size=16)

class MonoPerfCapDataset(PoseRefinedDataset):
    n_vals = {'weipeng': 230, 'nadia': 327}

    # define the attribute for rendering data
    render_skip = 10
    N_render = 15

    refined_paths = {
        'weipeng': ('data/MonoPerfCap/Weipeng_outdoor/weipeng_refined.tar', True),
        'nadia': ('data/MonoPerfCap/Nadia_outdoor/nadia_refined.tar', True),
    }

    def init_meta(self):
        dataset = h5py.File(self.h5_path, 'r', swmr=True)

        train_idxs = np.arange(len(dataset['imgs']))

        self._idx_map = None
        if self.split != 'full':
            n_val = self.n_vals[self.subject]
            val_idxs = train_idxs[-n_val:]
            train_idxs = train_idxs[:-n_val]

            if self.split == 'train':
                self._idx_map = train_idxs
            elif self.split == 'val':
                self._idx_map = val_idxs
            else:
                raise NotImplementedError(f'Split {self.split} is not implemented.')

        self.temp_validity = np.ones(len(train_idxs))
        self.temp_validity[0] = 0
        dataset.close()
        super(MonoPerfCapDataset, self).init_meta()
        # the estimation for MonoPerfCap is somehow off by a small scale (possibly due do the none 1:1 aspect ratio)
        self.c2ws[..., :3, -1] /= 1.05


if __name__ == '__main__':
    #process_perfcap_data('data/', subject='Weipeng_outdoor')
    process_perfcap_data('data/', subject='Nadia_outdoor')
