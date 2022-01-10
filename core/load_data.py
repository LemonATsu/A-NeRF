import os
import cv2
import math
from torch.utils.data import DataLoader

from .dataset import *
from .load_surreal import SurrealDataset
from .load_h36m import H36MDataset
from .load_mixamo import MixamoDataset
from .load_perfcap import MonoPerfCapDataset
from .load_zju import ZJUMocapDataset
from .utils.skeleton_utils import rotate_y, rotate_x, rotate_z
from .utils.skeleton_utils import SMPLSkeleton, Mpi3dhpSkeleton, CanonicalSkeleton

DATASET_SKELETON_MAP = {"3dhp": SMPLSkeleton,
                        "surreal": SMPLSkeleton,
                        "h36m": SMPLSkeleton,
                        "zju": SMPLSkeleton,
                        "mixamo": SMPLSkeleton,
                        "perfcap": SMPLSkeleton,}

DATASET_CATALOG = {
    'h36m': {
        'S9': 'data/h36m/S9_processed_h5py.h5',
        'S9c': 'data/h36m/h36m_full/S9_60457274_processed_h5py.h5',
        'S11': 'data/h36m/S11_processed_h5py.h5',
        'S11c': 'data/h36m/h36m_full/S11_60457274_processed_h5py.h5',
    },
    'perfcap': {
        'weipeng': 'data/MonoPerfCap/Weipeng_outdoor/Weipeng_outdoor_processed_h5py.h5',
        'nadia': 'data/MonoPerfCap/Nadia_outdoor/Nadia_outdoor_processed_h5py.h5',
    },
    'surreal': {
        'female': 'data/surreal/surreal_train_h5py.h5',
    },
    'mixamo': {
        'james': 'data/mixamo/James_processed_h5py.h5',
        'archer': 'data/mixamo/Archer_processed_h5py.h5',
    },
    'zju': {k: f'data/zju_mocap/{k}_train_h5py.h5' for k in ['315', '377', '386', '387', '390', '392', '393',
                                                             '394']
    },
}

def generate_bullet_time(c2w, n_views=20, axis='y'):
    if axis == 'y':
        rotate_fn = rotate_y
    elif axis == 'x':
        rotate_fn = rotate_x
    elif axis == 'z':
        rotate_fn = rotate_z
    else:
        raise NotImplementedError(f'rotate axis {axis} is not defined')

    y_angles = np.linspace(0, math.radians(360), n_views+1)[:-1]
    c2ws = []
    for a in y_angles:
        c = rotate_fn(a) @ c2w
        c2ws.append(c)
    return np.array(c2ws)

def generate_bullet_time_kp(kp, n_views=20):
    kp = kp - kp[:1, :]
    y_angles = np.linspace(0, math.radians(360), n_views+1)[:-1]
    kps = []
    for a in y_angles:
        k = kp @ rotate_y(a)[:3, :3]
        kps.append(k)
    return np.array(kps)

def load_data(args):

    dataset = get_dataset(args)
    # Main loop controls the iteration, so simply set N_iter to something > args.n_iters
    sampler = RayImageSampler(dataset, N_images=args.N_sample_images,
                              N_iter=args.n_iters + 10)
    # initialize dataloader
    dataloader = DataLoader(dataset, batch_sampler=sampler,
                            num_workers=args.num_workers, pin_memory=True,
                            collate_fn=ray_collate_fn)
    data_attrs = dataset.get_meta()
    render_data = dataset.get_render_data()

    return dataloader, render_data, data_attrs


def get_dataset(args):

    # should specify at least one dataset.
    # assume all subjects are from the same dataset when only have one dataset
    subject, dataset_type = args.subject, args.dataset_type
    assert len(subject) >= len(dataset_type)
    if len(subject) > len(dataset_type):
        assert len(dataset_type) == 1
        dataset_type = dataset_type * len(subject)

    datasets = []
    N_samples = args.N_rand // args.N_sample_images
    N_nms = N_samples * args.P_nms
    assert N_samples <= args.N_rand, 'N_sample_images needs to be smaller than N_rand!'
    for i, (d, s) in enumerate(zip(dataset_type, subject)):
        datasets.append(get_dataset_from_catalog(args, N_samples, d, s, N_nms))

    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = ConcatH5Dataset(datasets)

    if args.use_temp_loss:
        dataset = TemporalDatasetWrapper(dataset)

    return dataset


def get_dataset_from_catalog(args, N_samples, dataset_type, subject=None, N_nms=0):

    split = 'full' if not args.use_val else 'train'

    shared_kwargs = {'N_samples': N_samples,
                     'split': split,
                     'mask_img': args.mask_image,
                     'patch_size': args.patch_size,
                     'subject': subject,
                     'N_nms': N_nms,
                     'multiview': args.multiview,}
    refined_kwargs = {'load_refined': args.load_refined}

    path = DATASET_CATALOG[dataset_type][subject]
    if dataset_type == 'h36m':
        dataset = H36MDataset(path, **shared_kwargs, **refined_kwargs)
    elif dataset_type == 'perfcap':
        dataset = MonoPerfCapDataset(path, **shared_kwargs, **refined_kwargs)
    elif dataset_type == 'mixamo':
        dataset = MixamoDataset(path, **shared_kwargs, **refined_kwargs)
    elif dataset_type == 'surreal':
        shared_kwargs['split'] = 'train'
        dataset = SurrealDataset(path, N_cams=args.N_cams, N_rand_kps=args.rand_train_kps,
                                 **shared_kwargs)
    elif dataset_type == 'zju':
        dataset = ZJUMocapDataset(path, **shared_kwargs)
    else:
        raise NotImplementedError(f'Dataset {dataset_type} is not implemented!')
    return dataset

