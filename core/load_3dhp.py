# TODO Rework this whole file for new dataloader format
import os
import glob
import torch
import imageio
import numpy as np
import pickle as pkl
from .process_spin import process_spin_data
from .load_surreal import dilate_masks
from collections import OrderedDict

def read_3dhp_spin_data(data_path, subject='S1', ext_scale=0.001, bbox_res=224):

    res_map = {
        'S1': (768, 768),
        'S2': (768, 768),
        'S3': (768, 768),
        'S4': (768, 768),
        'S5': (768, 1365),
        'S6': (768, 1365),
    }

    spin_data = dd.io.load(data_path)
    img_paths = []
    img_idxs = []

    for i, p in enumerate(spin_data['img_path']):
        if subject in p:
            img_paths.append(p)
            img_idxs.append(i)
    img_idxs = np.array(img_idxs)

    betas = torch.tensor(spin_data['pred_betas'][img_idxs]).cpu()
    joints = torch.cat([spin_data['pred_output'][i].joints for i in img_idxs]).cpu()
    rot_mats = torch.tensor(spin_data['pred_rot_mat'][img_idxs]).cpu()
    bboxes = torch.tensor(spin_data['bbox_params'][img_idxs]).cpu()
    try:
        cameras = torch.tensor(spin_data['pred_camera'][img_idxs]).cpu()
    except:
        cameras = torch.tensor(spin_data['pred_cam'][img_idxs]).cpu()
    img_res = res_map[subject]

    processed_ret = process_spin_data(betas, cameras, joints, rot_mats,
                                      bboxes, res=img_res, resized_res=bbox_res,
                                      ext_scale=ext_scale, scale_rest_pose=True,
                                      )

    processed_ret['img_path'] = np.array(img_paths)
    processed_ret['hw'] = img_res
    if subject != 'S5' and subject != 'S6':
        processed_ret['gt_kp3d'] = spin_data['pose_3d'][img_idxs]
    else:
        h5_pose = os.path.join(os.path.dirname(data_path), 'MPI_SPIN_rect_output-maxmin.h5')
        pose_path = dd.io.load(h5_pose, '/img_path')
        pose_3d = dd.io.load(h5_pose, '/pose_3d')
        idxs = []
        for i, p in enumerate(pose_path):
            if subject in p:
                idxs.append(i)
        # for debugging
        # subject_idx = np.array(pose_path)[idxs]
        processed_ret['gt_kp3d'] = pose_3d[idxs]
    processed_ret['betas'] = spin_data['pred_betas'][img_idxs]
    processed_ret['bboxes'] = bboxes.numpy()

    return processed_ret

def extract_background(img_paths, data_path='data/mpi_3dhp/', subject='S1'):
    subject_paths = [p for p in img_paths if subject in p]

    imgs = []
    for p in subject_paths:
        img = imageio.imread(os.path.join(data_path, p))
        imgs.append(img)
    imgs = np.array(imgs)
    bkgd = np.median(imgs, axis=0)

    return bkgd

# TODO: rework this for new dataloader format
def process_3dhp_data(data_path, subject='S1', ext_scale=0.001, bbox_res=224,
                      extend_iter=5):
    '''
    :param data_path: path to h3.6m dataset root
    :param subject: subject directory
    :param ext_scale: to scale human poses and camera location
    :param bbox_res: resolution of bounding box when running the pose estimator
    :param extend_iter: extend mask to obtain sampling mask
    '''

    #spin_h5 = os.path.join(data_path, "MPI_SPIN_rect_output.h5")
    if subject != 'S5' and subject != 'S6':
        spin_h5 = os.path.join(data_path, "MPI_SPIN_rect_output-maxmin.h5")
        bkgd = imageio.imread(os.path.join(data_path, f"{subject}_bkgd.png"))
    else:
        spin_h5 = os.path.join(data_path, "mpi_3dhp", "3DHP-S5S6.h5")
        bkgd = imageio.imread(os.path.join(data_path, "mpi_3dhp", f"{subject}_bkgd.png"))
    print(f"Read from {spin_h5}")


    """
    mask_h5 = os.path.join(data_path, f"{subject}_masks.h5")
    masks = dd.io.load(mask_h5)['masks']
    masks = masks[..., None]
    # mask is not perfect, dilate it a bit
    masks = dilate_masks(masks, extend_iter=2)[..., None]
    # dilate it more to get sampling mask
    sampling_masks = dilate_masks(masks, extend_iter=5)[..., None]
    H, W = masks.shape[1], masks.shape[2]
    """


    #bkgd = imageio.imread(os.path.join(data_path, f"{subject}_bkgd.png"))
    processed_est = read_3dhp_spin_data(spin_h5, subject, ext_scale, bbox_res=bbox_res)

    imgs, masks = [], []
    img_paths = processed_est['img_path']
    cam_idxs = np.zeros((len(img_paths),))
    for i in range(len(img_paths)):
        img = imageio.imread(os.path.join(data_path, img_paths[i]))
        mask = imageio.imread(os.path.join(data_path, img_paths[i].replace("/imageSequence/", "/masks/")))
        mask[mask < 2] = 0
        mask[mask >= 2] = 1
        #mask = masks[i] // 255
        #img = img * mask + (1 - mask) * bkgd
        imgs.append(img)
        masks.append(mask)
    masks = np.array(masks)[..., None]
    sampling_masks = dilate_masks(masks[..., 0], extend_iter=2)[..., None]

    data = {'imgs': np.array(imgs).astype(np.uint8),
            'bkgd_idxs': np.array(cam_idxs),
            'train_idxs': np.arange(len(imgs)),
            'val_idxs': np.array(len(imgs)),
            'bkgds': bkgd[None],
            'masks': masks,
            'sampling_masks': sampling_masks,
            **processed_est}
    h5_name = f"{subject}_processed.h5"
    dd.io.save(os.path.join(data_path, h5_name), data)

def load_3dhp_data(data_path, subject='S1'):

    data_h5 = os.path.join(data_path, f"{subject}_processed.h5")
    print(f"Reading from {data_h5}")
    data = dd.io.load(data_h5)
    imgs = data["imgs"]
    H, W = imgs.shape[1], imgs.shape[2]
    temporal_validity = np.ones((len(imgs),))
    temporal_validity[0] = 0

    if subject == "S2":
        data["bkgds"][...] = 0 # background tone is too close to the subject..
    if subject == "S5":
        data["c2ws"][..., :3, -1] /= 0.82
        pass
    if subject == "S6":
        data["c2ws"][..., :3, -1] /= 0.82
        pass


    train_kp_dict = {
        "bone_poses": data["bones"],
        "cyls": data["cyls"],
        "img_pose_indices": np.arange(len(data["imgs"])),
        "kp3d": data["kp3d"],
        "sampling_masks": data["sampling_masks"],
        "skts": data["skts"],
        "rest_pose": data["rest_pose"],
        "n_views": len(data["imgs"]),
        "gt_kp3d": data["gt_kp3d"],
        "gt_skts": data["skts"].copy(),
        "gt_bone_poses": data["bones"].copy(),
        "temporal_validity": temporal_validity,
        "betas": data["betas"].mean(0)[None],
    }


    train_data = {
        "c2ws": data["c2ws"],
        "fg_masks": data["masks"],
        "hwf": (H, W, data["focals"]),
        "imgs": data["imgs"],
        "keypoint_dict": train_kp_dict,
        "bg_imgs": data["bkgds"],
        "bg_indices": data["bkgd_idxs"].astype(np.int32),
        "ext_scale": data["ext_scale"],
        "beta_full": data["betas"],
    }

    val_idxs = np.arange(len(data["imgs"]))[::9]
    val_kp_dict = {
        "bone_poses": data["bones"][val_idxs],
        "cyls": data["cyls"][val_idxs],
        "img_pose_indices": val_idxs,
        "kp3d": data["kp3d"][val_idxs],
        "sampling_masks": data["sampling_masks"][val_idxs],
        "skts": data["skts"][val_idxs],
        "rest_pose": data["rest_pose"],
        "n_views": len(val_idxs),
    }

    val_data = {
        "c2ws": data["c2ws"][val_idxs],
        "fg_masks": data["masks"][val_idxs],
        "hwf": (H, W, data["focals"][val_idxs]),
        "imgs": data["imgs"][val_idxs],
        "keypoint_dict": val_kp_dict,
        "bg_imgs": data["bkgds"],
        "bg_indices": data["bkgd_idxs"][val_idxs].astype(np.int32),
        "ext_scale": data["ext_scale"],
    }

    return train_data, val_data


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Arguments for masks extraction')
    parser.add_argument("-s", "--subject", type=str, default="S1",
                        help='subject to extract')
    parser.add_argument("-b", "--base_path", type=str, default="data/mpi_3dhp/",
                        help='path to dataset')

    args = parser.parse_args()
    data_path = args.base_path
    subject = args.subject
    #img_paths = glob.glob(os.path.join(data_path, f"data/test/{subject}/Seq1/imageSequence/video_0/*.jpg"))
    #import pdb; pdb.set_trace()
    #print
    process_3dhp_data(args.base_path, args.subject)
