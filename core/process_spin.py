import h5py
import torch
import numpy as np
import pickle as pkl
from scipy.spatial.transform import Rotation
from collections.abc import Iterable

from .utils.skeleton_utils import smpl_rest_pose, calculate_bone_length, get_smpl_l2ws,\
                                   get_kp_bounding_cylinder, swap_mat, SMPLSkeleton

# mapper to map joints
SMPL_JOINT_MAPPER = lambda joints: joints[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]]

def read_spin_data(data_path, ext_scale=0.001, img_res=1000, bbox_res=224):
    import deepdish as dd

    if data_path.endswith(".pkl"):
        spin_data = pkl.load(open(data_path, "rb"))
    else:
        spin_data = dd.io.load(data_path)

    img_paths = spin_data['img_path']
    betas = torch.tensor(spin_data['pred_betas']).cpu()
    joints = torch.cat([spin_data['pred_output'][i].joints for i in range(len(img_paths))]).cpu()
    rot_mats = torch.tensor(spin_data['pred_rot_mat']).cpu()
    bboxes = torch.tensor(spin_data['bbox_params']).cpu()
    try:
        cameras = torch.tensor(spin_data['pred_camera']).cpu()
    except:
        cameras = torch.tensor(spin_data['pred_cam']).cpu()

    processed_ret = process_spin_data(betas, cameras, joints, rot_mats,
                                      bboxes, res=img_res, resized_res=bbox_res,
                                      ext_scale=ext_scale, scale_rest_pose=True)

    processed_ret['img_path'] = img_paths
    if 'pose_3d' in spin_data:
        processed_ret['gt_kp3d'] = np.array(spin_data['pose_3d']).astype(np.float32)
    if 'selected_idx' in spin_data:
        processed_ret['selected_idx'] = spin_data['selected_idx']
    processed_ret['betas'] = betas.numpy()
    return processed_ret



def convert_crop_cam_to_orig_img_and_focal(cam, bbox, img_width, img_height,
                                           focal=5000., resized_width=224, resized_height=224,
                                           new_focal=None):
    """
    Borrow from VIBE
    """
    '''
    Convert predicted camera from cropped image coordinates
    to original image coordinates
    :param cam (ndarray, shape=(3,)): weak perspective camera in cropped img coordinates
    :param bbox (ndarray, shape=(4,)): bbox coordinates (c_x, c_y, h)
    :param img_width (int): original image width
    :param img_height (int): original image height
    :param focal: focal length of the predicted camera in cropped img coordinate
    :param resized_width (int): the size of the cropped img (which is resized to a fixed value)
    :param resized_height (int): the size of the cropped img (which is resized to a fixed value)
    :return:
    f: focal length in the original image
    tx, ty, cz: camera location (x, y, )
    '''
    # Note: we assume the bounding box to be a squared one here
    # Note: we only need to translate x and y in this case, as the camera is not moved along z-axis
    # calculate the z location as in SPIN and VIBE
    cz = 2 * focal / (resized_width * cam[:, 0])
    cx, cy, h = bbox[:, 0], bbox[:, 1], bbox[:, 2]
    hw, hh = img_width / 2., img_height / 2.

    # first, we can know the actual focal length by scaling the cropped focal length
    f = h / resized_width * focal

    # similarly, undo the bounding box scaling here
    # Note: cam[:, 0] = 2 * focal / (resized_width * cz)
    # so the following equation is equivalent to
    # 2 * focal / (resized_width * cz) * (h / img_width) = (h / resized_width) * (1/img_width) * (focal / cz)
    # basically, the first term undo the bounding box scaling
    sx = cam[:, 0] * (1. / (img_width / h))
    sy = cam[:, 0] * (1. / (img_height / h))

    # now, apply the scale to move the camera
    tx = ((cx - hw) / hw / sx) + cam[:, 1]
    ty = ((cy - hh) / hh / sy) + cam[:, 2]

    if new_focal is not None:
        print(f"old cz {cz}, focal {f}")
        cz = cz * new_focal / f
        f[:] = new_focal
        print(f"new cz {cz}, focal {f}")


    return np.stack([f, tx, ty, cz], axis=-1)

def get_keypoints_from_betas(betas, joints, rot_mats,
                             ext_scale=1.0, align_joint_idx=8,
                             ref_pose=smpl_rest_pose, scale_rest_pose=True,
                             mapper=SMPL_JOINT_MAPPER, gender='NEUTRAL'):
    '''
    betas: shape parameters
    joints: predicted joints from SPIN
    ext_scale: extrinsic scale to scale the keypoints
    align_joint_idx: index to the predicted joints to align the reconstructed human pose
    ref_pose: the reference pose for scaling
    scale_rest_pose: to make the pose the same scale as the reference pose
    '''
    from smplx import SMPL
    from torchgeometry import rotation_matrix_to_angle_axis

    with torch.no_grad():
        dummy = torch.eye(3).view(1, 1, 3, 3).expand(len(betas), 24, 3, 3)
        smpl = SMPL(f"smpl/SMPL_{gender}.pkl",
                     joint_mapper=mapper
                    )
        smpl_outputs = smpl(betas=betas,
                            body_pose=dummy[:, 1:],
                            global_orient=dummy[:, :1],
                            pose2rot=False)
        rest_poses = smpl_outputs.joints.cpu().numpy()
        # center the rest pose
        rest_poses -= rest_poses[:, 0:1]

    # use the average as the smpl shape
    rest_pose = rest_poses.mean(0)

    if scale_rest_pose:
        ref_pose = ref_pose * ext_scale
        bone_len = calculate_bone_length(rest_pose).mean()
        ref_bone_len = calculate_bone_length(ref_pose).mean()
        pose_scale = ref_bone_len / bone_len
    else:
        pose_scale = 1.0

    rest_pose = rest_pose * pose_scale

    # Note: A lot of rotation can be achieved by swaping axis (as we did above).
    # However, for clarity, we use matrix multiplication below instead.

    # multiply by pose_scale to make it close to the original surreal scale
    pelvis = joints[:, align_joint_idx].numpy() # (N, 3)
    pelvis = pelvis[..., None] * pose_scale

    zeros = torch.FloatTensor(rot_mats.shape[0], 24, 3, 1).zero_()
    rot_mats = torch.cat([rot_mats, zeros], dim=-1)
    bones = rotation_matrix_to_angle_axis(rot_mats.view(-1, 3, 4)).view(rot_mats.shape[0], 24, 3).numpy()
    l2ws = np.array([get_smpl_l2ws(bone, rest_pose=rest_pose) for bone in bones])

    pelvis = pelvis.reshape(pelvis.shape[0], 1, 3)
    l2ws[:, :, :3, -1] += pelvis
    kp3d = l2ws[:, :, :3, -1].copy()
    skts = np.array([np.linalg.inv(l2w) for l2w in l2ws])

    return kp3d, bones, skts, rest_pose, pose_scale

def pred_cams_to_orig_cam_params(cameras, bboxes,
                                 img_width=512, img_height=512,
                                 resized_width=224, resized_height=224,
                                 focal=5000., ext_scale=1.0,
                                 new_focal=None):

    orig_cams = convert_crop_cam_to_orig_img_and_focal(cameras, bboxes,
                                                       img_width=img_width,
                                                       img_height=img_height,
                                                       resized_width=resized_width,
                                                       resized_height=resized_height,
                                                       focal=focal,
                                                       new_focal=new_focal)
    focals = orig_cams[:, 0]
    cam_translations = orig_cams[:, 1:] * ext_scale
    c2ws = np.eye(4, dtype=np.float32)
    c2ws = c2ws[None].repeat(len(orig_cams), 0)
    c2ws[:, :3, -1] = -cam_translations

    # turn this to NeRF format
    c2ws = swap_mat(c2ws)

    return focals, c2ws


def process_spin_data(betas, cameras, joints, rot_mats, bboxes,
                      ref_pose=smpl_rest_pose,
                      align_joint_idx=8, focal=5000, res=512,
                      resized_res=224, ext_scale=0.001,
                      dataset_ext_scale = 0.25 / 0.00035,
                      align_surreal_cam=False,
                      scale_rest_pose=True,
                      new_focal=None,
                      skel_type=SMPLSkeleton):

    if isinstance(res, int):
        res_H = res_W = res
    else:
        res_H, res_W = res

    ext_scale = ext_scale * dataset_ext_scale

    kp3d, bones, skts, rest_pose, pose_scale = get_keypoints_from_betas(
                                                  betas, joints, rot_mats,
                                                  ext_scale, align_joint_idx,
                                                  ref_pose, scale_rest_pose)

    cyls = get_kp_bounding_cylinder(kp3d,
                                    ext_scale=ext_scale / dataset_ext_scale,
                                    skel_type=skel_type,
                                    extend_mm=250,
                                    head='-y')


    # arrays of [focal, cx, cy, cz]
    focals, c2ws = pred_cams_to_orig_cam_params(cameras, bboxes,
                                                img_width=res_W,
                                                img_height=res_H,
                                                resized_width=resized_res,
                                                resized_height=resized_res,
                                                focal=focal,
                                                ext_scale=pose_scale,
                                                new_focal=new_focal)

    kp3d = kp3d.astype(np.float32)
    bones = bones.astype(np.float32)
    cyls = cyls.astype(np.float32)
    skts = skts.astype(np.float32)
    rest_pose = rest_pose.astype(np.float32)
    c2ws = c2ws.astype(np.float32)
    focals = focals.astype(np.float32)
    print("Done procesing spin data")

    return {"kp3d": kp3d, "bones": bones, "cyls": cyls, "skts": skts,
            "rest_pose": rest_pose, "ext_scale": ext_scale,
            "c2ws": c2ws, "focals": focals, "pose_scale": pose_scale}

def write_to_h5py(filename, data, img_chunk_size=64,
                compression='gzip'):

    imgs = data['imgs']
    H, W = imgs.shape[1:3]

    # TODO: any smarter way for this?
    redundants = ['index', 'img_path']
    img_to_chunk = ['imgs', 'bkgds', 'masks']
    img_to_keep_whole = ['sampling_masks']

    for r in redundants:
        if r in data:
            data.pop(r)

    chunk = (1, int(img_chunk_size**2),)
    whole = (1, H * W,)

    h5_file = h5py.File(filename, 'w')

    # store meta
    ds = h5_file.create_dataset('img_shape', (4,), np.int32)
    ds[:] = np.array([*imgs.shape])

    for k in data.keys():
        if not isinstance(data[k], Iterable):
            print(f'{k}: non-iterable')
            ds = h5_file.create_dataset(k, (), type(data[k]))
            ds[()] = data[k]
            continue

        d_shape = data[k].shape
        C = d_shape[-1]
        N = d_shape[0]
        if k in img_to_chunk or k in img_to_keep_whole:
            data_chunk = chunk + (C,) if k in img_to_chunk else whole + (C,)
            flatten_shape = (N, H * W, C)
            print(f'{k}: img to chunk in size {data_chunk}, flatten as {flatten_shape}')
            # flatten the image for faster indexing
            ds = h5_file.create_dataset(k, flatten_shape, data[k].dtype,
                                        chunks=data_chunk, compression=compression)
            for idx in range(N):
                ds[idx] = data[k][idx].reshape(*flatten_shape[1:])
            #ds[:] = data[k].reshape(*flatten_shape)
        elif k == 'img_paths':
            img_paths = data[k].astype('S')
            ds = h5_file.create_dataset(k, (len(img_paths),), img_paths.dtype)
            ds[:] = img_paths
        else:
            if np.issubdtype(data[k].dtype, np.floating):
                dtype = np.float32
            elif np.issubdtype(data[k].dtype, np.integer):
                dtype = np.int64
            else:
                raise NotImplementedError('Unknown datatype for key {k}: {data[k].dtype}')

            ds = h5_file.create_dataset(k, data[k].shape, dtype,
                                        compression=compression)
            ds[:] = data[k][:]
            print(f'{k}: data to store as {dtype}')
        pass

    h5_file.close()

