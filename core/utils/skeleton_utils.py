import io
import cv2
import torch
import numpy as np
import plotly.graph_objects as go
from PIL import Image
from collections import deque
from copy import copy, deepcopy
from collections import namedtuple
from scipy.spatial.transform import Rotation
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pytorch3d.transforms.rotation_conversions as p3dr

#################################
#         Skeleton Helpers      #
#################################

Skeleton = namedtuple("Skeleton", ["joint_names", "joint_trees", "root_id", "nonroot_id", "cutoffs", "end_effectors"])

def rotate_x(phi):
    cos = np.cos(phi)
    sin = np.sin(phi)
    return np.array([[1,   0,    0, 0],
                     [0, cos, -sin, 0],
                     [0, sin,  cos, 0],
                     [0,   0,    0, 1]], dtype=np.float32)

def rotate_z(psi):
    cos = np.cos(psi)
    sin = np.sin(psi)
    return np.array([[cos, -sin, 0, 0],
                     [sin,  cos, 0, 0],
                     [0,      0, 1, 0],
                     [0,      0, 0, 1]], dtype=np.float32)
def rotate_y(theta):
    cos = np.cos(theta)
    sin = np.sin(theta)
    return np.array([[cos,   0, -sin, 0],
                     [0,     1,    0, 0],
                     [sin,   0,  cos, 0],
                     [0,   0,      0, 1]], dtype=np.float32)

def translate(tx, ty, tz):
    return np.array([[1, 0, 0, tx],
                     [0, 1, 0, ty],
                     [0, 0, 1, tz],
                     [0, 0, 0, 1]])

def arccos_safe(a):
    clipped = np.clip(a, -1.+1e-8, 1.-1e-8)
    return np.arccos(clipped)

def hmm(A, B):
    R_A, T_A = A[..., :3, :3], A[..., :3, -1:]
    R_B, T_B = B[..., :3, :3], B[..., :3, -1:]
    R = R_A @ R_B
    T = R_A @ T_B + T_A
    return torch.cat([R, T], dim=-1)

CanonicalSkeleton = Skeleton(
    joint_names=[
        # 0-4
        'head_top', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist',
        # 5-9
        'left_shoulder', 'left_elbow', 'left_wrist', 'right_hip', 'right_knee',
        # 10-14
        'right_ankle', 'left_hip', 'left_knee', 'left_ankle', 'pelvis',
        # 15-16
        'spine', 'head',
    ],
    joint_trees=np.array([
        1, 15, 1, 2, 3,
        1, 5, 6, 14, 8,
        9, 14, 11, 12, 14,
        14, 1]),
    root_id=14,
    nonroot_id=[i for i in range(16) if i != 14],
    cutoffs={},
    end_effectors=None,
)

SMPLSkeleton = Skeleton(
    joint_names=[
        # 0-3
        'pelvis', 'left_hip', 'right_hip', 'spine1',
        # 4-7
        'left_knee', 'right_knee', 'spine2', 'left_ankle',
        # 8-11
        'right_ankle', 'spine3', 'left_foot', 'right_foot',
        # 12-15
        'neck', 'left_collar', 'right_collar', 'head',
        # 16-19
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        # 20-23,
        'left_wrist', 'right_wrist', 'left_hand', 'right_hand'
    ],
    joint_trees=np.array(
                [0, 0, 0, 0,
                 1, 2, 3, 4,
                 5, 6, 7, 8,
                 9, 9, 9, 12,
                 13, 14, 16, 17,
                 18, 19, 20, 21]),
    root_id=0,
    nonroot_id=[i for i in range(24) if i != 0],
    cutoffs={'hip': 200, 'spine': 300, 'knee': 70, 'ankle': 70, 'foot': 40, 'collar': 100,
            'neck': 100, 'head': 120, 'shoulder': 70, 'elbow': 70, 'wrist': 60, 'hand': 60},
    end_effectors=[10, 11, 15, 22, 23],
)

# for backward compatibility
CMUSkeleton = SMPLSkeleton

SMPLSkeletonExtended = Skeleton(
    joint_names=[
        # 0-3
        'pelvis', 'left_hip', 'right_hip', 'spine1',
        # 4-7
        'left_knee', 'right_knee', 'spine2', 'left_ankle',
        # 8-11
        'right_ankle', 'spine3', 'left_foot', 'right_foot',
        # 12-15
        'neck', 'left_collar', 'right_collar', 'head',
        # 16-19
        'left_shoulder', 'right_shoulder', 'left_upper_arm', 'right_upper_arm',
        # 20-23,
        'left_elbow', 'right_elbow', 'left_lower_arm', 'right_lower_arm',
        # 24-27
        'left_wrist', 'right_wrist', 'left_hand', 'right_hand'
    ],
    joint_trees=np.array(
                [0, 0, 0, 0,
                 1, 2, 3, 4,
                 5, 6, 7, 8,
                 9, 9, 9, 12,
                 13, 14, 16, 17,
                 18, 19, 20, 21,
                 22, 23, 24, 25]),
    root_id=0,
    nonroot_id=[i for i in range(27) if i != 0],
    cutoffs={},
    end_effectors=None,
)



Mpi3dhpSkeleton = Skeleton(
    joint_names=[
    # 0-3
    'spine3', 'spine4', 'spine2', 'spine',
    # 4-7
    'pelvis', 'neck', 'head', 'head_top',
    # 8-11
    'left_clavicle', 'left_shoulder', 'left_elbow', 'left_wrist',
    # 12-15
    'left_hand',  'right_clavicle', 'right_shoulder', 'right_elbow',
    # 16-19
    'right_wrist', 'right_hand', 'left_hip', 'left_knee',
    # 20-23
    'left_ankle', 'left_foot', 'left_toe', 'right_hip',
    # 24-27
    'right_knee', 'right_ankle', 'right_foot', 'right_toe'
    ],
    joint_trees=np.array([
        2, 0, 3, 4,
        4, 1, 5, 6,
        5, 8, 9, 10,
        11, 5, 13, 14,
        15, 16, 4, 18,
        19, 20, 21, 4,
        23, 24, 25, 26
    ]),
    root_id=4,
    nonroot_id=[i for i in range(27) if i != 4],
    cutoffs={},
    end_effectors=None,
)

def get_skeleton_type(kps):

    if kps.shape[-2] == 17:
        skel_type = CanonicalSkeleton
    elif kps.shape[-2] == 28:
        skel_type = Mpi3dhpSkeleton
    else:
        skel_type = SMPLSkeleton
    return skel_type

def get_index_mapping_to_canon(skel_def):

    canon_joint_names = CanonicalSkeleton.joint_names
    try:
        idx_map = [skel_def.joint_names.index(j) for j in canon_joint_names]
    except:
        raise ValueError("Skeleton joint does not match the canonical one.")
    return idx_map

def canonicalize(skel, skel_type="3dhp"):

    if skel_type == "3dhp":
        skel_def = Mpi3dhpSkeleton
        idx_map = [7, 5, 14, 15, 16, 9, 10, 11, 23, 24, 25, 18, 19, 20, 4, 3, 6]

    if idx_map is None:
        idx_map = get_index_mapping_to_canon(skel_def)

    if len(skel.shape) == 3:
        canon_skel = deepcopy(skel[:, idx_map, :])
    elif len(skel.shape) == 2:
        canon_skel = deepcopy(skel[idx_map, :])
    else:
        raise ValueError(f"Skeleton should have either 3 or 2 dimensions, but got {len(skel)} instead.")

    return canon_skel

def coord_to_homogeneous(c):
    assert c.shape[-1] == 3

    if len(c.shape) == 2:
        h = np.ones((c.shape[0], 1)).astype(c.dtype)
        return np.concatenate([c, h], axis=1)
    elif len(c.shape) == 1:
        h = np.array([0, 0, 0, 1]).astype(c.dtype)
        h[:3] = c
        return h
    else:
        raise NotImplementedError(f"Input must be a 2-d or 1-d array, got {len(c.shape)}")

#################################
#       smpl processing         #
#################################
"""
smpl_rest_pose = np.array([[ 0.00000000e+00, -9.86228770e-08, -2.30003661e-09],
                           [ 1.63832515e-01, -2.89178602e-02,  2.17391014e-01],
                           [-1.57855421e-01, -2.09642015e-02,  2.14761734e-01],
                           [-7.04505108e-03, -4.11837511e-02, -2.50450850e-01],
                           [ 2.42021069e-01, -3.14962119e-02,  1.08830070e+00],
                           [-2.47206554e-01, -3.06970738e-02,  1.10715497e+00],
                           [ 3.95125849e-03, -4.03754264e-02, -5.94849110e-01],
                           [ 2.12680623e-01, -1.29327580e-01,  1.99382353e+00],
                           [-2.10857525e-01, -1.23002514e-01,  2.01218796e+00],
                           [ 9.39484313e-03,  2.06931755e-02, -7.19204426e-01],
                           [ 2.63385147e-01,  1.46775618e-01,  2.12222481e+00],
                           [-2.51970559e-01,  1.60450473e-01,  2.12153077e+00],
                           [ 3.83779174e-03, -9.78838727e-02, -1.22592449e+00],
                           [ 1.91201791e-01, -6.21964522e-02, -1.00385976e+00],
                           [-1.77145526e-01, -7.55542740e-02, -9.96228695e-01],
                           [ 1.68482102e-02,  2.44048554e-02, -1.38698268e+00],
                           [ 4.01985168e-01, -7.47655183e-02, -1.07928419e+00],
                           [-3.98825467e-01, -9.96334553e-02, -1.07523870e+00],
                           [ 1.00236952e+00, -1.35129794e-01, -1.05217218e+00],
                           [-9.86728609e-01, -1.40235111e-01, -1.04515052e+00],
                           [ 1.56646240e+00, -1.37338534e-01, -1.06961894e+00],
                           [-1.56946480e+00, -1.53905824e-01, -1.05935931e+00],
                           [ 1.75282109e+00, -1.68231070e-01, -1.04682994e+00],
                           [-1.75758195e+00, -1.77773550e-01, -1.04255080e+00]], dtype=np.float32)
"""
smpl_rest_pose = np.array([[ 0.00000000e+00,  2.30003661e-09, -9.86228770e-08],
                           [ 1.63832515e-01, -2.17391014e-01, -2.89178602e-02],
                           [-1.57855421e-01, -2.14761734e-01, -2.09642015e-02],
                           [-7.04505108e-03,  2.50450850e-01, -4.11837511e-02],
                           [ 2.42021069e-01, -1.08830070e+00, -3.14962119e-02],
                           [-2.47206554e-01, -1.10715497e+00, -3.06970738e-02],
                           [ 3.95125849e-03,  5.94849110e-01, -4.03754264e-02],
                           [ 2.12680623e-01, -1.99382353e+00, -1.29327580e-01],
                           [-2.10857525e-01, -2.01218796e+00, -1.23002514e-01],
                           [ 9.39484313e-03,  7.19204426e-01,  2.06931755e-02],
                           [ 2.63385147e-01, -2.12222481e+00,  1.46775618e-01],
                           [-2.51970559e-01, -2.12153077e+00,  1.60450473e-01],
                           [ 3.83779174e-03,  1.22592449e+00, -9.78838727e-02],
                           [ 1.91201791e-01,  1.00385976e+00, -6.21964522e-02],
                           [-1.77145526e-01,  9.96228695e-01, -7.55542740e-02],
                           [ 1.68482102e-02,  1.38698268e+00,  2.44048554e-02],
                           [ 4.01985168e-01,  1.07928419e+00, -7.47655183e-02],
                           [-3.98825467e-01,  1.07523870e+00, -9.96334553e-02],
                           [ 1.00236952e+00,  1.05217218e+00, -1.35129794e-01],
                           [-9.86728609e-01,  1.04515052e+00, -1.40235111e-01],
                           [ 1.56646240e+00,  1.06961894e+00, -1.37338534e-01],
                           [-1.56946480e+00,  1.05935931e+00, -1.53905824e-01],
                           [ 1.75282109e+00,  1.04682994e+00, -1.68231070e-01],
                           [-1.75758195e+00,  1.04255080e+00, -1.77773550e-01]], dtype=np.float32)

def get_noisy_joints(kp3d, ext_scale, noise_mm):
    noise = np.random.normal(scale=noise_mm * ext_scale, size=kp3d.shape)
    print(f"noise of scale {noise_mm * ext_scale}")
    return kp3d + noise

def get_noisy_bones(bones, noise_degree):
    noise_scale = np.pi / 180. * noise_degree
    mask = (np.random.random(bones.shape) > 0.5).astype(np.float32)
    noise = np.random.normal(0, noise_scale, bones.shape) * mask
    noisy_bones = bones + noise
    print(np.abs(noise).max())
    return noisy_bones

def perturb_poses(bone_poses, kp_3d, ext_scale, noise_degree=0.1,
                  noise_mm=None, dataset_ext_scale=0.25 / 0.00035,
                  noise_pelvis=None,
                  skel_type=SMPLSkeleton):

    noisy_bones = bone_poses if noise_degree is None else get_noisy_bones(bone_poses, noise_degree)
    rest_poses = smpl_rest_pose.copy()[None]
    # scale rest poses accordingly
    rest_poses = rest_poses.repeat(kp_3d.shape[0], 0) * ext_scale
    if noise_mm is not None:
        rest_poses = get_noisy_joints(rest_poses,
                                      ext_scale / dataset_ext_scale,
                                      noise_mm)

    pelvis_loc = kp_3d[:, skel_type.root_id].copy()
    if noise_pelvis is not None:
        noise = np.random.normal(scale=noise_pelvis * ext_scale / dataset_ext_scale, size=pelvis_loc.shape)
        pelvis_loc += noise
    l2ws = np.array([get_smpl_l2ws(bone, pose, scale=1.) for bone, pose in zip(noisy_bones, rest_poses)])
    l2ws[:, :, :3, -1] += pelvis_loc[:, None]

    noisy_skts = np.array([np.linalg.inv(l2w) for l2w in l2ws])
    noisy_kp = l2ws[:, :, :3, -1]

    return noisy_bones, noisy_skts, noisy_kp

def skt_from_smpl(bone_poses, scale, kp_3d=None,
                  pelvis_loc=None, skel_type=SMPLSkeleton):
    l2ws = np.array([get_smpl_l2ws(bone_pose, scale=scale) for bone_pose in bone_poses])

    if kp_3d is not None:
        l2ws[:, :, :3, -1] = kp_3d
    if pelvis_loc is not None:
        l2ws[:, :, :3, -1] += pelvis_loc[:, None]
    skts = np.array([np.linalg.inv(l2w) for l2w in l2ws])
    return skts, l2ws

def get_smpl_l2ws(pose, rest_pose=None, scale=1., skel_type=SMPLSkeleton, coord="xxx"):
    # TODO: take root as well

    def mat_to_homo(mat):
        last_row = np.array([[0, 0, 0, 1]], dtype=np.float32)
        return np.concatenate([mat, last_row], axis=0)

    joint_trees = skel_type.joint_trees
    if rest_pose is None:
        # original bone parameters is in (x,-z,y), while rest_pose is in (x, y, z)
        rest_pose = smpl_rest_pose


    # apply scale
    rest_kp = rest_pose * scale
    mrots = [Rotation.from_rotvec(p).as_matrix()  for p in pose]
    mrots = np.array(mrots)

    l2ws = []
    # TODO: assume root id = 0
    # local-to-world transformation
    l2ws.append(mat_to_homo(np.concatenate([mrots[0], rest_kp[0, :, None]], axis=-1)))
    mrots = mrots[1:]
    for i in range(rest_kp.shape[0] - 1):
        idx = i + 1
        # rotation relative to parent
        joint_rot = mrots[idx-1]
        joint_j = rest_kp[idx][:, None]

        parent = joint_trees[idx]
        parent_j = rest_kp[parent][:, None]

        # transfer from local to parent coord
        joint_rel_transform = mat_to_homo(
            np.concatenate([joint_rot, joint_j - parent_j], axis=-1)
        )

        # calculate kinematic chain by applying parent transform (to global)
        l2ws.append(l2ws[parent] @ joint_rel_transform)

    l2ws = np.array(l2ws)

    return l2ws

def get_rest_pose_from_l2ws(l2ws, skel_type=SMPLSkeleton):
    """
    extract rest pose from local to world matrices for each joint
    """

    joint_trees = skel_type.joint_trees

    rest_pose = [l2ws[skel_type.root_id, :3, -1]]
    kp = l2ws[:, :3, -1]
    for i in range(len(l2ws[1:])):
        idx = i + 1
        parent = joint_trees[idx]
        # rotate from world to the parent rest pose!
        joint_rel_pos =  l2ws[parent, :3, :3].T @ (kp[idx] - kp[parent])
        joint_pos = rest_pose[parent] + joint_rel_pos
        rest_pose.append(joint_pos)

    return np.array(rest_pose)

def bones_to_rot(bones):
    if bones.shape[-1] == 3:
        return axisang_to_rot(bones)
    elif bones.shape[-1] == 6:
        return rot6d_to_rotmat(bones)
    else:
        raise NotImplementedError

def rot_to_axisang(rot):
    return p3dr.matrix_to_axis_angle(rot)

def rot_to_rot6d(rot):
    return rot[..., :3, :2].flatten(start_dim=-2)

def axisang_to_rot(axisang):
    return p3dr.axis_angle_to_matrix(axisang)

def axisang_to_quat(axisang):
    return p3dr.axis_angle_to_quaternion(axisang)

def rot6d_to_axisang(rot6d):
    return rot_to_axisang(rot6d_to_rotmat(rot6d))

def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (*,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x_shape = x.shape[:-1]
    x = x.reshape(-1,3,2)

    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1).reshape(*x_shape, 3, 3)

#################################
#       Coordinate System       #
#################################

def nerf_c2w_to_extrinsic(c2w):
    return np.linalg.inv(swap_mat(c2w))

## LEGACY BELOW ##
def nerf_bones_to_smpl(bones):
    # undo local transformation
    bones = torch.cat([bones[..., 0:1], -bones[..., 2:3], bones[..., 1:2]], dim=-1)
    rots = axisang_to_rot(bones.view(-1, 3)).view(*bones.shape[:2], 3, 3)
    # undo global transformation
    root_rot = torch.tensor([[1., 0., 0.],
                             [0., 0.,-1.],
                             [0., 1., 0.]]).to(bones.device)
    root_rot = root_rot.expand(len(rots), 3, 3)
    rots[:, 0] = root_rot @ rots[:, 0]
    return rots

def smpl_pts_to_surreal(pts):
    return np.concatenate([
        pts[..., 0:1],
        pts[..., 2:3],
       -pts[..., 1:2],
    ], axis=-1)

def smpl_rot_to_surreal(rot, np=False):
    R = [[1, 0, 0],
         [0, 0, 1],
         [0,-1, 0]]
    R = torch.FloatTensor(R).to(rot.device) if not np else np.array(R, dtype=np.float32)
    if len(rot.shape) == 3:
        return R[None] @ rot
    return R @ rot
## LEGACY ABOVE ##

def skeleton3d_to_2d(kps, c2ws, H, W, focals, centers=None):

    exts = np.array([nerf_c2w_to_extrinsic(c2w) for c2w in c2ws])

    kp2ds = []
    for i, (kp, ext) in enumerate(zip(kps, exts)):
        f = focals[i] if not isinstance(focals, float) else focals
        h = H if isinstance(H, int) else H[i]
        w = W if isinstance(W, int) else W[i]
        center = centers[i] if centers is not None else None
        kp2d = world_to_cam(kp, ext, h, w, f, center)
        kp2ds.append(kp2d)

    return np.array(kp2ds)

#################################
#       Keypoint processing     #
#################################
def create_local_coord(vec):
    '''
    Find the coordinate system that have z-axis aligned with vec
    '''
    x_axis = np.array([1., 0., 0.], dtype=np.float32)
    y_axis = np.array([0., 1., 0.], dtype=np.float32)
    z_axis = np.array([0., 0., 1.], dtype=np.float32)

    if np.isclose(np.linalg.norm(vec), 0.):
        return np.stack([x_axis, y_axis, z_axis])

    # first, find rotation around y-axis
    vec_xz = vec[[0, 2]] / np.linalg.norm(vec[[0, 2]])
    # note: we care only about the projection on z-axis
    # note: reference point is on z, so if x is negative, then we are in 3 or 4th quadrant
    theta = arccos_safe(vec_xz[-1]) * np.sign(vec_xz[0])
    rot_y = rotate_y(theta)
    rotated_y = rot_y[:3, :3] @ vec

    # then, find rotation around x-axis
    vec_yz = rotated_y[1:3] / np.linalg.norm(rotated_y[1:3])
    # similarly, make sign correct
    psi = arccos_safe(vec_yz[-1]) * np.sign(vec_yz[0])
    rot_x = rotate_x(psi)
    rotated_x = rot_x[:3, :3] @ rotated_y
    # rotate the coordinate system to so that z_axis == vec
    rot = np.linalg.inv(rot_x @ rot_y)

    coord = np.stack([x_axis, y_axis, z_axis])

    return coord @ rot[:3, :3].T

def get_per_joint_coords(rest_pose, skel_type=SMPLSkeleton):
    """
    compute the per-joint coordinate systems
    """

    coords = []
    print("parent-centered")
    for i, j in enumerate(skel_type.joint_trees):
        #vec = (rest_pose[i] - rest_pose[j])
        vec = (rest_pose[j] - rest_pose[i])
        vec = vec / (np.linalg.norm(vec) + 1e-5)
        coord = create_local_coord(vec)

        coords.append(coord)
    return np.array(coords)


def get_kp_bounding_cylinder(kp, skel_type=None, ext_scale=0.00035,
                             extend_mm=250, top_expand_ratio=1.,
                             bot_expand_ratio=0.25, head=None):
    '''
    head: -y for most dataset (SPIN estimated), z for SURREAL
    '''

    # g_axes: axes that define the ground plane
    # h_axis: axis that is perpendicular to the ground
    # flip: to flip the height (if the sky is on the negative part)
    assert head is not None, 'need to specify the direction of ground plane (i.e., the direction when the person stand up straight)!'
    print(f'Head direction: {head}')
    if head.endswith('z'):
        g_axes = [0, 1]
        h_axis = 2
    elif head.endswith('y'):
        g_axes = [0, 2]
        h_axis = 1
    else:
        raise NotImplementedError(f'Head orientation {head} is not implemented!')
    flip = 1 if not head.startswith('-') else -1

    if skel_type is None:
        skel_type = get_skeleton_type(kp)

    n_dim = len(kp.shape)
    # find root location
    root_loc = kp[..., skel_type.root_id, :]

    # calculate distance to center line
    if n_dim == 2:
        dist = np.linalg.norm(kp[:, g_axes] - root_loc[g_axes], axis=-1)
    elif n_dim == 3: # batch
        dist = np.linalg.norm(kp[..., g_axes] - root_loc[:, None, g_axes], axis=-1)
        max_height = (flip * kp[..., h_axis]).max()
        min_height = (flip * kp[..., h_axis]).min()

    # find the maximum distance to center line (in mm*ext_scale)
    max_dist = dist.max(-1)
    max_height = (flip * kp[..., h_axis]).max(-1)
    min_height = (flip * kp[..., h_axis]).min(-1)

    # set the radius of cylinder to be a bit larger
    # so that every part of the human is covered
    extension = extend_mm * ext_scale
    radius = max_dist + extension
    top = flip * (max_height + extension * top_expand_ratio) # extend head a bit more
    bot = flip * (min_height - extension * bot_expand_ratio) # don't need that much for foot
    cylinder_params = np.stack([root_loc[..., g_axes[0]], root_loc[..., g_axes[1]],
                               radius, top, bot], axis=-1)
    return cylinder_params

def calculate_angle(a, b=None):
    if b is None:
        b = torch.Tensor([0., 0., 1.]).view(1, 1, -1)
    dot_product = (a * b).sum(-1)
    norm_a = torch.norm(a, p=2, dim=-1)
    norm_b = torch.norm(b, p=2, dim=-1)
    cos = dot_product / (norm_a * norm_b)
    cos = torch.clamp(cos, -1. + 1e-6, 1. - 1e-6)
    angle = torch.acos(cos)
    assert not torch.isnan(angle).any()

    return angle - 0.5 * np.pi

def cylinder_to_box_2d(cylinder_params, hwf, w2c=None, scale=1.0,
                       center=None, make_int=True):

    H, W, focal = hwf

    root_loc, radius = cylinder_params[..., :2], cylinder_params[..., 2:3]
    top, bot = cylinder_params[..., 3:4], cylinder_params[..., 4:5]

    rads = np.linspace(0., 2 * np.pi, 50)

    if len(root_loc.shape) == 1:
        root_loc = root_loc[None]
        radius = radius[None]
        top = top[None]
        bot = bot[None]
    N = root_loc.shape[0]

    x = root_loc[..., 0:1] + np.cos(rads)[None] * radius
    z = root_loc[..., 1:2] + np.sin(rads)[None] * radius

    y_top = top * np.ones_like(x)
    y_bot = bot * np.ones_like(x)
    w = np.ones_like(x) # to make homogenous coord

    top_cap = np.stack([x, y_top, z, w], axis=-1)
    bot_cap = np.stack([x, y_bot, z, w], axis=-1)

    cap_pts = np.concatenate([top_cap, bot_cap], axis=-2)
    cap_pts = cap_pts.reshape(-1, 4)

    intrinsic = focal_to_intrinsic_np(focal)

    if w2c is not None:
        cap_pts = cap_pts @ w2c.T
    cap_pts = cap_pts @ intrinsic.T
    cap_pts = cap_pts.reshape(N, -1, 3)
    pts_2d = cap_pts[..., :2] / cap_pts[..., 2:3]

    max_x = pts_2d[..., 0].max(-1)
    min_x = pts_2d[..., 0].min(-1)
    max_y = pts_2d[..., 1].max(-1)
    min_y = pts_2d[..., 1].min(-1)

    if make_int:
        max_x = np.ceil(max_x).astype(np.int32)
        min_x = np.floor(min_x).astype(np.int32)
        max_y = np.ceil(max_y).astype(np.int32)
        min_y = np.floor(min_y).astype(np.int32)

    tl = np.stack([min_x, min_y], axis=-1)
    br = np.stack([max_x, max_y], axis=-1)

    if center is None:
        offset_x = int(W * .5)
        offset_y = int(H * .5)
    else:
        offset_x, offset_y = int(center[0]), int(center[1])


    tl[:, 0] += offset_x
    tl[:, 1] += offset_y

    br[:, 0] += offset_x
    br[:, 1] += offset_y

    # scale the box
    if scale != 1.0:
        box_width = (max_x - min_x) * 0.5 * scale
        box_height = (max_y - min_y) * 0.5 * scale
        center_x = (br[:, 0] + tl[:, 0]).copy() * 0.5
        center_y = (br[:, 1] + tl[:, 1]).copy() * 0.5

        tl[:, 0] = center_x - box_width
        br[:, 0] = center_x + box_width
        tl[:, 1] = center_y - box_height
        br[:, 1] = center_y + box_height

    tl[:, 0] = np.clip(tl[:, 0], 0, W-1)
    br[:, 0] = np.clip(br[:, 0], 0, W-1)
    tl[:, 1] = np.clip(tl[:, 1], 0, H-1)
    br[:, 1] = np.clip(br[:, 1], 0, H-1)

    if N == 1:
        tl = tl[0]
        br = br[0]
        pts_2d = pts_2d[0]

    return tl, br, pts_2d

def get_skeleton_transformation(kps, skel_type=None, align_z=True):


    if skel_type is None:
        skel_type = get_skeleton_type(kps)

    skt_builder = HierarchicalTransformation(skel_type.joint_trees, skel_type.joint_names,
                                             align_z=align_z)
    skts = []
    transformed_kps = []

    for kp in kps:
        skt_builder.build_transformation(kp)
        skts.append(np.array(skt_builder.get_all_w2ls()))
        transformed_kps.append(skt_builder.transform_w2ls(kp))
    skts = np.stack(skts)
    transformed_kps = np.array(transformed_kps)[..., :3]
    return skts, transformed_kps

def extend_cmu_skeleton(kps):
    original_joints = SMPLSkeleton.joint_names
    extended_joints = SMPLSkeletonExtended.joint_names

    mapping = [extended_joints.index(name) for name in original_joints]
    name_to_idx = {name: i for i, name in enumerate(extended_joints)}

    extended = np.zeros((kps.shape[0], 28, 3)).astype(np.float32)
    extended[:, mapping] = kps

    # get the indices we want to set
    left_upper = name_to_idx["left_upper_arm"]
    left_lower = name_to_idx["left_lower_arm"]
    right_upper = name_to_idx["right_upper_arm"]
    right_lower = name_to_idx["right_lower_arm"]

    # get the indices for calculation
    left_shoulder = name_to_idx["left_shoulder"]
    left_elbow = name_to_idx["left_elbow"]

    right_shoulder = name_to_idx["right_shoulder"]
    right_elbow = name_to_idx["right_elbow"]

    left_wrist = name_to_idx["left_wrist"]
    right_wrist = name_to_idx["right_wrist"]

    extended[:, left_upper] = (extended[:, left_shoulder] + extended[:, left_elbow]) * 0.5
    extended[:, right_upper] = (extended[:, right_shoulder] + extended[:, right_elbow]) * 0.5

    extended[:, left_lower] = (extended[:, left_wrist] + extended[:, left_elbow]) * 0.5
    extended[:, right_lower] = (extended[:, right_wrist] + extended[:, right_elbow]) * 0.5

    return extended

def get_geodesic_dists(kp, skel_type=SMPLSkeleton):
    """
    Run Floyd-Warshall algo to get geodesic distances
    """

    N_J = kp.shape[-2]
    dists = np.ones((N_J, N_J)) * 100000.
    joint_trees = skel_type.joint_trees

    for i, parent in enumerate(joint_trees):
        dist = ((kp[i] - kp[parent])**2.0).sum()**0.5
        dists[i, parent] = dist
        dists[parent, i] = dist
        dists[i, i] = 0.

    for k in range(N_J):
        for i in range(N_J):
            for j in range(N_J):
                new_dist = dists[i][k] + dists[k][j]
                if dists[i][j] > new_dist:
                    dists[i][j] = new_dist

    return dists


def create_kp_masks(masks, kp2ds=None, kp3ds=None, c2ws=None, hwf=None,
                   extend_iter=3, skel_type=SMPLSkeleton):
    '''
    Create extra foreground region for the (estimated) keypoints.
    ----
    masks: (N, H, W, 1) original masks
    kp2ds: (N, NJ, 2) batch of 2d keypoints
    kp3ds: (N, NJ, 3) batch of 3d keypoints
    c2ws: (N, 4, 4) batch of camera-to-world matrices
    hwf: height, width and focal(s)
    extend_iter: number of times to expand the keypoint masks. Each iter create around 2 pixels.
    '''

    if kp2ds is None:
        assert kp3ds is not None
        H, W, focal = hwf
        kp2ds = skeleton3d_to_2d(kp3d, c2ws, H, W, focal)

    H, W, C = masks[0].shape

    kp_masks = []

    for i, kp in enumerate(kp2ds):
        kp_mask = create_kp_mask(masks[i], kp, H, W, extend_iter, skel_type)
        kp_masks.append(kp_mask)

    return kp_masks

def create_kp_mask(mask, kp, H, W, extend_iter=3, skel_type=SMPLSkeleton):

    # draw skeleton on blank image
    blank = np.zeros((H, W, 3), dtype=np.uint8)
    drawn = draw_skeleton2d(blank, kp, skel_type)
    drawn = drawn.mean(axis=-1)
    drawn[drawn > 0] = 1

    # extend the skeleton a bit more so it covers some space
    # and merge it with mask
    d_kernel = np.ones((5, 5))
    dilated = cv2.dilate(drawn, kernel=d_kernel, iterations=extend_iter)[..., None]
    kp_mask = np.logical_or(mask, dilated).astype(np.uint8)

    return kp_mask

#################################
#       Plotting Helpers        #
#################################

def plot_skeleton3d(skel, fig=None, skel_id=None, skel_type=None,
                    cam_loc=None, layout_kwargs=None, line_width=2,
                    marker_size=3, blank=False):
    """
    Plotting function for canonicalized skeleton
    """

    # x, y, z of lines in the center/left/right body part
    clx, cly, clz = [], [], []
    llx, lly, llz = [], [], []
    rlx, rly, rlz = [], [], []

    x, y, z = list(skel[..., 0]), list(skel[..., 1]), list(skel[..., 2])

    if skel_type is None:
        skel_type = get_skeleton_type(skel)

    joint_names = skel_type.joint_names
    joint_tree = skel_type.joint_trees
    root_id = skel_type.root_id

    for i, (j, name) in enumerate(zip(joint_tree, joint_names)):
        if "left" in name:
            lx, ly, lz = llx, lly, llz
        elif "right" in name:
            lx, ly, lz = rlx, rly, rlz
        else:
            lx, ly, lz = clx, cly, clz

        lx += [x[i], x[j], None]
        ly += [y[i], y[j], None]
        lz += [z[i], z[j], None]

    joint_names = [f"{i}_{name}" for i, name in enumerate(joint_names)]
    if skel_id is not None:
        joint_names = [f"{skel_id}_{name}" for name in joint_names]

    points = go.Scatter3d(x=x, y=y, z=z, mode="markers", marker=dict(size=marker_size),
                          line=dict(color="blue"),
                          text=joint_names)
    center_lines = go.Scatter3d(x=clx, y=cly, z=clz, mode="lines",
                                line=dict(color="black", width=line_width),
                                hoverinfo="none")
    left_lines = go.Scatter3d(x=llx, y=lly, z=llz, mode="lines",
                              line=dict(color="blue", width=line_width),
                              hoverinfo="none")
    right_lines = go.Scatter3d(x=rlx, y=rly, z=rlz, mode="lines",
                               line=dict(color="red", width=line_width),
                               hoverinfo="none")
    data = [points, center_lines, left_lines, right_lines]

    if fig is None:
        fig = go.Figure(data=data)
    else:
        for d in data:
            fig.add_trace(d)

    camera = dict(up=dict(x=0, y=1, z=0))
    if cam_loc is not None:
        camera['eye'] = dict(x=cam_loc[0], y=cam_loc[1], z=cam_loc[2])
    fig.update_layout(scene_camera=camera)
    if blank:
        if layout_kwargs is not None:
            print('WARNING! layout_kwargs is overwritten by blank=True')
        fig.update_layout(scene=dict(
            xaxis=dict(backgroundcolor='rgb(255, 255, 255)'),
            yaxis=dict(backgroundcolor='rgb(255, 255, 255)'),
            zaxis=dict(backgroundcolor='rgb(255, 255, 255)'),
        ))
    elif layout_kwargs is not None:
        fig.update_layout(**layout_kwargs)
    return fig

def plot_skeleton2d(skel, skel_type=None, img=None):

    if skel_type is None:
        skel_type = get_skeleton_type(skel)

    joint_names = skel_type.joint_names
    joint_tree = skel_type.joint_trees

    if img is not None:
        plt.imshow(img)

    for i, j in enumerate(skel):
        name = joint_names[i]
        parent = skel[joint_tree[i]]
        offset = parent - j

        if "left" in name:
            color = "red"
        elif "right" in name:
            color = "blue"
        else:
            color = "green"
        plt.arrow(j[0], j[1], offset[0], offset[1], color=color)

def plot_points3d(pts, fig=None, label=False, marker_size=3, color="orange"):

    if label:
        names = [f'{i}' for i in range(len(pts))]
    else:
        names = None
    pts = go.Scatter3d(x=pts[..., 0], y=pts[..., 1], z=pts[..., 2],
                       mode='markers', marker=dict(size=marker_size), text=names,
                       line=dict(color=color))

    data = [pts]
    if fig is None:
        fig = go.Figure(data=data)
    else:
        for d in data:
            fig.add_trace(d)
    return fig

def plot_cameras(extrinsics=None, viewmats=None, fig=None):
    if extrinsics is not None:
        viewmats = np.array([np.linalg.inv(ext) for ext in extrinsics])
    cam_pos = viewmats[:, :3, 3]
    rights = viewmats[:, :3, 0] * 0.5
    ups = viewmats[:, :3, 1] * 0.5
    fwds = viewmats[:, :3, 2] * 0.5

    rlx, rly, rlz = [], [], []
    ulx, uly, ulz = [], [], []
    flx, fly, flz = [], [], []

    for cp, r, u, f in zip(cam_pos, rights, ups, fwds):
        rlx += [cp[0], cp[0] + r[0], None]
        rly += [cp[1], cp[1] + r[1], None]
        rlz += [cp[2], cp[2] + r[2], None]

        ulx += [cp[0], cp[0] + u[0], None]
        uly += [cp[1], cp[1] + u[1], None]
        ulz += [cp[2], cp[2] + u[2], None]

        flx += [cp[0], cp[0] + f[0], None]
        fly += [cp[1], cp[1] + f[1], None]
        flz += [cp[2], cp[2] + f[2], None]

    points = go.Scatter3d(x=cam_pos[:, 0], y=cam_pos[:, 1], z=cam_pos[:, 2],
                          mode="markers",
                          line=dict(color="orange"),
                          text=[f"cam {i}" for i in range(len(cam_pos))])

    up_lines = go.Scatter3d(x=ulx, y=uly, z=ulz, mode="lines",
                                line=dict(color="magenta", width=2),
                                hoverinfo="none")
    right_lines = go.Scatter3d(x=rlx, y=rly, z=rlz, mode="lines",
                                line=dict(color="green", width=2),
                                hoverinfo="none")
    forward_lines = go.Scatter3d(x=flx, y=fly, z=flz, mode="lines",
                                line=dict(color="orange", width=2),
                                hoverinfo="none")

    data = [points, up_lines, right_lines, forward_lines]
    if fig is None:
        fig = go.Figure(data=data)
    else:
        for d in data:
            fig.add_trace(d)

    camera = dict(up=dict(x=0, y=1, z=0))
    fig.update_layout(scene_camera=camera)

    return fig

def get_plotly_plane(tl, tr, br, bl, N=3, color='blue',
                     showscale=False, opacity=0.3, **kwargs):
    x = np.linspace(tl[0], tr[0], N)
    y = np.linspace(tl[1], bl[1], N)
    z = np.array([[tl[2], tr[2]], [bl[2], br[2]]])
    colorscale = [[0, color], [1, color]]

    return go.Surface(x=x, y=y, z=z, colorscale=colorscale,
                      showscale=showscale, opacity=opacity, **kwargs)

def plot_frustum(os, ds, near=1.0, far=6.0, color="red", fig=None, near_only=False):
    lx, ly, lz = [], [], []
    near_planes = []
    far_planes = []

    for o, d in zip(os, ds):
        nf = np.array(d) * far
        nd = np.array(d) * near
        for i in range(4):
            if near_only:
                line = nd
            else:
                line = nf
            lx += [o[0], o[0] + line[i][0], None]
            ly += [o[1], o[1] + line[i][1], None]
            lz += [o[2], o[2] + line[i][2], None]


        # plot near plane
        nx = [o[0] + nd[0][0], o[0] + nd[1][0],
              o[0] + nd[2][0], o[0] + nd[3][0],
              o[0] + nd[0][0], None]
        ny = [o[1] + nd[0][1], o[1] + nd[1][1],
              o[1] + nd[2][1], o[1] + nd[3][1],
              o[1] + nd[0][1], None]
        nz = [o[2] + nd[0][2], o[2] + nd[1][2],
              o[2] + nd[2][2], o[2] + nd[3][2],
              o[2] + nd[0][2], None]
        near_plane = go.Scatter3d(x=nx, y=ny, z=nz, mode="lines",
                                  surfaceaxis=0, opacity=0.3,
                                  line=dict(color="blue", width=2)
                                 )
        near_planes.append(near_plane)

        if not near_only:
            fx = [o[0] + nf[0][0], o[0] + nf[1][0],
                  o[0] + nf[2][0], o[0] + nf[3][0],
                  o[0] + nf[0][0]]
            fy = [o[1] + nf[0][1], o[1] + nf[1][1],
                  o[1] + nf[2][1], o[1] + nf[3][1],
                  o[1] + nf[0][1]]
            fz = [o[2] + nf[0][2], o[2] + nf[1][2],
                  o[2] + nf[2][2], o[2] + nf[3][2],
                  o[2] + nf[0][2]]
            far_plane = go.Scatter3d(x=fx, y=fy, z=fz, mode="lines",
                                      line=dict(color="green", width=2)
                                     )
            far_planes.append(far_plane)

    ray_lines = go.Scatter3d(x=lx, y=ly, z=lz, mode="lines",
                             line=dict(color=color, width=2),
                             hoverinfo="none")

    data = [ray_lines, *near_planes, *far_planes]
    if fig is None:
        fig = go.Figure(data=data)
    else:
        for d in data:
            fig.add_trace(d)

    camera = dict(up=dict(x=0, y=1, z=0))
    fig.update_layout(scene_camera=camera)

    return fig

def plot_3d_bounding_box(vertices, fig=None):
    v0, v1, v2, v3, v4, v5, v6, v7 = vertices

    # v0->v1->v2->v3->v0
    lower_cap = np.concatenate([vertices[:4], vertices[:1]], axis=0)
    lower_cap = go.Scatter3d(x=lower_cap[:, 0], y=lower_cap[:, 1], z=lower_cap[:, 2],
                             mode="lines",
                             line=dict(color="red"))
    # v4->v5->v6->v7->v4
    upper_cap = np.concatenate([vertices[4:], vertices[4:5]], axis=0)
    upper_cap = go.Scatter3d(x=upper_cap[:, 0], y=upper_cap[:, 1], z=upper_cap[:, 2],
                             mode="lines",
                             line=dict(color="blue"))
    lx, ly, lz = [], [], []
    # v0->v4, v1->v5, v2->v6, v3->v7
    for i in range(4):
        lx += [vertices[i, 0], vertices[i+4, 0], None]
        ly += [vertices[i, 1], vertices[i+4, 1], None]
        lz += [vertices[i, 2], vertices[i+4, 2], None]

    lines = go.Scatter3d(x=lx, y=ly, z=lz, mode="lines",
                         line=dict(color="black"))

    data = [lower_cap, upper_cap, lines]

    if fig is None:
        fig = go.Figure(data=data)
    else:
        for d in data:
            fig.add_trace(d)
    camera = dict(up=dict(x=0, y=1, z=0))
    fig.update_layout(scene_camera=camera)

    return fig


def byte2array(byte):
    buf = io.BytesIO(byte)
    img = Image.open(buf)
    return np.asarray(img)

def create_plane(x=None, y=None, z=None, length_x=10., length_y=5,
                 length_z=10, n_sample=50):

    x_r = length_x / 2.
    y_r = length_y / 2.
    z_r = length_z / 2.
    if z is not None:
        plane ='xy'
        x, y = np.meshgrid(np.linspace(-x_r, x_r, n_sample), np.linspace(-y_r, y_r, n_sample))
        plane = np.stack([x, y, np.ones_like(x) * z], axis=-1)
    elif y is not None:
        plane = 'xz'
        x, z = np.meshgrid(np.linspace(-x_r, x_r, n_sample), np.linspace(-z_r, z_r, n_sample))
        plane = np.stack([x, np.ones_like(x) * y, z], axis=-1)
    else:
        plane = 'yz'
        y, z = np.meshgrid(np.linspace(-y_r, y_r, n_sample), np.linspace(-z_r, z_r, n_sample))
        plane = np.stack([np.ones_like(y) * x, y, z], axis=-1)

    return plane.astype(np.float32)

def dist_to_joints(kp3d, pts):
    return np.linalg.norm(pts[:, None, :] - kp3d[None], axis=-1)

def plot_bounding_cylinder(kp, cylinder_params=None, fig=None, **kwargs):
    if cylinder_params is None:
        cylinder_params = get_kp_bounding_cylinder(kp, **kwargs)

    root_loc, radius = cylinder_params[:2], cylinder_params[2]
    top, bot = cylinder_params[3], cylinder_params[4]

    # create circle
    rads = np.linspace(0., 2 * np.pi, 50)
    x = root_loc[0] + np.cos(rads) * radius
    z = root_loc[1] + np.sin(rads) * radius

    y_top = top * np.ones_like(x)
    y_bot = bot * np.ones_like(x)

    #cap_x = x.tolist() + [None] + x.tolist()
    #cap_y = y_top.tolist() + [None] + y_bot.tolist()
    #cap_z = z.tolist() + [None] + z.tolist()

    top_x = x.tolist() + [None]
    top_y = y_top.tolist() + [None]
    top_z = z.tolist() + [None]

    bot_x = x.tolist() + [None]
    bot_y = y_bot.tolist() + [None]
    bot_z = z.tolist() + [None]

    top_cap = go.Scatter3d(x=top_x, y=top_y, z=top_z, mode="lines",
                        line=dict(color="red", width=2))
    bot_cap = go.Scatter3d(x=bot_x, y=bot_y, z=bot_z, mode="lines",
                        line=dict(color="blue", width=2))
    data = [top_cap, bot_cap]
    if fig is None:
        fig = go.Figure(data=data)
    else:
        for d in data:
            fig.add_trace(d)

    return fig

def plot_surface(coords, fig=None, amp=None):
    surface = go.Surface(x=coords[..., 0], y=coords[..., 1], z=coords[...,2],
                         surfacecolor=amp, colorscale="Spectral", opacity=0.5,
                         cmax=1., cmin=-1.)

    if fig is not None:
        for trace in fig['data']:
            trace['showlegend'] = False
        fig.add_trace(surface)
    else:
        fig = go.Figure(data=[surface])
    return fig

def get_surface_fig(kp, embedder, joint_names, n_sample=640, joint_idx=0,
                    freq=0, freq_skip=2, fig=None, sin=True,
                    x_offset=0, y_offset=0, z_offset=0, n_split=2,
                    x_only=False, y_only=False, z_only=False, v_range=2):
    #xy_plane = create_plane(z=kp[joint_idx,2]+z_offset, n_sample=n_sample)
    #yz_plane = create_plane(x=kp[joint_idx,0]+x_offset, n_sample=n_sample)
    #xz_plane = create_plane(y=kp[joint_idx,1]+y_offset, n_sample=n_sample)
    xy_plane = create_plane(z=z_offset, n_sample=n_sample)
    yz_plane = create_plane(x=x_offset, n_sample=n_sample)
    xz_plane = create_plane(y=y_offset, n_sample=n_sample)
    if x_only:
        planes = [yz_plane]
    elif y_only:
        planes = [xz_plane]
    elif z_only:
        planes = [xy_plane]
    else:
        planes = [xy_plane, yz_plane, xz_plane]

    embeds = []
    for plane in planes:

        dist = dist_to_joints(kp, plane.reshape(-1, 3))
        embed = [embedder.embed(torch.FloatTensor(dist[:, i:i+1])).numpy().reshape(n_sample, n_sample, -1)
                for i in range(dist.shape[1])]
        embeds.append(embed)
    embeds = np.array(embeds)

    embed_sins = embeds[..., 1::2][..., ::freq_skip]
    embed_coss = embeds[..., 2::2][..., ::freq_skip]
    p_fn = "Sin" if sin else "Cos"
    #title = f"{p_fn} - Joint: {joint_names[joint_idx]} - Freq: 2^{freq_skip * freq}"
    fig = plot_skeleton3d(kp, fig=fig)
    embeds = embed_sins if sin else embed_coss
    for i, (p, e) in enumerate(zip(planes, embeds)):
        fig = plot_surface(p, fig=fig, amp=e[joint_idx, ..., freq])

    fig.update_layout(#title_text=title,
                      scene=dict(
                          xaxis = dict(range=[-v_range, v_range]),
                          yaxis = dict(range=[-v_range, v_range]),
                          zaxis = dict(range=[-v_range, v_range]),

                      ),
                      scene_aspectmode='cube',
                      margin=dict(r=10, l=10, b=10, t=10),)
    return fig

def generate_sweep_vid(embedder, path, cam_pose, kp, axis="x",
                       joint_idx=SMPLSkeleton.joint_names.index("right_hand"),
                       offset=5, n_sample=80, freq=0, ext_scale=0.001, H=512, W=512, focal=100):
    import imageio
    from ray_utils import get_corner_rays
    offsets = np.linspace(-offset, offset, 28)
    imgs = []

    x_only = True if axis == "x" else False
    y_only = True if axis == "y" else False
    z_only = True if axis == "z" else False

    #x_only, y_only, z_only = False, False, False

    for offset in offsets:
        fig = plot_cameras(viewmats=swap_mat(cam_pose))
        #rays_o, rays_d = get_corner_rays(H, W, focal, cam_pose)
        #fig = plot_frustum(np.array(rays_o), np.array(rays_d),
        #                  near=0.5, far=4.2,
        #                  fig=fig, near_only=False)
        #fig = plot_bounding_cylinder(kp, ext_scale=ext_scale, fig=fig,
        #                             extend_mm=500)

        fig = get_surface_fig(kp, embedder, joint_names=SMPLSkeleton.joint_names, n_sample=n_sample, fig=fig,
                              joint_idx=joint_idx, freq=freq, z_offset=offset, y_offset=offset, x_offset=offset,
                              n_split=4, x_only=x_only, y_only=y_only, z_only=z_only, v_range=np.max(offsets)+0.3)
        img = byte2array(fig.to_image(format="png"))
        imgs.append(img)
    imgs = np.array(imgs)
    imageio.mimwrite(path, (imgs).astype(np.uint8), fps=14, quality=8)
    return imgs


def plot_joint_axis(kp, l2ws=None, fig=None, scale=0.1):

    rights = np.array([1., 0., 0., 1. / scale]) * scale
    ups    = np.array([0., 1., 0., 1. / scale]) * scale
    fwds   = np.array([0., 0., 1., 1. / scale]) * scale

    rights = np.array([l2w @ rights for l2w in l2ws])
    ups = np.array([l2w @ ups for l2w in l2ws])
    fwds = np.array([l2w @ fwds for l2w in l2ws])

    rlx, rly, rlz = [], [], []
    ulx, uly, ulz = [], [], []
    flx, fly, flz = [], [], []

    for k, r, u, f in zip(kp, rights, ups, fwds):
        #import pdb; pdb.set_trace()
        rlx += [k[0], r[0], None]
        rly += [k[1], r[1], None]
        rlz += [k[2], r[2], None]

        ulx += [k[0], u[0], None]
        uly += [k[1], u[1], None]
        ulz += [k[2], u[2], None]

        flx += [k[0], f[0], None]
        fly += [k[1], f[1], None]
        flz += [k[2], f[2], None]

    up_lines = go.Scatter3d(x=ulx, y=uly, z=ulz, mode="lines",
                            line=dict(color="magenta", width=2),
                            hoverinfo="none")
    right_lines = go.Scatter3d(x=rlx, y=rly, z=rlz, mode="lines",
                               line=dict(color="green", width=2),
                               hoverinfo="none")
    forward_lines = go.Scatter3d(x=flx, y=fly, z=flz, mode="lines",
                                 line=dict(color="orange", width=2),
                                 hoverinfo="none")
    data = [up_lines, right_lines, forward_lines]
    if fig is not None:
        for d in data:
            fig.add_trace(d)
    else:
        fig = go.Figure(data=data)
    return fig

def swap_mat(mat):
    # [right, -up, -forward]
    # equivalent to right multiply by:
    # [1, 0, 0, 0]
    # [0,-1, 0, 0]
    # [0, 0,-1, 0]
    # [0, 0, 0, 1]
    return np.concatenate([
        mat[..., 0:1], -mat[..., 1:2], -mat[..., 2:3], mat[..., 3:]
        ], axis=-1)

def focal_to_intrinsic_np(focal):
    if isinstance(focal, float) or (len(focal.reshape(-1)) < 2):
        focal_x = focal_y = focal
    else:
        focal_x, focal_y = focal
    return np.array([[focal_x,      0, 0, 0],
                     [     0, focal_y, 0, 0],
                     [     0,       0, 1, 0]],
                    dtype=np.float32)



def world_to_cam(pts, extrinsic, H, W, focal, center=None):

    if center is None:
        offset_x = W * .5
        offset_y = H * .5
    else:
        offset_x, offset_y = center

    if pts.shape[-1] < 4:
        pts = coord_to_homogeneous(pts)

    intrinsic = focal_to_intrinsic_np(focal)

    cam_pts = pts @ extrinsic.T @ intrinsic.T
    cam_pts = cam_pts[..., :2] / cam_pts[..., 2:3]
    cam_pts[cam_pts == np.inf] = 0.
    cam_pts[..., 0] += offset_x
    cam_pts[..., 1] += offset_y
    return cam_pts

def calculate_bone_length(kp, skel_type=SMPLSkeleton):
    assert skel_type.root_id == 0
    joint_trees = skel_type.joint_trees
    bone_lens = []
    for i in range(1, kp.shape[0]):
        parent = joint_trees[i]
        bone_lens.append((((kp[i] - kp[parent])**2).sum()**0.5))
    return np.array(bone_lens)

#################################
#          Draw Helpers         #
#################################
def draw_skeletons_3d(imgs, skels, c2ws, H, W, focals, centers=None,
                      width=3, flip=False, skel_type=SMPLSkeleton):
    skels_2d = skeleton3d_to_2d(skels, c2ws, H, W, focals, centers)
    return draw_skeletons(imgs, skels_2d, skel_type, width, flip)

def draw_skeletons(imgs, skels, skel_type=SMPLSkeleton, width=3, flip=False):
    skel_imgs = []
    for img, skel in zip(imgs, skels):
        skel_img = draw_skeleton2d(img, skel, skel_type, width, flip)
        skel_imgs.append(skel_img)
    return np.array(skel_imgs)

def draw_skeleton2d(img, skel, skel_type=None, width=3, flip=False):
    if skel_type is None:
        skel_type = get_skeleton_type(skel)

    joint_names = skel_type.joint_names
    joint_tree = skel_type.joint_trees

    img = img.copy()

    for i, j in enumerate(skel):
        name = joint_names[i]
        parent = skel[joint_tree[i]]
        offset = parent - j

        if "left" in name:
            color = (100, 149, 237) if not flip else (0, 128, 128)
        elif "right" in name:
            color = (255, 0, 0) if not flip else (128, 128, 0)
        else:
            color = (46, 204, 13) if not flip else (128, 0, 128)
        cv2.line(img, (int(round(j[0])), int(round(j[1]))),
                      (int(round(j[0]+offset[0])), int(round(j[1]+offset[1]))),
                 color, thickness=width)
    return img

def draw_opt_poses(c2ws, hwf, anchor_kps, opt_kps, gt_kps=None,
                   base_imgs=None, centers=None, res=0.1, skel_type=SMPLSkeleton,
                   thickness=4, marker_size=30):
    '''
    Plot both anchor and optimized keypoints
    '''
    root_id = skel_type.root_id
    h, w, focals = hwf

    anchors_2d = skeleton3d_to_2d(anchor_kps, c2ws, h, w, focals, centers)
    opts_2d = skeleton3d_to_2d(opt_kps, c2ws, h, w, focals, centers)

    shift_h = anchors_2d[:, root_id, 0] - opts_2d[:, root_id, 0]
    shift_w = anchors_2d[:, root_id, 1] - opts_2d[:, root_id, 1]
    anchors_2d[..., 0] -= shift_h[:, None]
    anchors_2d[..., 1] -= shift_w[:, None]

    # Step 2: draw the image
    imgs = []
    for i, (anchor_2d, opt_2d) in enumerate(zip(anchors_2d, opts_2d)):
        img = base_imgs[i]
        img = draw_markers(opt_2d, img=img, H=h, W=w, res=res, thickness=thickness, marker_size=marker_size, color=(255, 149, 0, 32))
        img = draw_markers(anchor_2d, img=img, H=h, W=w, res=res, thickness=thickness,
                           marker_size=marker_size, color=(0, 200, 0, 32),
                           marker_type=cv2.MARKER_TILTED_CROSS)
        imgs.append(img)
    return np.array(imgs)

def draw_markers(pts, img=None, H=1000, W=1000, res=0.1, marker_type=cv2.MARKER_CROSS,
                 thickness=3, color=(255, 0, 0), marker_size=5):
    '''
    Draw 2d points on the input image
    '''
    if img is None:
        img = np.ones((int(H * res), int(W * res), 3), dtype=np.uint8) * 255
    elif res != 1.0:
        img = cv2.resize(img, (int(W * res), int(H * res)))
    for p in pts:
        img = cv2.drawMarker(img, (int(p[0] * res), int(p[1] * res)),
                             color=color, markerType=marker_type,
                             thickness=thickness, markerSize=marker_size)
    return img

