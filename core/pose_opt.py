import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from smplx import SMPL
import time

from .utils.skeleton_utils import SMPLSkeleton, axisang_to_rot, rot6d_to_rotmat,\
                                  rot_to_axisang, rot6d_to_axisang, get_smpl_l2ws,\
                                  get_kp_bounding_cylinder, bones_to_rot, hmm
from .process_spin import SMPL_JOINT_MAPPER


def create_popt(args, data_attrs, ckpt=None, device=None):


    # get attributes from dict
    skel_type = data_attrs['skel_type']
    rest_pose = data_attrs['rest_pose'].reshape(-1, len(skel_type.joint_names), 3)
    rest_pose = torch.tensor(rest_pose, requires_grad=False)
    beta = torch.tensor(data_attrs['betas'])
    init_kps = torch.tensor(data_attrs['kp3d'])
    init_bones = torch.tensor(data_attrs['bones'])
    kp_map = data_attrs.get('kp_map', None)
    kp_uidxs = data_attrs.get('kp_uidxs', None)
    rest_pose_idxs = data_attrs.get('rest_pose_idxs', None)

    poseopt_kwargs = {}
    poseopt_class = PoseOptLayer

    popt_layer = poseopt_class(init_kps.clone().to(device),
                               init_bones.clone().to(device),
                               rest_pose.to(device),
                               beta=beta,
                               skel_type=skel_type,
                               kp_map=kp_map,
                               kp_uidxs=kp_uidxs,
                               rest_pose_idxs=rest_pose_idxs,
                               use_cache=args.opt_pose_cache,
                               use_rot6d=args.opt_rot6d,
                               **poseopt_kwargs)

    grad_opt_vars = list(popt_layer.parameters())
    pose_optimizer = torch.optim.Adam(params=grad_opt_vars,
                                      lr=args.opt_pose_lrate,
                                      betas=(0.9, 0.999))

    # initalize anchor (for regularization loss)
    anchor_kps, anchor_bones, anchor_beta = init_kps, init_bones, beta

    if (ckpt is not None or args.init_poseopt is not None) and not (args.no_poseopt_reload):
        # load from checkpoint
        pose_ckpt = torch.load(args.init_poseopt) if args.init_poseopt is not None else ckpt
        popt_layer.load_state_dict(pose_ckpt["poseopt_layer_state_dict"])
        pose_optimizer.load_state_dict(pose_ckpt["pose_optimizer_state_dict"])

        if "poseopt_anchors" in pose_ckpt:
            anchor_kps = pose_ckpt["poseopt_anchors"]["kps"]
            anchor_bones = pose_ckpt["poseopt_anchors"]["bones"]
            anchor_beta = pose_ckpt["poseopt_anchors"]["beta"]

        if args.use_ckpt_anchor:
            # use the poses data from ckpt as optimization constraint
            with torch.no_grad():
                anchor_kps, anchor_bones, _, anchor_rots = popt_layer(torch.arange(anchor_bones.shape[0]))
            anchor_kps, anchor_bones = anchor_kps.cpu().clone(), anchor_bones.cpu().clone()
            anchor_beta = popt_layer.get_beta()
        print('load smpl state dict')

    # recompute anchor_rots to ensure it's consistent with bones
    anchor_rots = axisang_to_rot(anchor_bones.view(-1, 3)).view(*anchor_kps.shape[:2], 3, 3)
    popt_anchors = {"kps": anchor_kps, "bones": anchor_bones, "rots": anchor_rots, "beta": anchor_beta}

    if popt_layer.use_cache:
        print("update cache for faster forward")
        popt_layer.update_cache()

    pose_optimizer.zero_grad() # to remove gradients from ckpt
    popt_kwargs = {'popt_anchors': popt_anchors,
                   'popt_layer': popt_layer,
                   'skel_type': skel_type}

    return pose_optimizer, popt_kwargs

def sample_patch_idx(H, W, valid_y, valid_x, patch_size, N_patch):
    W_min = H_min = patch_size // 2

    valid_y = torch.clamp(valid_y, H_min, H)
    valid_x = torch.clamp(valid_x, W_min, W)

    indices = torch.randperm(len(valid_y))[:N_patch]

    y = valid_y[indices]
    x = valid_x[indices]

    sampled_h = []
    sampled_w = []
    for cy, cx in zip(y, x):
        hs = cy - patch_size // 2
        he = cy + patch_size // 2
        ws = cx - patch_size // 2
        we = cx + patch_size // 2

        h = torch.arange(hs, he)
        w = torch.arange(ws, we)
        h, w = torch.meshgrid(h, w)
        sampled_h.append(h)
        sampled_w.append(w)
    sampled_h = torch.stack(sampled_h)
    sampled_w = torch.stack(sampled_w)

    return sampled_h.view(-1), sampled_w.view(-1)

def mat_to_hom(mat):
    """
    To homogeneous coordinates
    """
    last_row = torch.tensor([[0., 0., 0., 1.]]).to(mat.device)

    if mat.dim() == 3:
        last_row = last_row.expand(mat.size(0), 1, 4)
    return torch.cat([mat, last_row], dim=-2)

def get_kp_reg_loss(preds, regs, gts=None, opt_pose_coefs=1.0,
                    kp_idx=None, args=None, skel_type=SMPLSkeleton):

    # TODO: deal with redundancy
    kps, bones, rots = preds["kps"], preds["bones"], preds["rots"]
    reg_kps, reg_bones, reg_rots= regs["kps"], regs["bones"], regs["rots"]
    gt_kps = None if gts is None else gts["kps"]

    root_id = skel_type.root_id
    kp_sqr_diff = (reg_kps - kps).pow(2).sum(-1)
    opt_type = args.opt_pose_type

    mse_no_reduce = lambda a, b: (a - b).pow(2)
    l1_no_reduce = lambda a, b: (a - b).abs()

    loss_fn = l1_no_reduce if "L1" in opt_type else mse_no_reduce
    coef_on_global = not("E" in opt_type)
    root_id = skel_type.root_id

    if args.opt_rot6d:
        reg_bones = reg_rots[..., :3, :2].flatten(start_dim=-2)

    if opt_type.startswith("B"):
        bone_loss = loss_fn(reg_bones, bones)
        pelv_loss = loss_fn(reg_kps[:, root_id], kps[:, root_id]).sum(-1)
    elif opt_type.startswith("RD"):
        bone_loss = loss_fn(rots, reg_rots)
        pelv_loss = loss_fn(reg_kps[:, root_id], kps[:, root_id]).sum(-1)
    else:
        raise NotImplementedError("Regularization target un-specified")

    # loss = 0 if bone_loss < tol
    # loss = bone_loss - tol otherwise
    tol = args.opt_pose_tol
    loss_mask = (bone_loss > tol).float()
    bone_loss = torch.lerp(torch.zeros_like(bone_loss), bone_loss-tol, loss_mask).sum(-1)

    if coef_on_global:
        kp_loss = (bone_loss.mean() + pelv_loss.mean()) * opt_pose_coefs
    else:
        bone_loss = bone_loss[:, root_id+1:].mean()
        kp_loss = bone_loss * opt_pose_coefs #+ bone_g_loss + pelv_loss.mean()

    temp_loss = torch.tensor(0.)
    if args.use_temp_loss:
        # TODO: only do it on rotation now.
        NB = bones.shape[0]
        temp_valid = regs["temp_valid"]
        temp_bones = regs["temp_rots"][..., :3, :2].flatten(start_dim=-2) if args.opt_rot6d else regs["temp_bones"]
        temp_kps = regs["temp_kps"]
        assert (NB * 2) == temp_bones.shape[0]
        prev_bones, next_bones = torch.chunk(temp_bones, chunks=2, dim=0)
        prev_kps , next_kps = torch.chunk(temp_kps, chunks=2, dim=0)

        if not args.use_temp_vel:
            temp_loss = loss_fn(prev_bones, bones).sum(-1)
            temp_loss = (temp_loss * temp_valid[..., None]).mean() * args.temp_coef
        else:
            temp_valid_next = regs["temp_valid_next"]
            # both previous and next bones need to be valid
            temp_valid = (temp_valid_next + temp_valid) // 2
            # change of velocity between steps!
            ang_vel = ((bones - prev_bones) - (next_bones - bones)).pow(2.).sum(-1)
            joint_vel = ((kps - prev_kps) - (next_kps - kps)).pow(2.).sum(-1)
            #temp_loss = temp_loss * temp_valid[..., None]
            temp_loss = (ang_vel + joint_vel) * temp_valid[..., None]
            temp_loss = temp_loss.mean() * args.temp_coef

        kp_loss = kp_loss + temp_loss

    mpjpc = kp_sqr_diff.detach().pow(0.5).mean() / args.ext_scale

    if gts is not None:
        kp_gt_dist = torch.norm(kps.detach() - gt_kps, p=2, dim=-1).mean() / args.ext_scale
    else:
        kp_gt_dist = None

    return kp_loss, temp_loss, mpjpc, kp_gt_dist

def load_bones_from_state_dict(state_dict, device='cpu'):
    bones = state_dict['poseopt_layer_state_dict']['bones']
    if bones.shape[-1] == 6:
        N, N_J, _ = bones.shape
        bones = bones.view(-1, 6)
        rots = rot6d_to_rotmat(bones)
        bones = rot_to_axisang(rots).view(N, N_J, -1)
    return bones.to(device)

def load_poseopt_from_state_dict(state_dict):
    '''
    Assume no kp_map for now...
    '''
    print("Loading pose opt state dict")
    pelvis = state_dict['poseopt_layer_state_dict']['pelvis']
    # bones do not necessarily have the right shape
    bones = state_dict['poseopt_layer_state_dict']['bones']

    # prob if it is multiview
    kp_map = kp_uidxs = None
    if 'kp_map' in state_dict['poseopt_layer_state_dict']:
        kp_map = state_dict['poseopt_layer_state_dict']['kp_map'].cpu().numpy()
        kp_uidxs = state_dict['poseopt_layer_state_dict']['kp_uidxs'].cpu().numpy()

    N, N_J, N_D = pelvis.shape[0], *bones.shape[1:]
    # multiview setting has root bone removed and stored separately,
    # so need to add 1 back to the bone dimension
    if kp_map is not None:
        N_J += 1
    dummy_kp = torch.zeros(N, N_J, 3)
    dummy_bone = torch.zeros(N, N_J, 3)

    poseopt = PoseOptLayer(dummy_kp, dummy_bone, dummy_kp[0:1],
                           use_rot6d= N_D==6, kp_map=kp_map, kp_uidxs=kp_uidxs)
    poseopt.load_state_dict(state_dict['poseopt_layer_state_dict'])
    return poseopt

class PoseOptLayer(nn.Module):

    def __init__(self, kps, bones, rest_pose, skel_type=SMPLSkeleton,
                 kp_map=None, kp_uidxs=None, use_cache=False, use_rot6d=False,
                 beta=None, rest_pose_idxs=None):
        """
        kps: (N, N_joints, 3)
        bones: (N, N_joints, 3)
        rest_pose: (1, N_joints, 3), assume to be the same for all kps
        """
        super().__init__()

        self.skel_type = skel_type
        self.use_cache = use_cache
        self.unroll_kinematic_chain = False
        self.use_rot6d = use_rot6d
        self.rest_pose_idxs = rest_pose_idxs

        if kp_map is not None:
            self.register_buffer('kp_map', torch.tensor(kp_map).long())
            self.register_buffer('kp_uidxs', torch.tensor(kp_uidxs).long())
        else:
            self.kp_map = self.kp_uidxs = None

        if skel_type == SMPLSkeleton:
            self.root_id = skel_type.root_id
            self.unroll_kinematic_chain = True
        else:
            raise NotImplementedError("only support SMPLSkeleton now")

        self.init_kp_params(kps, bones, rest_pose, kp_uidxs, beta)
        self.N_kps = self.pelvis.shape[0]

        if self.use_cache:
            self.update_cache()

    def init_kp_params(self, kps, bones, rest_pose, kp_uidxs, beta):

        self.beta = torch.tensor(beta, requires_grad=False) if beta is not None else None
        self.register_buffer('rest_pose', torch.tensor(rest_pose, requires_grad=False))

        # pelvis is optimized for each view
        self.register_parameter('pelvis', torch.nn.Parameter(kps[:, self.root_id], requires_grad=True))

        if self.use_rot6d:
            # https://arxiv.org/abs/1812.07035
            NJ = bones.shape[1]
            rots = axisang_to_rot(bones.view(-1, 3)).view(-1, NJ, 3, 3)
            bones = rots[..., :3, :2].reshape(-1, NJ, 6) # 6D rep.

        if self.kp_map is None:
            self.register_parameter('bones', torch.nn.Parameter(bones, requires_grad=True))
            return
        print("SET UP WITH MULTI_VIEWS!")
        self.register_parameter('root_bones', torch.nn.Parameter(bones[:, self.root_id], requires_grad=True))
        self.register_parameter('bones', torch.nn.Parameter(bones[kp_uidxs, self.root_id+1:], requires_grad=True))

    def update_cache(self):

        N = len(self.pelvis)
        idxs = np.arange(N)

        kps, bones, skts, l2ws, rots = self.calculate_kinematic(idxs)

        self.cache_kps = kps
        self.cache_bones = bones
        self.cache_skts = skts
        self.cache_l2ws = l2ws
        self.cache_rots = rots
        #self.rest_pose = self.get_rest_pose()

    def forward(self, idxs, rest_pose_idxs=None):
        if not self.use_cache:
            return self.calculate_kinematic(idxs, rest_pose_idxs)
        return self.cache_kps[idxs], self.cache_bones[idxs], \
                self.cache_skts[idxs], self.cache_l2ws[idxs], \
                 self.cache_rots[idxs]

    def idx_to_params(self, idx):


        torch_idx = torch.tensor(idx).long().view(-1, 1)
        bones = self.bones
        pelvis = self.pelvis[idx]

        if self.kp_map is None:
            return pelvis, bones[idx]

        map_idx = self.kp_map[idx]
        root_bones = self.root_bones[idx, None, :]
        bones = self.bones[map_idx]

        return pelvis, torch.cat([root_bones, bones], dim=1)

    def get_pelvis(self, idx=None):
        if idx is None:
            idx = np.arange(self.N_kps)
        pelvis, _ = self.idx_to_params(idx)
        return pelvis

    def get_beta(self):
        return self.beta

    def get_bones(self, idx=None):
        if idx is None:
            idx = np.arange(self.N_kps)
        _, bones = self.idx_to_params(idx)
        if self.use_rot6d:
            bones = rot6d_to_axisang(bones)

        return bones

    def to_bones3d(self, bones):
        if bones.shape[-1] == 3:
            return bones
        NJ = bones.shape[1]
        rots = rot6d_to_rotmat(bones.view(-1, 3))
        zeros = torch.zeros(*rots.shape[:-1], 1).to(rots.device)
        rots = torch.cat([rots, zeros], dim=-1)
        #bones = rotation_matrix_to_angle_axis(rots.view(-1, 3, 4))
        bones = rot_to_axisang(rots.view(-1, 3, 4))
        bones = bones.view(-1, NJ, 3)
        return bones

    def get_rest_pose(self, kp_idxs=None, rest_pose_idxs=None):
        if len(self.rest_pose) == 1:
            return self.rest_pose
        if rest_pose_idxs is not None:
            return self.rest_pose[rest_pose_idxs]
        return self.rest_pose[self.rest_pose_idxs[kp_idxs]]


    def calculate_kinematic(self, idxs, rest_pose_idxs=None):

        if idxs is None:
            idxs = np.arange(len(self.pelvis))
        if isinstance(idxs, int):
            indices = [idxs]

        # the indices could be redundant, only calculate the unique ones
        unique_idxs, inverse_idxs = np.unique(idxs, return_inverse=True)

        joint_trees = self.skel_type.joint_trees
        root_id = self.skel_type.root_id

        rest_pose = self.get_rest_pose(unique_idxs, rest_pose_idxs)
        #bone = self.bones[unique_indices]
        pelvis, bone = self.idx_to_params(unique_idxs)

        N, N_J, _ = bone.shape

        if self.use_rot6d:
            rots = rot6d_to_rotmat(bone.view(-1, 6)).view(N, N_J, 3, 3)
        else:
            rots = axisang_to_rot(bone.view(-1, 3)).view(N, N_J, 3, 3)
        # TODO: check if this will be a problem.
        rest_pose = rest_pose.expand(N, N_J, 3)

        l2ws = []
        root_l2w = mat_to_hom(torch.cat([rots[:, root_id], rest_pose[:, root_id, :, None]], dim=-1))
        l2ws.append(root_l2w)

        # create joint-to-joint transformation
        children_rots = torch.cat([rots[:, :root_id], rots[:, root_id+1:]], dim=1)
        children_locs = torch.cat([rest_pose[:, :root_id], rest_pose[:, root_id+1:]], dim=1)[..., None]
        parent_ids = np.concatenate([joint_trees[:root_id], joint_trees[root_id+1:]], axis=0)
        parent_locs = rest_pose[:, parent_ids, :, None]

        joint_rel_transforms = mat_to_hom(
                torch.cat([children_rots, children_locs - parent_locs], dim=-1).view(-1, 3, 4)
        ).view(-1, N_J-1, 4, 4)

        if self.unroll_kinematic_chain:
            l2ws = unrolled_kinematic_chain(root_l2w, joint_rel_transforms)
            l2ws = torch.cat(l2ws, dim=-3)
        else:
            for i, parent in enumerate(parent_ids):
                l2ws.append(l2ws[parent] @ joint_rel_transforms[:, i])
            l2ws = torch.stack(l2ws, dim=-3)
            import pdb; pdb.set_trace()
            print("hold up this shouldn't happen!")

        # can't do inplace, so do this instead
        zeros = torch.zeros(N, 4, 3).to(l2ws.device)
        # pad to obtain (N, 4)
        #pelvis = torch.cat([self.pelvis[unique_indices], torch.zeros(N, 1)], dim=-1)[..., None]
        pelvis = torch.cat([pelvis, torch.zeros(N, 1)], dim=-1)[..., None]

        # pad to (N, 4, 4)
        pelvis = torch.cat([zeros, pelvis], dim=-1)

        # add pelvis shift
        l2ws = l2ws + pelvis[:, None]

        # inverse l2ws to get skts (w2ls)
        skts = torch.inverse(l2ws)

        # reconstruct the originally requested sequence
        skts = skts[inverse_idxs]
        bone = bone[inverse_idxs]
        l2ws = l2ws[inverse_idxs]
        rots = rots[inverse_idxs]

        kp = l2ws[..., :3, -1]

        return kp, bone, skts, l2ws, rots


# TODO better naming?
def get_kinematic_chain_T(rest_pose, bones, skel_type=SMPLSkeleton):


    root_id = skel_type.root_id
    joint_trees = skel_type.joint_trees

    # assume to be 24 joints
    N_B, N_J, N_D = bones.shape
    rest_pose = rest_pose.reshape(-1, N_J, 3).expand(N_B, N_J, 3)
    rots = bones_to_rot(bones.reshape(-1, N_D)).reshape(-1, N_J, 3, 3)

    l2ws = []
    root_l2w = mat_to_hom(torch.cat([rots[:, root_id], rest_pose[:, root_id, :, None]], dim=-1))
    l2ws.append(root_l2w)

    # create joint-to-joint transformation
    children_rots = torch.cat([rots[:, :root_id], rots[:, root_id+1:]], dim=1)
    children_locs = torch.cat([rest_pose[:, :root_id], rest_pose[:, root_id+1:]], dim=1)[..., None]
    parent_ids = np.concatenate([joint_trees[:root_id], joint_trees[root_id+1:]], axis=0)
    parent_locs = rest_pose[:, parent_ids, :, None]

    joint_rel_transforms = mat_to_hom(
            torch.cat([children_rots, children_locs - parent_locs], dim=-1).view(-1, 3, 4)
    ).view(-1, N_J-1, 4, 4)

    l2ws = unrolled_kinematic_chain(root_l2w, joint_rel_transforms)
    l2ws = torch.cat(l2ws, dim=-3)

    skts = torch.inverse(l2ws)
    kps = l2ws[..., :3, -1]

    return kps, bones, skts, l2ws, rots

def unrolled_kinematic_chain(root_l2w, joint_rel_transforms):
    """
    unrolled kinematic chain for SMPL skeleton. Should run faster..
    """

    root_l2w = root_l2w[:, None]
    # parallelize left_hip (1), right_hip (2) and spine1 (3)
    # Note: that joint indices are substracted by 1 for joint_rel_transforms
    # because root is excluded
    chain_1 = root_l2w.expand(-1, 3, 4, 4) @ joint_rel_transforms[:, 0:3]

    # parallelize left_knee (4), right_knee (5) and spine2 (6)
    chain_2 = chain_1 @ joint_rel_transforms[:, 3:6]

    # parallelize left_ankle (7), right_angle (8) and spine3 (9)
    chain_3 = chain_2 @ joint_rel_transforms[:, 6:9]

    # parallelize left_foot (10), right_foot (11),
    # neck (12), left_collar (13) and right_collar (14)
    # Note: that last 3 are children of spine3
    chain_4 = chain_3[:, [0, 1, 2, 2, 2]] @ joint_rel_transforms[:, 9:14]

    # parallelize head (15), left_shoulder (16), right_shoulder (17)
    # Note: they connect to neck, left_collar, and right_collar respectively
    # i.e., the last 3 elements in chain_4
    chain_5 = chain_4[:, -3:] @ joint_rel_transforms[:, 14:17]

    # parallelize left_elbow (18) and right_elbow (19)
    # Note: connect to left_collar and right_collar respectively,
    # i.e., the last 2 elelments in chain_5
    chain_6 = chain_5[:, -2:] @ joint_rel_transforms[:, 17:19]

    # parallelize left_wrist (20) and right_wrist (21)
    chain_7 = chain_6 @ joint_rel_transforms[:, 19:21]

    # parallelize left_wrist (22) and right_wrist (23)
    chain_8 = chain_7 @ joint_rel_transforms[:, 21:23]

    return [root_l2w, chain_1, chain_2, chain_3,
            chain_4, chain_5, chain_6, chain_7, chain_8]

def pose_ckpt_to_pose_data(path=None, popt_sd=None, ext_scale=0.001, legacy=False, skel_type=SMPLSkeleton):
    # TODO: ext_scale is hard-coded for now. Fix it later
    if popt_sd is None:
        popt_sd = torch.load(path)['poseopt_layer_state_dict']
    poseopt = load_poseopt_from_state_dict({'poseopt_layer_state_dict': popt_sd})

    with torch.no_grad():
        pelvis = poseopt.get_pelvis().cpu().numpy()
        bones = poseopt.get_bones().cpu().numpy()
        rest_pose = poseopt.get_rest_pose()[0].cpu().numpy()

    if legacy:
        pelvis[..., 1:] *= -1
        rest_pose = np.concatenate([ rest_pose[..., :1],
                                    -rest_pose[..., 2:3],
                                     rest_pose[..., 1:2],], axis=-1)
        bones = np.concatenate([ bones[..., :1],
                                -bones[..., 2:3],
                                 bones[..., 1:2],], axis=-1)
        root_rot = axisang_to_rot(torch.tensor(bones[..., :1, :].reshape(-1, 3)))
        rot_on_root = torch.tensor([[1., 0., 0.],
                                    [0., 0.,-1.],
                                    [0., 1., 0.]]).float()
        root_rot = rot_to_axisang(rot_on_root[None, :3, :3] @ root_rot).reshape(-1, 3).cpu().numpy()
        bones[..., 0, :] = root_rot
        print(f'{path} - legacy')

    # create local-to-world matrices and add global translation
    l2ws = np.array([get_smpl_l2ws(b, rest_pose=rest_pose, scale=1.) for b in bones])
    l2ws[..., :3, -1] += pelvis[:, None]
    kp3d = l2ws[..., :3, -1].copy().astype(np.float32)
    skts = np.linalg.inv(l2ws).astype(np.float32)
    l2ws = l2ws.astype(np.float32)
    # get cylinders
    cyls = get_kp_bounding_cylinder(kp3d, ext_scale=ext_scale, skel_type=skel_type,
                                    extend_mm=250, head='-y').astype(np.float32)
    return kp3d, bones, skts, cyls, rest_pose, pelvis


def update_pose_opt_params(lrate, lrate_decay, decay_rate, optimizer,
                           poseopt_layer, decay_unit, opt_pose_coefs,
                           anchors, reg_rate=5., reg_step=None):
    decay_steps = lrate_decay * decay_unit
    smallest_lrate = 1.0

    step = optimizer.state[optimizer.param_groups[0]['params'][0]]['step']
    new_lrate = lrate * (decay_rate ** (step / decay_steps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lrate

    if reg_step is not None and step % reg_step == 0:
        opt_pose_coefs *= reg_rate
        indices = torch.arange(len(anchors[0]))
        with torch.no_grad():
            opt_kps, opt_bones, opt_skts, _, opt_rots = poseopt_layer(indices)
        new_anchors = [opt_kps.cpu(), opt_bones.cpu(), opt_skts.cpu(), opt_rots.cpu()]
    else:
        new_anchors = anchors

    return new_lrate, new_anchors, opt_pose_coefs

class PoseOptFlipFlop:

    def __init__(self, args, optimizer, pose_optimizer,
                 poseopt_layer, network_fn, network_fine=None):
        self.args = args
        self.optimizer = optimizer
        self.pose_optimizer = pose_optimizer
        self.poseopt_layer = poseopt_layer
        self.network_fn = network_fn
        self.network_fine = network_fine
        self.pose_turn = False if not args.opt_pose_joint else True
        self.opt_pose_reset = args.opt_pose_reset
        self.opt_pose_mtn = args.opt_pose_mtn
        self.opt_pose_anchor_mtn = args.opt_pose_anchor_mtn
        self.opt_pose_mtr = args.opt_pose_mtr
        self.testopt = args.testopt

        self.set_requires_grad()
        self.reset_kp_loss_tracker()
        #self.reset_nerf_loss_tracker()

    def reset_poseopt(self):
        for name, param in self.poseopt_layer.named_parameters():
            param.data.copy_(self.reset_state_dict[name])

    def set_requires_grad(self):

        print(f"set pose_opt -> {self.pose_turn}")
        print(f"set nerf -> {not self.pose_turn}")
        for p in self.poseopt_layer.parameters():
            p.requires_grad = self.pose_turn
        if self.poseopt_layer.use_cache:
            self.poseopt_layer.update_cache()

        # when args.opt_pose_joint == True,
        # we jointly learn NeRF and pose if self.pose_turn == True
        if not self.args.opt_pose_joint or self.testopt:
            for name, p in self.network_fn.named_parameters():
                p.requires_grad = not self.pose_turn
            if self.network_fine is not None:
                for name, p in self.network_fine.named_parameters():
                    p.requires_grad = not self.pose_turn

    def peek_pose_turn(self, i):
        #if i % self.args.opt_pose_interval == 0:
        #    return not self.pose_turn
        opt_pose_stop = self.args.opt_pose_stop is not None and i > self.args.opt_pose_stop
        opt_pose_warmup = i < self.args.opt_pose_warmup
        return self.pose_turn and not opt_pose_stop and not opt_pose_warmup

    def reset_kp_loss_tracker(self):
        # use to record the loss when optimizing poses
        # init with high loss so the training would not be biased to un-optimized pose at the beginning
        self.kp_loss_tracker = torch.ones(self.poseopt_layer.N_kps) * 10.
        self.kp_loss_cnt = torch.zeros(self.poseopt_layer.N_kps)

    def accumulate_loss(self, loss, kp_idx):
        if loss is None:
            return
        loss = loss.view(-1)

        # TODO: note that we will encounter problems when kp_idx is not unique (tho it's unlikely?)
        ones = torch.ones_like(loss)
        kp_idx = torch.LongTensor(kp_idx).to(loss.device)

        # collect kp loss
        acc_loss = torch.zeros_like(self.kp_loss_tracker)
        acc_loss.scatter_add_(0, kp_idx, loss)
        # update moving average by:
        # CMA = kp_loss_tracker * n (current mean)
        # CMA_new = (kp_loss_tracker + current_loss) / (n + 1)
        # set minimum to at least 1 to avoid div-by-zero
        self.kp_loss_cnt.scatter_add_(0, kp_idx, ones)
        n = torch.clamp(self.kp_loss_cnt, min=1)
        # print(f"{n.max()}, {n.min()} {acc_loss.shape} {self.kp_loss_tracker.max()}")
        CMA = self.kp_loss_tracker
        self.kp_loss_tracker =  CMA + (acc_loss - CMA) / n


    def set_poseopt_ckpt(self):
        print("set poseopt checkpoint")
        state_dict = self.poseopt_layer.state_dict()
        self.reset_state_dict = {k: state_dict[k].clone() for k in state_dict.keys()}

    def get_trackers(self, idx=None):
        if idx is None:
            idx = np.arange(len(self.kp_loss_tracker))

        kp_cnt = self.kp_loss_cnt[idx]
        ones = torch.ones_like(kp_cnt)
        kp_cnt = torch.maximum(kp_cnt, ones)
        return self.kp_loss_tracker[idx] / kp_cnt


    def step(self, i, kp_loss, kp_idx, anchors):
        """
        fed in global step
        """
        args = self.args
        if args.opt_pose_joint:
            self.accumulate_loss(kp_loss, kp_idx)
            self.optimizer.step()

            if i % args.opt_pose_step == 0:
                self.pose_optimizer.step()
                self.pose_optimizer.zero_grad()
                if self.poseopt_layer.use_cache:
                    self.poseopt_layer.update_cache()
            return self.get_trackers()

        just_turned = False
        if i % args.opt_pose_interval == 0:
            self.pose_turn = not self.pose_turn
            self.set_requires_grad()
            just_turned = True
            if self.pose_turn:
                if self.opt_pose_reset or self.opt_pose_mtn:
                    self.set_poseopt_ckpt()
                #self.reset_kp_loss_tracker()
            #else:
            #    #self.reset_nerf_loss_tracker()

        #self.accumulate_loss(losses, kp_idx, just_turned)


        # if it just switch from nerf->pose, nerf should still be updated (pose_opt started in the next iter)
        if (not self.pose_turn and not just_turned) or (self.pose_turn and just_turned):
            self.optimizer.step()
        elif i % args.opt_pose_step == 0:

            self.pose_optimizer.step()
            self.pose_optimizer.zero_grad()


            #if just_turned and not self.pose_turn and self.opt_pose_reset:
            #    self.reset_poseopt_by_loss()

            if just_turned and not self.pose_turn and self.opt_pose_mtn:
                self.update_poseopt_mtn()

            if self.poseopt_layer.use_cache:
                self.poseopt_layer.update_cache()

        return self.get_trackers()
