import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils.skeleton_utils import rot6d_to_rotmat, axisang_to_rot, axisang_to_quat, calculate_angle

# SKT (skeleton transformation-related)
def transform_batch_pts(pts, skt):

    N_rays, N_samples = pts.shape[:2]
    NJ = skt.shape[-3]

    if skt.shape[0] < pts.shape[0]:
        skt = skt.expand(pts.shape[0], *skt.shape[1:])

    # make it from (N_rays, N_samples, 4) to (N_rays, NJ, 4, N_samples)
    pts = torch.cat([pts, torch.ones(*pts.shape[:-1], 1)], dim=-1)
    pts = pts.view(N_rays, -1, N_samples, 4).expand(-1, NJ, -1, -1).transpose(3, 2).contiguous()
    # MM: (N_rays, NJ, 4, 4) x (N_rays, NJ, 4, N_samples) -> (N_rays, NJ, 4, N_samples)
    # permute back to (N_rays, N_samples, NJ, 4)
    mm = (skt @ pts).permute(0, 3, 1, 2).contiguous()

    return mm[..., :-1] # don't need the homogeneous part

def transform_batch_rays(rays_o, rays_d, skt):

    # apply only the rotational part
    N_rays, N_samples = rays_d.shape[:2]
    NJ = skt.shape[-3]
    rot = skt[..., :3, :3]

    if rot.shape[0] < rays_d.shape[0]:
        rot = rot.expand(rays_d.shape[0], *rot.shape[1:])
    rays_d = rays_d.view(N_rays, -1, N_samples, 3).expand(-1, NJ, -1, -1).transpose(3, 2).contiguous()
    mm = (rot @ rays_d).permute(0, 3, 1, 2).contiguous()

    return mm

class BaseEncoder(nn.Module):

    def __init__(self, N_joints=24, N_dims=None):
        super().__init__()
        self.N_joints = N_joints
        self.N_dims = N_dims if N_dims is not None else 1

    @property
    def dims(self):
        return self.N_joints * self.N_dims

    @property
    def encoder_name(self):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

class IdentityEncoder(BaseEncoder):

    @property
    def encoder_name(self):
        return 'Identity'

    @property
    def dims(self):
        return self.N_dims

    def forward(self, inputs, refs, *args, **kwargs):
        return inputs


class IdentityExpandEncoder(BaseEncoder):

    @property
    def encoder_name(self):
        return 'IdentityExpand'

    def forward(self, inputs, refs, *args, **kwargs):
        N_rays, N_samples = refs.shape[:2]
        return inputs[:, None].view(N_rays, 1, -1).expand(-1, N_samples, -1)

class WorldToLocalEncoder(BaseEncoder):

    @property
    def encoder_name(self):
        return 'W2LEncoder'

    def forward(self, pts, skts, *args, **kwargs):
        return transform_batch_pts(pts, skts)

class JointCenteredEncoder(BaseEncoder):

    @property
    def encoder_name(self):
        return 'JCEncoder'

    def forward(self, pts, skts, rots=None, bones=None,
                coords=None, ref=None, kps=None, *args, **kwargs):
        return pts[..., None, :] - kps[:, None]

# KP-position encoders
class RelDistEncoder(BaseEncoder):

    def __init__(self, N_joints=24, N_dims=1):
        super().__init__(N_joints, N_dims)

    @property
    def encoder_name(self):
        return 'RelDist'

    def forward(self, pts, pts_t, kps, *args, **kwargs):
        '''
        Args:
          pts (N_rays, N_pts, 3): 3d queries in world space.
          pts_t (N_rays, N_pts, N_joints, 3): 3d queries in local space (joints at (0, 0, 0)).
          kps (N_rays, N_joints, 3): 3d human keypoints.

        Returns:
          v (N_rays, N_pts, N_joints): relative distance encoding in the paper.
        '''
        if pts_t is not None:
            return torch.norm(pts_t, dim=-1, p=2)
        return torch.norm(pts[:, :, None] - kps[:, None], dim=-1, p=2)

class RelPosEncoder(BaseEncoder):

    def __init__(self, N_joints=24, N_dims=3):
        super().__init__(N_joints, N_dims)

    @property
    def dims(self):
        return self.N_joints * 3

    @property
    def encoder_name(self):
        return 'RelPos'

    def forward(self, pts, pts_t, kps, *args, **kwargs):
        '''Return relative postion in 3D
        '''
        if pts_t is not None:
            return pts_t.flatten(start_dim=-2)
        return (pts[:, :, None] - kps[:, None]).flatten(start_dim=-2)

class KPCatEncoder(BaseEncoder):

    def __init__(self, N_joints=24, N_dims=3):
        super().__init__(N_joints, N_dims)

    @property
    def dims(self):
        return self.N_joints * self.N_dims + self.N_dims

    @property
    def encoder_name(self):
        return 'KPCat'

    def forward(self, pts, pts_t, kps, *args, **kwargs):
        '''
        Args:
          pts (N_rays, N_pts, 3): 3d queries in world space
          pts_t (N_rays, N_pts, N_joints, 3): 3d queries in local space (joints at (0, 0, 0))
          kps (N_rays, N_joints, 3): 3d human keypoints

        Returns:
            cat (N_rays, N_pts, N_joints * 3 + 3): relative distance encoding in the paper.
        '''
        # to shape (N_rays, N_pts, N_joints * 3)
        kps = kps[:, None].expand(*pts.shape[:2], kps.shape[-2:]).flatten(start_dim=-2)
        return torch.cat([pts, kps], dim=-1)

# View/Bone encoding
class VecNormEncoder(BaseEncoder):

    def __init__(self, N_joints=24, N_dims=3):
        super().__init__(N_joints, N_dims)

    @property
    def encoder_name(self):
        return 'VecNorm'

    def forward(self, vecs, refs=None, *args, **kwargs):
        '''
        Args:
          vecs (N_rays, *, ...): vector to normalize.
          refs (N_rays, N_pts, ...): reference tensor for shape expansion.
        Returns:
          (N_rays, N_pts, *): normalized vector with expanded shape (if needed).
        '''
        n = F.normalize(vecs, dim=-1, p=2).flatten(start_dim=2)
        # expand to match N samples
        if refs is not None:
            n = n.expand(*refs.shape[:2], -1)
        return n

class RayAngEncoder(BaseEncoder):

    def __init__(self, N_joints=24, N_dims=1):
        super().__init__(N_joints, N_dims)

    @property
    def encoder_name(self):
        return 'RayAng'

    def forward(self, rays_t, pts_t, *args, **kwargs):
        '''
        Args:
          rays_t (N_rays, 1, N_joints, 3): rays direction in local space (joints at (0, 0, 0))
          pts_t (N_rays, N_pts, N_joints, 3): 3d queries in local space (joints at (0, 0, 0))
        Returns:
          d (N_rays, N_pts, N_joints*3): normalized ray direction in local space
        '''
        return calculate_angle(pts_t, rays_t)
