import torch
import numpy as np
from .skeleton_utils import get_kp_bounding_cylinder, cylinder_to_box_2d, swap_mat, nerf_c2w_to_extrinsic

# Ray helpers
def get_rays(H, W, focal, c2w, center=None):
    # i, j both of size (H, W)
    if isinstance(focal, float) or (len(focal.reshape(-1)) < 2):
        focal_x = focal_y = focal
    else:
        focal_x, focal_y = focal

    if center is None:
        offset_x, offset_y = W * 0.5, H * 0.5
    else:
        offset_x, offset_y = center

    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    # neg-y : image indexing is ascending from top to bottom. Make it descending and centered in the middle
    dirs = torch.stack([(i-offset_x)/focal_x, -(j-offset_y)/focal_y, -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    # TODO: verify this
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, focal, c2w, mesh=None, center=None):

    if mesh is None:
        i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    else:
        i, j = mesh

    if isinstance(focal, float) or (len(focal.reshape(-1)) < 2):
        focal_x = focal_y = focal
    else:
        focal_x, focal_y = focal

    if center is None:
        offset_x = W * 0.5
        offset_y = H * 0.5
    else:
        offset_x, offset_y = center
    #dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    dirs = np.stack([(i-offset_x)/focal_x, -(j-offset_y)/focal_y, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    I = np.eye(3)
    if np.isclose(I, c2w[:3, :3]).all():
        rays_d = dirs # no rotation required if rotation is identity
    elif np.isclose(I, np.abs(c2w[:3, :3])).all():
        signs = c2w[:3, :3].sum(-1)
        rays_d = dirs * signs
    else:
        rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]

    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)

    return rays_o, rays_d

def kp_to_valid_rays(poses, H, W, focal, kps=None, cylinder_params=None,
                     skts=None, centers=None, ext_scale=0.00035):
    if cylinder_params is None:
        assert kps is not None
        kps_np = kps.cpu().numpy()
        #expand_ratio = 1.10
        bot_expand_ratio = 1.10
        # For MonoPerfCap Evaluation
        top_expand_ratio = 1.60
        # For Surreal
        #top_expand_ratio = 1.10
        extend_mm = 250

        # neuralbody
        #bot_expand_ratio = 1.0
        #top_expand_ratio = 1.1
        #extend_mm = 100
        cylinder_params = get_kp_bounding_cylinder(kps_np, ext_scale=ext_scale,
                                                   extend_mm=extend_mm,
                                                   top_expand_ratio=top_expand_ratio,
                                                   bot_expand_ratio=bot_expand_ratio,
                                                   head='-y')
        cylinder_params = torch.FloatTensor(cylinder_params).to(kps.device)

    valid_idxs = []
    rays = []
    bboxes = []
    for i, c2w in enumerate(poses):
        # assume #kp <= #poses,
        # and we want to render images from the same pose consecutively
        cyl_idx = i % kps.shape[0]
        cyl_param = cylinder_params[cyl_idx]
        f = focal if isinstance(focal, float) else focal[i]
        center = None if centers is None else centers[i]
        h = H if isinstance(H, int) else H[i]
        w = W if isinstance(W, int) else W[i]

        ray_o, ray_d = get_rays(h, w, f, c2w, center=center)

        #w2c = np.linalg.inv(swap_mat(c2w.cpu().numpy()))
        w2c = nerf_c2w_to_extrinsic(c2w.cpu().numpy())
        tl, br, _ = cylinder_to_box_2d(cyl_param.cpu().numpy(), [h, w, f], w2c,
                                       center=center)

        h_range = torch.arange(tl[1], br[1])
        w_range = torch.arange(tl[0], br[0])
        valid_h, valid_w = torch.meshgrid(h_range, w_range)
        valid_idx =  (valid_h * w + valid_w).view(-1)

        rays.append((ray_o.view(-1, 3)[valid_idx], ray_d.view(-1, 3)[valid_idx]))
        valid_idxs.append(valid_idx)
        bboxes.append((tl, br))

    return rays, valid_idxs, cylinder_params, bboxes

def get_corner_rays(H, W, focal, poses):
    rays_o, rays_d = [], []

    for p in poses:
        os, ds = get_rays_np(H, W, focal, p)
        #ds = ds / np.linalg.norm(ds, axis=-1)[..., None]
        rays_o.append(os[0, 0])
        # tl, tr, br, bl
        rays_d.append([ds[0, 0],
                       ds[0, -1],
                       ds[-1, -1],
                       ds[-1, 0]])

    # original of rays, and the dir. to 4 corners on the image plane
    return rays_o, rays_d



# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf.detach(), u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])


    return samples

# Sampling
def sample_from_lineseg(near, far, N_lines, N_samples, perturb=0., lindisp=False, pytest=False):
    """
    Sample points on line segments (rays) defined by near (start point) and far (end point).
    --
    near, far: start and endpoints of the line segments
    N_samples: number of sample points for each line segment
    N_lines: replicate the sampled points for this many lines
    perturb: noise added when sampling the points
    lindisp: whether sample on the depth (length) or the inverse depth (length)
    pytest: to use pytest
    Return:
    z_vals: sampled points on the line segments
    """

    t_vals = torch.linspace(0., 1., steps=N_samples)

    if not isinstance(near, float) and not isinstance(far, float):
        t_vals = t_vals.expand([near.size(0), N_samples])

    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    if isinstance(near, float) and isinstance(far, float):
        z_vals = z_vals.expand([N_lines, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        # set lower bound and upper bound for sampling
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand
    if torch.isnan(z_vals).any():
        import pdb; pdb.set_trace()
        print

    return z_vals



def isample_from_lineseg(z_vals, weights, N_importance,
                         det=False, pytest=False, is_only=False,
                         alpha_base=0.01):
    """
    Importance sampling on the line segments
    --
    z_vals: original sample points to center on
    weights: weighted distribution for importance sampling
    N_importance: number of importance samples
    det: deterministic sampling
    pytest: pytest flag
    Return:
    z_vals: original and weighted sample points on the line segments
    z_samples: weighted sample points
    """
    z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
    if is_only:
        # weights = 0.5 * (max(w_l, w_k) + max(w_k, w_u)) + alpha
        w_l = weights[..., 0:-2]
        w_k = weights[..., 1:-1]
        w_u = weights[..., 2:]

        dist_weights = 0.5 * (torch.maximum(w_l, w_k) + torch.maximum(w_k, w_u)) + alpha_base
    else:
        dist_weights = weights[..., 1:-1]

    #z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance,
    #                       det=det, pytest=pytest)
    z_samples = sample_pdf(z_vals_mid, dist_weights, N_importance,
                           det=det, pytest=pytest)
    z_samples = z_samples.detach()

    # merge both new samples (z_samples) and coarse samples (z_vals)
    z_vals, sorted_idxs = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
    return z_vals, z_samples, sorted_idxs


def get_near_far_in_cylinder(rays_o, rays_d, cyl, near=0.35, far=2.75,
                             g_axes=[0, -1]):
    '''
    g_axes: ground plane, by default it's x-z
    '''

    r_near = (rays_o + rays_d * near)[..., g_axes]
    r_far = (rays_o + rays_d * far)[..., g_axes]

    radius = cyl[..., 2:3]
    center = cyl[..., :2]

    nc = center - r_near # near to center
    nf = r_far - r_near  # near to far
    nf_norm = torch.norm(nf, dim=-1, p=2)

    ray_dir = nf / nf_norm[..., None]
    scale = torch.norm(rays_d[..., g_axes], dim=-1, p=2)[..., None]

    # find distance from center to the ray
    cross = nc[..., 0] * nf[..., 1] - nc[..., 1] * nf[..., 0]
    dist = (torch.abs(cross) / nf_norm)[..., None]

    # pythagorean theorem  (R^2 - D^2) = Q^2
    # where Q is the distance from the closest point to intersecting points
    Q = (radius.pow(2) - dist.pow(2)).pow(0.5)
    # distance from near to the closest point (to the center)
    K = ((nc * nf).sum(-1) / nf_norm)[..., None]
    mask = (Q < K).float() # Q > K indicates that the near point is inside the circle

    # new_near = (r_near + ray_dir * (K - Q)) (1. - mask) + mask * r_near
    # new_far = r_near + ray_dir * (K + Q)    # use original near plane if Q > K
    # use original near plane if Q > K
    new_near = near + mask * (K - Q) / scale
    new_far = near + (K + Q) / scale

    if torch.isnan(new_near).any():
        # due to the coarse resolution, some rays lie slightly outside of the cylinder.
        # find these cases (Q == nan) and fill in proper near/far
        idx, _ = torch.where(torch.isnan(Q))
        avg_near = np.nanmean(new_near.cpu().numpy())
        if not np.isnan(avg_near):
            new_near[idx, :] = float(avg_near)
        else:
            new_near[idx, :] = near[idx, :]

        avg_far = np.nanmean(new_far.cpu().numpy())
        if not np.isnan(avg_far):
            new_far[idx, :] = float(avg_far)
        else:
            new_far[idx, :] = far[idx, :]

    return new_near, new_far

def get_near_far_in_cylinder_np(rays_o, rays_d, cyl, near=0.35, far=2.75):

    r_near = (rays_o + rays_d * near)[..., [0, -1]]
    r_far = (rays_o + rays_d * far)[..., [0, -1]]

    radius = cyl[..., 2:3]
    center = cyl[..., :2]

    nc = center - r_near # near to center
    nf = r_far - r_near  # near to far
    nf_norm = np.linalg.norm(nf, axis=-1)

    ray_dir = nf / nf_norm[..., None]
    scale = np.linalg.norm(rays_d[..., [0, -1]], axis=-1)[..., None]

    # find distance from center to the ray
    cross = nc[..., 0] * nf[..., 1] - nc[..., 1] * nf[..., 0]
    dist = (np.abs(cross) / nf_norm)[..., None]

    # pythagorean theorem  (R^2 - D^2) = Q^2
    # where Q is the distance from the closest point to intersecting points
    Q = (radius**2 - dist**2)**0.5
    # distance from near to the closest point (to the center)
    K = ((nc * nf).sum(-1) / nf_norm)[..., None]
    mask = (Q < K).astype(np.float32) # Q > K indicates that the near point is inside the circle


    # new_near = (r_near + ray_dir * (K - Q)) (1. - mask) + mask * r_near
    # new_far = r_near + ray_dir * (K + Q)    # use original near plane if Q > K
    # use original near plane if Q > K
    new_near = near + mask * (K - Q) / scale
    new_far = near + (K + Q) / scale

    return new_near, new_far
