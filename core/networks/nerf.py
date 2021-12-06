from .embedding import Optcodes

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import itertools

# Standard NeRF
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_bones=0, input_ch_views=3,
                 output_ch=4, skips=[4], use_viewdirs=False, use_framecode=False,
                 framecode_ch=16, n_framecodes=0, skel_type=None, density_scale=1.0):
        """
        idx_maps: for mapping framecode_idx
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W

        self.input_ch = input_ch
        self.input_ch_bones = input_ch_bones
        self.input_ch_views = input_ch_views

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.use_framecode = use_framecode
        self.framecode_ch = framecode_ch
        self.n_framecodes = n_framecodes

        self.cam_ch = 1 if self.use_framecode else 0

        self.N_joints = 24
        self.output_ch = output_ch
        self.skel_type = skel_type
        self.density_scale = density_scale

        self.init_density_net()
        self.init_radiance_net()

        ## Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)

    @property
    def dnet_input(self):
        # input size of density network
        return self.input_ch + self.input_ch_bones

    @property
    def vnet_input(self):
        # input size of radiance (view) network
        view_ch_offset = 0 if not self.use_framecode else self.framecode_ch
        return self.input_ch_views + view_ch_offset + self.W

    def init_density_net(self):

        W, D = self.W, self.D

        layers = [nn.Linear(self.dnet_input, W)]

        for i in range(D-1):
            if i not in self.skips:
                layers += [nn.Linear(W, W)]
            else:
                layers += [nn.Linear(W + self.dnet_input, W)]

        self.pts_linears = nn.ModuleList(layers)

        if self.use_viewdirs:
            self.alpha_linear = nn.Linear(W, 1)

    def init_radiance_net(self):

        W = self.W

        # Note: legacy code, don't really need nn.ModuleList
        self.views_linears = nn.ModuleList([nn.Linear(self.vnet_input, W//2)])

        if self.use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, self.output_ch)

        if self.use_framecode:
            self.framecodes = Optcodes(self.n_framecodes, self.framecode_ch)

    def forward_batchify(self, inputs, chunk=1024*64, **kwargs):
        return torch.cat([self.forward(inputs[i:i+chunk], **{k: kwargs[k][i:i+chunk] for k in kwargs})
                          for i in range(0, inputs.shape[0], chunk)], -2)

    def forward_density(self, input_pts):
        h = input_pts

        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
        return h

    def forward_feature(self, input_pts, input_views):
        h = self.forward_density(input_pts)
        if self.use_framecode:
            input_views, framecode_idx = torch.split(input_views, [self.input_ch_views-1, 1], dim=-1)
            framecodes = self.framecodes(framecode_idx, None)
            input_views = torch.cat([input_views, framecodes], dim=-1)
        feature = self.feature_linear(h)
        h = torch.cat([feature, input_views], -1)
        return self.views_linears[0](h)

    def forward_view(self, input_views, h, frame_idxs):
        '''
        input_views: ray direction encoding
        h: feature from density network
        frame_idxs: specifying index for retrieving frame code
        '''
        # produce features for color/radiance
        feature = self.feature_linear(h)
        if self.use_framecode:
            framecodes = self.framecodes(frame_idxs)
            input_views = torch.cat([input_views, framecodes], dim=-1)
        h = torch.cat([feature, input_views], -1)

        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h, inplace=True)

        return self.rgb_linear(h)

    def forward(self, x):
        # standard NeRF
        input_pts, input_views, frame_idxs = torch.split(x, [self.input_ch + self.input_ch_bones,
                                                             self.input_ch_views, self.cam_ch],
                                                         dim=-1)
        h = self.forward_density(input_pts)

        if self.use_viewdirs:
            # predict density and radiance separately
            alpha = self.alpha_linear(h)
            rgb = self.forward_view(input_views, h, frame_idxs)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

    def raw2outputs(self, raw, z_vals, rays_d, raw_noise_std=0, pytest=False,
                    B=0.01, rgb_act=torch.sigmoid, act_fn=F.relu, rgb_eps=0.001, **kwargs):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            raw: [num_rays, num_samples along ray, 4]. Prediction from model.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            acc_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.
        """
        raw2alpha = lambda raw, dists, noise, act_fn=act_fn: 1.-torch.exp(-(act_fn(raw/B + noise))*dists)

        dists = z_vals[...,1:] - z_vals[...,:-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

        rgb = rgb_act(raw[...,:3]) * (1 + 2 * rgb_eps) - rgb_eps # [N_rays, N_samples, 3]
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn(raw[...,3].shape) * raw_noise_std * B

            # Overwrite randomly sampled data if pytest
            if pytest:
                np.random.seed(0)
                noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
                noise = torch.Tensor(noise)

        alpha = raw2alpha(raw[...,3], dists, noise)  # [N_rays, N_samples]

        # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
        # sum_{i=1 to N samples} prob_of_already_hit_particles * alpha_for_i * color_for_i
        # C(r) = sum [T_i * (1 - exp(-sigma_i * delta_i)) * c_i] = sum [T_i * alpha_i * c_i]
        # alpha_i = 1 - exp(-sigma_i * delta_i)
        # T_i = exp(sum_{j=1 to i-1} -sigma_j * delta_j) = torch.cumprod(1 - alpha_i)
        # standard NeRF
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / (torch.sum(weights, -1)  + 1e-10))

        invalid_mask = torch.ones_like(disp_map)
        invalid_mask[torch.isclose(weights.sum(-1), torch.tensor(0.))] = 0.
        disp_map = disp_map * invalid_mask

        acc_map = torch.minimum(torch.sum(weights, -1), torch.tensor(1.))

        return {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map,
                'weights': weights, 'alpha': alpha}

