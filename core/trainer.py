import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# TODO: loss fns, put them in a different file later
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

def img2mse(x, y, reduction="mean", to_yuv=False, scale_yuv=[0.1, 1.0, 1.0]):

    if to_yuv:
        x = rgb_to_yuv(x)
        y = rgb_to_yuv(y)
        scale_yuv = torch.tensor(scale_yuv).float().view(1, 3)
        sqr_diff = (x - y)**2 * scale_yuv
    else:
        sqr_diff = (x - y)**2

    if reduction == "mean":
        return torch.mean(sqr_diff)
    elif reduction == "sum":
        return torch.sum(sqr_diff)
    else:
        return sqr_diff

def img2l1(x, y, reduction="mean", to_yuv=False, scale_yuv=[0.1, 1.0, 1.0]):

    if to_yuv:
        x = rgb_to_yuv(x)
        y = rgb_to_yuv(y)
        scale_yuv = torch.tensor(scale_yuv).float().view(1, 3)
        diff = (x - y).abs() * scale_yuv
    else:
        diff = (x - y).abs()

    if reduction == "mean":
        return torch.mean(diff)
    elif reduction == "sum":
        return torch.sum(diff)
    else:
        return diff

def acc2bce(x, y, reduction="mean", eps=1e-8):
    bce_loss = -(y * torch.log(x + eps) + (1. - y) * torch.log(1 - x + eps))
    if reduction == "mean":
        return torch.mean(bce_loss)
    elif reduction == "sum":
        return torch.sum(bce_loss)
    elif reduction == "off":
        bce_loss = bce_loss[y < 1.0]
        return torch.mean(bce_loss)
    else:
        return bce_loss


def img2huber(x, y, reduction="mean", beta=0.1):
    return F.smooth_l1_loss(x, y, reduction=reduction, beta=beta)

def img2psnr(img, target):
    return mse2psnr(img2mse(img, target))


def batchify_rays(rays_flat, chunk=1024*32, ray_caster=None, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}

    for i in range(0, rays_flat.shape[0], chunk):
        batch_kwargs = {k: kwargs[k][i:i+chunk] if torch.is_tensor(kwargs[k]) else kwargs[k]
                        for k in kwargs}
        ret = ray_caster(rays_flat[i:i+chunk], **batch_kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, focal, chunk=1024*32, rays=None, c2w=None,
           near=0., far=1., center=None,
           use_viewdirs=False, c2w_staticcam=None,
            **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if (c2w is not None) and (rays is None):
        # special case to render full image
        center = center.ravel() if center is not None else None
        import pdb; pdb.set_trace()
        rays_o, rays_d = get_rays(H, W, focal, c2w, center=center)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
        # normalize ray length
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    # Create ray batch
    sh = rays_d.shape # [..., 3]
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    #rays = torch.cat([rays_o, rays_d, near, far], -1)
    rays = torch.cat([rays_o.to(near.device), rays_d.to(near.device), near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        if all_ret[k].dim() >= 4:
            continue
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    return all_ret

def get_loss_fn(loss_name, beta):
    if loss_name == "MSE":
        loss_fn = img2mse
    elif loss_name == "L1":
        loss_fn = img2l1
    elif loss_name == "Huber":
        loss_fn = lambda x, y, reduction='mean', beta=beta: img2huber(x, y, reduction, beta)
    else:
        raise NotImplementedError

    return loss_fn

def get_reg_fn(reg_name):
    if reg_name is None:
        return None
    if reg_name == "L1":
        reg_fn = img2l1
    elif reg_name == "MSE":
        reg_fn = img2mse
    elif reg_name == "BCE":
        reg_fn = acc2bce
    else:
        raise NotImplementedError
    return reg_fn

# helper funcs
def decay_optimizer_lrate(lrate, lrate_decay, decay_rate, optimizer,
                          global_step=None, decay_unit=1000):

    #decay_steps = lrate_decay * decay_unit
    decay_steps = lrate_decay
    optim_step = optimizer.state[optimizer.param_groups[0]['params'][0]]['step'] // decay_unit
    #new_lrate = lrate * (decay_rate ** (global_step / decay_steps))
    new_lrate = lrate * (decay_rate ** (optim_step / decay_steps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lrate
    return new_lrate, None

def dict_to_device(d, device):
    return {k: d[k].to(device) if d[k] is not None else None for k in d}

def set_tag_name(name, coarse):
    return name if not coarse else name + '0'

@torch.no_grad()
def get_gradnorm(module):
    total_norm  = 0.0
    cnt = 0
    for p in module.parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
        cnt += 1
    avg_norm = (total_norm / cnt) ** 0.5
    total_norm = total_norm ** 0.5
    return total_norm, avg_norm

class Trainer:

    def __init__(self, args, data_attrs, optimizer, pose_optimizer,
                 render_kwargs_train, render_kwargs_test, popt_kwargs=None,
                 device=None):
        '''
        args: args for training
        data_attrs: data attributes needed for training
        optimizer: optimizer for NeRF network
        pose_optimizer: optimizer for poses
        render_kwargs_train: kwargs for rendering outputs on training batch
        render_kwargs_test: kwargs for rendering testing data
        popt_kwargs: pose optimization layer and relevant keyword arguments
        '''
        self.args = args
        self.optimizer = optimizer
        self.pose_optimizer = pose_optimizer
        self.render_kwargs_train = render_kwargs_train
        self.render_kwargs_test = render_kwargs_test
        self.popt_kwargs = popt_kwargs
        self.device = device

        self.hwf = data_attrs['hwf']
        self.data_attrs = data_attrs

    def train_batch(self, batch, i=0, global_step=0):
        '''
        i: current iteration
        global_step: global step (TODO: does this matter?)
        '''
        args = self.args
        H, W, focal = self.hwf
        batch = dict_to_device(batch, self.device)

        # step 1a: get keypoint/human poses data
        popt_detach = not (args.opt_pose_stop is None or i < args.opt_pose_stop)
        kp_args, extra_args = self.get_kp_args(batch, detach=popt_detach)

        # step 1b: get other args for forwarding
        fwd_args = self.get_fwd_args(batch)

        # step 2: ray caster forward
        preds = render(H, W, focal, chunk=args.chunk, verbose=i < 10,
                      retraw=False, **kp_args, **fwd_args, **self.render_kwargs_train)

        # step 3: loss computation and updates
        loss_dict, stats = self.compute_loss(batch, preds, kp_opts={**kp_args, **extra_args},
                                             popt_detach=popt_detach or not args.opt_pose)

        optim_stats = self.optimize(loss_dict['total_loss'], i, popt_detach or not args.opt_pose)

        # step 4: rate decay
        new_lrate, _ = decay_optimizer_lrate(args.lrate, args.lrate_decay,
                                             decay_rate=args.lrate_decay_rate,
                                             optimizer=self.optimizer,
                                             global_step=global_step,
                                             decay_unit=args.decay_unit)

        ray_caster = self.render_kwargs_train['ray_caster']
        if not args.finetune:
            ray_caster.module.update_embed_fns(global_step, args)

        # step 5: logging
        stats = {'lrate': new_lrate,
                 'alpha': preds['acc_map'].mean().item(),
                 'cutoff': ray_caster.module.embed_fn.get_tau(),
                 **stats, **optim_stats}

        return loss_dict, stats

    def get_fwd_args(self, batch):
        '''
        get basic data required for ray casting from batch
        '''
        return {
            'rays': batch['rays'],
            'cams': batch['cam_idxs'] if self.args.opt_framecode else None,
            'subject_idxs': batch.get('subject_idxs', None),
        }

    def get_kp_args(self, batch, detach=False):
        '''
        get human pose data from either batch or popt layer
        '''

        extras = {}
        if self.popt_kwargs is None or self.popt_kwargs['popt_layer'] is None:
            return self._create_kp_args_dict(kps=batch['kp3d'], skts=batch['skts'],
                                             bones=batch['bones'], cyls=batch['cyls']), extras

        kp_idx = batch['kp_idx']
        if torch.is_tensor(kp_idx):
            kp_idx = kp_idx.cpu().numpy()

        # forward to get optimized pose data
        popt_layer = self.popt_kwargs['popt_layer']
        kps, bones, skts, _, rots = popt_layer(kp_idx)
        kp_args = self._create_kp_args_dict(kps=kps, skts=skts,
                                            bones=bones, cyls=batch['cyls'])

        # not for ray casting, for loss computation only.
        extras['rots'] = rots

        if detach:
            kp_args = {k: kp_args[k].detach() for k in kp_args if kp_args[k] is not None}
            extras = {k: extras[k].detach() for k in extras if extras[k] is not None}

        return kp_args, extras


    def _create_kp_args_dict(self, kps, skts, bones=None, cyls=None, rots=None):
        # TODO: unified the variable names (across all functions)?
        return {'kp_batch': kps, 'skts': skts, 'bones': bones, 'cyls': cyls}

    def compute_loss(self, batch, preds,
                     kp_opts=None, popt_detach=False):
        '''
        popt_detach: False if we need kp_loss

        Assume returns from _compute_loss functions are tuples of (losses, stats)
        '''
        args = self.args

        total_loss = 0.
        results = []

        # rgb loss of nerf
        results.append(self._compute_nerf_loss(batch, preds['rgb_map'], preds['acc_map']))
        if 'rgb0' in preds:
            results.append(self._compute_nerf_loss(batch, preds['rgb0'], preds['acc0'], coarse=True))

        # need to update human poses, computes it.
        if not popt_detach:
            results.append(self._compute_kp_loss(batch, kp_opts))

        # collect losses and stats
        loss_dict, stats = {}, {}
        for i, (loss, stat) in enumerate(results):
            for k in loss:
                total_loss = total_loss + loss[k]
                loss_dict[k] = loss[k]
            for k in stat:
                stats[k] = stat[k].item()

        loss_dict['total_loss'] = total_loss

        return loss_dict, stats

    def _compute_nerf_loss(self, batch, rgb_pred, acc_pred, base_bg=1.0,
                           coarse=False):
        '''compute loss
        TODO: base_bg from argparse?
        '''
        args = self.args
        loss_fn = get_loss_fn(args.loss_fn, args.loss_beta)
        reg_fn = get_reg_fn(args.reg_fn)

        bgs = batch['bgs'] if 'bgs' in batch else base_bg
        if args.use_background:
            rgb_pred = rgb_pred + (1. - acc_pred)[..., None] * bgs
        rgb_loss = loss_fn(rgb_pred, batch['target_s'], reduction='mean')
        if coarse:
            rgb_loss = rgb_loss * args.coarse_weight
        psnr = img2psnr(rgb_pred.detach(), batch['target_s'])

        loss_tag = set_tag_name(f'rgb_loss', coarse)
        stat_tag = set_tag_name(f'psnr', coarse)

        loss_dict = {loss_tag: rgb_loss}

        if reg_fn is not None:
            reg_loss = reg_fn(acc_pred, batch['fgs'][..., 0], reduction='off') * args.reg_coef
            reg_tag = set_tag_name('reg_loss', coarse)
            loss_dict[reg_tag] = reg_loss

        return loss_dict, {stat_tag: psnr}

    def _compute_kp_loss(self, batch, kp_opts):

        args = self.args
        anchors = self.popt_kwargs['popt_anchors']
        kp_idx = batch['kp_idx']

        # shapes (N_rays, N_joints, *)
        if args.opt_rot6d:
            reg_bones = anchors['rots'][kp_idx, ..., :3, :2].flatten(start_dim=-2)
            bones = kp_opts['rots'][..., :3, :2].flatten(start_dim=-2)
        else:
            reg_bones = anchors['bones'][kp_idx]
            bones = kp_opts['bones']
        assert len(reg_bones) == len(bones)

        # loss = 0 if bone_loss < tol
        # loss = bone_loss - tol otherwise
        tol = args.opt_pose_tol
        kp_loss = (reg_bones - bones).pow(2.)[:, 1:] # exclude root (idx = 0)
        loss_mask = (kp_loss > tol).float()
        kp_loss = torch.lerp(torch.zeros_like(kp_loss),kp_loss-tol, loss_mask).sum(-1)
        kp_loss = kp_loss.mean() * args.opt_pose_coef

        loss_dict = {'kp_loss': kp_loss}
        # TODO: add temporal loss
        if args.use_temp_loss:
            popt_layer = self.popt_kwargs['popt_layer']
            # obtain indidces of previous/next pose
            kp_idx_prev = batch['kp_idx'] - 1
            kp_idx_next = (batch['kp_idx'] + 1) % len(popt_layer.bones)
            if torch.is_tensor(kp_idx_prev):
                kp_idx_prev = kp_idx_prev.cpu().numpy()
                kp_idx_next = kp_idx_next.cpu().numpy()

            # obtain optimized prev/next poses
            prev_kps, prev_bones, _, _, prev_rots = popt_layer(kp_idx_prev)
            next_kps, next_bones, _, _, next_rots = popt_layer(kp_idx_next)

            if args.opt_rot6d:
                prev_bones = prev_rots[..., :3, :2].flatten(start_dim=-2)
                next_bones = next_rots[..., :3, :2].flatten(start_dim=-2)

            # detach the regualizer
            prev_kps, prev_bones = prev_kps.detach(), prev_bones.detach()
            next_kps, next_bones = next_kps.detach(), next_bones.detach()
            temp_valid = batch['temp_val']
            kps = kp_opts['kp_batch']

            ang_vel = ((bones - prev_bones) - (next_bones - bones)).pow(2.).sum(-1)
            joint_vel = ((kps - prev_kps) - (next_kps - kps)).pow(2.).sum(-1)
            temp_loss = (ang_vel + joint_vel) * temp_valid[..., None]
            temp_loss = temp_loss.mean() * args.temp_coef

            loss_dict['temp_loss'] = temp_loss

        # mean per-joint change
        pjpc = (anchors['kps'][kp_idx] - kp_opts['kp_batch'].detach()).pow(2.).sum(-1).pow(0.5)
        mpjpc = pjpc.mean() / args.ext_scale

        return loss_dict, {'MPJPC': mpjpc}

    def _optim_step(self):
        '''
        step the optimizers for per-iteration parameter updates.
        '''

        self.optimizer.step()
        self.optimizer.zero_grad()

    def optimize(self, loss, i, popt_detach=False):
        '''step the optimizer
        loss: loss tensor to backward
        i: current iteration
        popt_detach: True if no need to update poses
        '''
        args = self.args

        # no pose optimization required, just backward.
        if popt_detach:
            loss.backward()
            total_norm, avg_norm = get_gradnorm(self.render_kwargs_train['ray_caster'])
            self._optim_step()
            return {'total_norm': total_norm, 'avg_norm': avg_norm}

        # only need to retain comp. graph if step between pose update > 1
        retain_graph = args.opt_pose_cache and args.opt_pose_step > 1
        loss.backward(retain_graph=retain_graph)

        # reset gradient immediately after parameter update
        self._optim_step()

        total_norm, avg_norm = get_gradnorm(self.render_kwargs_train['ray_caster'])

        # update pose after enough gradient accumulation
        if i % args.opt_pose_step == 0:
            self.pose_optimizer.step()
            self.pose_optimizer.zero_grad()

            if args.opt_pose_cache:
                self.popt_kwargs['popt_layer'].update_cache()

        return {'total_norm': total_norm, 'avg_norm': avg_norm}

    def save_nerf(self, path, global_step):
        '''Save all state dict.
        path: path to save data
        global_step: global training step
        '''
        args = self.args
        ray_caster = self.render_kwargs_train['ray_caster']
        popt_sd, poptim_sd, popt_anchors = None, None, None
        if self.popt_kwargs is not None:
            popt_sd = self.popt_kwargs['popt_layer'].state_dict()
            poptim_sd = self.pose_optimizer.state_dict()
            popt_anchors = self.popt_kwargs['popt_anchors']

        torch.save({
            'global_step': global_step,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'poseopt_layer_state_dict': popt_sd,
            'pose_optimizer_state_dict': poptim_sd,
            'poseopt_anchors': popt_anchors,
            **ray_caster.module.state_dict(),
        }, path)
        print('Saved checkpoints at', path)

    def save_popt(self, path, global_step):
        '''Save pose optimization outcomes
        '''
        args = self.args
        torch.save({'global_step': global_step,
                    'poseopt_layer_state_dict': self.popt_kwargs['popt_layer'].state_dict(),
                    'poseopt_anchors': self.popt_kwargs['popt_anchors'],
                    }, path)
        print('Saved pose at', path)

