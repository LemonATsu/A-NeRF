import os, sys
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.profiler as profiler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

# our own codes
from core.trainer import *
from core.utils.ray_utils import *
from core.utils.run_nerf_helpers import *
from core.utils.evaluation_helpers import evaluate_metric, evaluate_pampjpe_from_smpl_params
from core.utils.skeleton_utils import draw_skeletons_3d

from core.raycasters import create_raycaster
from core.pose_opt import create_popt
from core.load_data import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = True


@torch.no_grad()
def render_path(render_poses, hwf, chunk, render_kwargs,
                centers=None, kp=None, skts=None, cyls=None, bones=None,
                gt_imgs=None, bg_imgs=None, bg_indices=None,
                cams=None, subject_idxs=None, render_factor=0,
                white_bkgd=False, ret_acc=False,
                ext_scale=0.00035, base_bg=1.0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        if isinstance(focal, float):
            focal = focal / render_factor
            if centers is not None:
                centers = centers / render_factor
        else:
            focal = focal.copy() / render_factor
            if centers is not None:
                centers = centers.copy() / render_factor

    if kp is not None or cyls is not None:
        # only render part of the image
        rays, valid_idxs, cyls, bboxes = kp_to_valid_rays(render_poses, H, W,
                                                          focal, kps=kp, cylinder_params=cyls,
                                                          skts=skts, ext_scale=ext_scale,
                                                          centers=centers)
    else:
        rays, valid_idxs = None, None

    rgbs, disps, accs = [], [], []

    t = time.time()
    # reuse human pose if #render_poses > #human_poses and #human_poses > 0
    def reuse_input(x, expand=None):
        if x is None:
            return x

        if x.shape[0] > 1:
            y = x[i%x.shape[0]:i%x.shape[0]+1].clone()
        else:
            y = x.clone()

        if expand is not None:
            y = y.expand(expand, *x.shape[1:])
        return y
    #reuse_input = lambda x: x[i%x.shape[0]:i%x.shape[0]+1] if x is not None and x.shape[0] > 1 else x

    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        h = H if isinstance(H, int) else H[i]
        w = W if isinstance(W, int) else W[i]

        ray_input = rays[i] if rays is not None else None
        expand = len(ray_input[0])
        kp_input = reuse_input(kp, expand)
        skt_input = reuse_input(skts, expand)
        cyl_input = reuse_input(cyls, expand)
        bone_input = reuse_input(bones, expand)
        cam_input = reuse_input(cams, expand)
        subject_input = reuse_input(subject_idxs, expand)
        #center_input = reuse_input(centers, expand)

        if len(ray_input[0]) > 0:
            ret_dict = render(h, w, focal, rays=ray_input, chunk=chunk, c2w=c2w[:3,:4],
                              kp_batch=kp_input, skts=skt_input, cyls=cyl_input, cams=cam_input,
                              subject_idxs=subject_input,
                              bones=bone_input, **render_kwargs)
            rgb, disp, acc = ret_dict['rgb_map'], ret_dict['disp_map'], ret_dict['acc_map']

        if valid_idxs is not None:
            # in this case, we only render the rays that are within the foreground or cylinder
            valid_idx = valid_idxs[i]

            # initialize the images
            if bg_imgs is not None and not white_bkgd:

                if bg_indices is not None:
                    bg = torch.tensor(bg_imgs[bg_indices[i]]).permute(2, 0, 1)[None]
                else:
                    # TODO: HACK, FIX THIS
                    bg = torch.tensor(bg_imgs[0]).permute(2, 0, 1)[None]
                rgb_img = F.interpolate(bg, size=(h, w), mode='bilinear',
                                   align_corners=False)[0].permute(1, 2, 0).view(h * w, 3)
            else:
                rgb_img = torch.zeros(h * w, 3) if not white_bkgd else torch.ones(h * w, 3)
            disp_img = torch.zeros(h * w)

            if len(valid_idx) > 0:
                # check if we have rays for this image
                bg = (1. - acc[..., None]) * rgb_img[valid_idx]

                # assign values to the corresponding ray location
                rgb_img[valid_idx] = rgb + bg
                disp_img[valid_idx] = disp
                if ret_acc:
                    acc_img = torch.zeros(h * w)
                    acc_img[valid_idx] = acc
                    accs.append(acc_img.view(h, w, 1).detach().cpu().numpy())

            rgbs.append(rgb_img.view(h, w, 3).detach().cpu().numpy())
            disps.append(disp_img.view(h, w, 1).detach().cpu().numpy())

        else:
            # render the whole image in this case, can append directly
            rgbs.append(rgb.cpu().numpy())
            disps.append(disp.cpu().numpy())

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)
    disps_nan = np.isnan(disps)
    disps[disps_nan] = 0.
    if ret_acc:
        accs = np.stack(accs, 0)

    return rgbs, disps, accs, valid_idxs, bboxes

@torch.no_grad()
def render_testset(poses, hwf, args, render_kwargs, kps=None, skts=None, cyls=None, cams=None,
                   bones=None, subject_idxs=None, gt_imgs=None, gt_masks=None, bg_imgs=None, bg_indices=None,
                   vid_base=None, rgb_vid="rgb.mp4", disp_vid="disp.mp4", eval_metrics=False,
                   render_factor=0, eval_postfix="", save_npy=False, eval_both=False, centers=None,
                   save_image=False):

    render_kwargs["ray_caster"].eval()
    rgbs, disps, _, valid_idxs, bboxes = render_path(poses, hwf, args.chunk//8, render_kwargs,
                                                     bg_imgs=bg_imgs, bg_indices=bg_indices,
                                                     centers=centers, kp=kps, skts=skts, cyls=cyls, bones=bones,
                                                     cams=cams, subject_idxs=subject_idxs, render_factor=args.render_factor,
                                                     ext_scale=args.ext_scale, white_bkgd=args.white_bkgd)
    render_kwargs["ray_caster"].train()
    if save_image:
        print("Warning: this is super hacky. Should terminate in the right way.")
        return

    if save_npy:
        import pickle as pkl
        np.save(vid_base + rgb_vid + ".npy", rgbs)
        np.save(vid_base + disp_vid + ".npy", disps)

    if eval_metrics:
        metrics = evaluate_metric(rgbs, gt_imgs, disps, gt_masks, valid_idxs, poses=poses,
                                  kps=kps, hwf=hwf, ext_scale=args.ext_scale, rgb_vid=rgb_vid,
                                  disp_vid=disp_vid, vid_base=vid_base, eval_postfix=eval_postfix,
                                  centers=centers, eval_both=eval_both, white_bkgd=False,
                                  render_factor = args.render_factor)
    else:
        metrics = {"psnr": None, "ssim": None}
    disps = disps / disps.max()

    return metrics, rgbs, disps



def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

    # training options
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')

    parser.add_argument("--N_rand", type=int, default=32*32*4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in args.decay_unit (default=1000) steps)')
    parser.add_argument("--lrate_decay_rate", type=float, default=0.1,
                        help='learning rate decay')
    parser.add_argument("--decay_unit", type=int, default=1000,
                        help='number of steps until the next decay happen')
    parser.add_argument("--weight_decay", type=float, default=None,
                        help='weight decay to apply on the nerf model')
    parser.add_argument("--single_net", action='store_true',
                        help='use a single network for coarse and fine network')
    parser.add_argument("--coarse_weight", type=float, default=1.0,
                        help='loss weight for coarse samples.')

    parser.add_argument("--use_temp_loss", action='store_true',
                        help='use temporal smoothness loss')
    parser.add_argument("--temp_coef", type=float, default=0.05,
                        help='coefficient on temporal smoothness loss')

    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--n_iters", type=int, default=200000,
                        help="number of training iter")
    parser.add_argument("--loss_fn", type=str, default='MSE',
                        help='Type of loss to use')
    parser.add_argument("--loss_beta", type=float, default=0.1,
                        help='beta for smoothed l1 loss')
    parser.add_argument("--reg_fn", type=str, default=None,
                        help='to regularize occupancy')
    parser.add_argument("--reg_coef", type=float, default=0.1,
                        help='coefficient for regularization loss')
    parser.add_argument("--init_poseopt", type=str, default=None,
                        help='initial poseopt layer from a specific checkpoint')
    parser.add_argument("--no_poseopt_reload", action='store_true',
                        help='ignore the poseopt ckpt')
    parser.add_argument("--finetune", action='store_true',
                        help='flag to indicate fine tune stage, which will not load optimizer and global step from ckpt')
    parser.add_argument("--fix_layer", type=int, default=0,
                        help='flag to fix the pts_linears layer weights before the specified layer')
    parser.add_argument("--use_yuv", action='store_true',
                        help='loss in yuv space instead of rgb')

    # rendering options
    parser.add_argument("--density_scale", type=float, default=1.0,
                       help='to scale the density')
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--P_nms", type=float, default=0.0,
                        help='percentage (in [0, 1] * 100%) of samples from out-of-mask area.')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (kp encoding)')
    parser.add_argument("--multires_pts", type=int, default=5,
                        help='log2 of max freq for positional encoding (mapped location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--multires_bones", type=int, default=0,
                        help='log2 of max freq for positional encoding on joint rotations (24 x 3D)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--ray_noise_std", type=float, default=0.,
                        help='std dev of noises to add stochasticity for pose optimization')

    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--save_image", action='store_true',
                        help='save rendering outcomes as images instead of videos.')

    # training options
    parser.add_argument("--nerf_type", type=str, default="nerf",
                        help='type of NeRF model to use')
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')
    parser.add_argument("--density_type", type=str, default='relu',
                        help='function for transforming raw density values to values >= 0')
    parser.add_argument("--softplus_shift", type=float, default=1.0,
                        help='shift the softplus activation input by subtracting this amount')

    # TODO: get this from dataloader?
    parser.add_argument("--n_subjects", type=int, default=2,
                        help='numbers of subjects to learn')

    # per-frame code optimization options
    parser.add_argument("--opt_framecode", action='store_true',
                        help='jointly optimize per-frame codes')
    parser.add_argument("--n_framecodes", type=int, default=None,
                        help="overwrite the number of framecode for the NeRF model")
    parser.add_argument("--framecode_size", type=int, default=16,
                        help='size of per-view frame code')

    # pose optimization options
    parser.add_argument("--opt_rot6d",action='store_true',
                        help='use continuous rotation matrix')
    parser.add_argument("--opt_posecode", action='store_true',
                        help='jointly optimize per-pose code')
    parser.add_argument("--opt_pose", action='store_true',
                        help='jointly optimize pose')
    parser.add_argument("--opt_pose_stop", type=int, default=None,
                        help='stop updating after this many steps')
    parser.add_argument("--opt_pose_coef", type=float, default=0.,
                        help='regularize so that the optimized pose wouldn\'t be too far from the original')
    parser.add_argument("--opt_pose_tol", type=float, default=0.,
                        help='tolerance of pose adjustment')
    parser.add_argument("--opt_pose_type", type=str, default="B",
                        help="type of objective to use for pose optimization")
    parser.add_argument("--opt_pose_step", type=int, default=1,
                        help='frequency of updating parameters')
    parser.add_argument("--opt_pose_lrate", type=float, default=5e-4,
                        help='learning rate for pose update')
    parser.add_argument("--opt_pose_lrate_decay", type=int, default=250,
                        help='exponential step size decay for pose optimizer (in opt_pose_decay steps)')
    parser.add_argument("--opt_pose_decay_rate", type=float, default=1.0,
                        help='exponential decay rate')
    parser.add_argument("--opt_pose_warmup", type=int, default=0,
                        help='wait this amount of iteration before we optimizing keypoints')
    parser.add_argument("--opt_pose_decay_unit", type=int, default=400,
                        help='unit of decay for opt pose')
    parser.add_argument("--opt_pose_cache", action='store_true',
                        help='use cache for slight speed up lol')
    parser.add_argument("--opt_pose_joint", action='store_true',
                        help='jointly learn NeRF and pose when pose_turn == True')
    parser.add_argument("--testopt", action='store_true',
                        help='doing test time optimization only, no NeRF update')

    # additional network
    parser.add_argument("--use_bgnet", action='store_true',
                        help='learn a background network')
    parser.add_argument("--use_uncertainty", action='store_true',
                        help='use uncertainty for density')
    parser.add_argument("--bgnet_stop", type=int, default=500000,
                        help='stop training bgnet after this amount of iterations')
    parser.add_argument("--bgnet_reg", type=float, default=0.01,
                        help='penalize bgnet for deviating from bg values')
    parser.add_argument("--use_bgfill", action='store_true',
                        help='fill in mean bg image at foreground, while keeping othe the same')

    parser.add_argument("--lbsnet_type", type=str, default="default",
                        help='type of LBSNET')
    parser.add_argument("--use_lbsnet", action='store_true',
                        help='use LBS blending')
    parser.add_argument("--n_lbs", type=int, default=1,
                        help='number of blended coordinates')
    parser.add_argument("--multires_lbs", type=int, default=10,
                        help='number of frequencies')
    parser.add_argument("--multires_lbsviews", type=int, default=4,
                        help='number of frequencies')

    parser.add_argument("--use_ckpt_anchor", action='store_true',
                        help='set ckpt as anchor')

    # dataset options
    parser.add_argument("--num_workers", type=int, default=16,
                        help='number of workers for dataloader (works only when --image_batching=True)')
    parser.add_argument("--dataset_type", type=str, default=['h36m'], nargs='+',
                        help='options: h36m / surreal / perfcap / mixamo')
    parser.add_argument("--subject", type=str, default=["S9"], nargs='+',
                        help='subject to train with on mixamo')
    parser.add_argument("--camera", type=int, default=None,
                        help='camera to use')
    parser.add_argument("--use_val", action='store_true',
                        help='use validation set during training')
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--ext_scale", type=float, default=0.001,
                        help="scaling the extrinsic matrix (world coordinate)")
    parser.add_argument("--use_background", action='store_true',
                        help="use static background during reconstructions.")
    parser.add_argument("--fg_ratio", type=float, default=None,
                        help="sample at least this amount of rays from the mask")
    parser.add_argument("--kp_dist_type", type=str, default="reldist",
                        help="specify the way we calculate kp input")
    parser.add_argument("--view_type", type=str, default="relray",
                        help="type of view as input")
    parser.add_argument("--bone_type", type=str, default="reldir",
                        help="type of bone as input")
    parser.add_argument("--pts_tr_type", type=str, default="local",
                        help='type of transformation to apply on 3D world points')

    # surreal dataset option
    parser.add_argument("--train_skip", type=int, default=1,
                        help="skip animated frames in surreal dataset")
    parser.add_argument("--view_skip", type=int, default=1,
                        help="skip camera view in surreal dataset")
    parser.add_argument("--N_cams", type=int, default=None,
                        help='number of cams to use for surreal dataset')
    parser.add_argument("--use_cutoff", action='store_true',
                        help='use cutoff embedder')
    parser.add_argument("--normalize_cutoff", action='store_true',
                        help='normalize cutoff so that we have a complete wave inside the cutoff range')
    parser.add_argument("--cutoff_mm", type=float, default=500,
                        help='distance to apply cutoff (default: 0.5m = 500mm)')
    parser.add_argument("--cutoff_inputs", action='store_true',
                        help='apply cutoff to the input as well')
    parser.add_argument("--cut_to_dist", action='store_true',
                        help='cut input distance to cutoff_mm')
    parser.add_argument("--cutoff_shift", action='store_true',
                        help='shift the input distance by (dist * 2 - 1), so that distance span a complete period')
    parser.add_argument("--cutoff_viewdir", action='store_true',
                        help='apply cutoff to view direction input using the initial inputs!')
    parser.add_argument("--opt_cutoff", action='store_true',
                        help='optimize cutoff threshold')
    parser.add_argument("--cutoff_step", type=int, default=250,
                        help='steps (in 1000) to increase the temperature parameter of cutoff by cutoff_rate')
    parser.add_argument("--cutoff_rate", type=float, default=10.,
                        help='exponential cutoff temperature increase')

    parser.add_argument("--cutoff_bones", action='store_true',
                        help='apply cutoff to bone rotations')
    parser.add_argument("--cutoff_ancestors", type=int, default=5,
                        help='numbers of ancestors to keep along the kinematic chain when doing bone cutoff')

    parser.add_argument("--freq_schedule", action='store_true',
                         help='schedule frequencies as in BARF')
    parser.add_argument("--freq_schedule_step", type=int, default=5,
                         help='target step to enable all frequencies (in 1000 steps)')
    parser.add_argument("--init_freq", type=float, default=0.,
                        help='initial frequencies that are enabled')

    # h36m dataset
    # TODO: enable this
    parser.add_argument("--multiview", action='store_true',
                        help='use multiview optimization')
    # TODO: REMOVE
    parser.add_argument("--training_res", type=float, default=1.0,
                        help='resize training images by this amount')

    parser.add_argument("--val_seq", nargs="+", type=int, default=[6, 18],
                        help='list of sequence number for validation')
    #parser.add_argument("--rand_train_kps", type=int, default=None,
    #                    help='randomly select a number of kps for training (note: for fixed random seq for 180 and 420)')
    parser.add_argument("--rand_train_kps", type=str, default=None,
                        help='randomly select a number of kps for training (note: for fixed random seq for 180 and 420)')


    parser.add_argument("--N_sample_images", type=int, default=8,
                        help='number of images to sample rays from')
    parser.add_argument("--image_batching", action='store_true',
                        help='sample rays from N_sample images')
    parser.add_argument("--mask_image", action='store_true',
                        help='mask out pixels that are not in the foreground when providing image target')
    parser.add_argument("--patch_size", type=int, default=1,
                        help='sample patches of rays from the image')
    parser.add_argument("--load_refined", action='store_true',
                        help='load refined poses for training')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_pose_weights", type=int, default=2000,
                        help='frequency of saving pose weights')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=10000,
                        help='frequency of render_poses video saving')

    # debug
    parser.add_argument("--debug", action='store_true',
                        help='debug flag')

    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()

    train_loader, render_data, data_attrs = load_data(args)
    skel_type = data_attrs['skel_type']
    hwf = data_attrs["hwf"]
    H, W, focal = hwf
    print("Loader initialized.")

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, loaded_ckpt = create_raycaster(args, data_attrs, device=device)
    global_step = start

    popt_kwargs, pose_optimizer = None, None
    if args.opt_pose:
        gt_kps = data_attrs['gt_kp3d']
        pose_optimizer, popt_kwargs = create_popt(args, data_attrs, ckpt=loaded_ckpt, device=device)
    print('done creating popt')

    N_iters = args.n_iters + 1

    # tensorboard summary writer
    writer = SummaryWriter(os.path.join(basedir, expname))

    start = start + 1
    ray_caster = render_kwargs_train['ray_caster']
    trainer = Trainer(args, data_attrs, optimizer, pose_optimizer,
                      render_kwargs_train, render_kwargs_test,
                      popt_kwargs, device=device)

    train_iter = iter(train_loader)
    for i in trange(start, N_iters):
        time0 = time.time()
        batch = next(train_iter)
        loss_dict, stats = trainer.train_batch(batch, i, global_step)

        # Rest is logging
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            trainer.save_nerf(path, global_step)

        save_opt_pose = args.opt_pose and ((args.opt_pose_stop is None) or (args.opt_pose_stop > i))
        if (i % args.i_pose_weights == 0) and save_opt_pose:
            path = os.path.join(basedir, expname, 'pose_weights_{:06d}.tar'.format(i))
            trainer.save_popt(path, i)


        # TODO: deal with this
        if i % args.i_testset==0 and i > 0:
            if args.opt_pose and render_data["kp_idxs"] is not None:
                popt_layer = popt_kwargs['popt_layer']
                with torch.no_grad():
                    kp_val, bone_val, skt_val, _, _ = popt_layer(render_data["kp_idxs"])
            else:
                kp_val = torch.tensor(render_data["kp3d"]).to(device)
                skt_val = torch.tensor(render_data["skts"]).to(device)
                bone_val = torch.tensor(render_data["bones"]).to(device)

            pose_val = torch.tensor(render_data["c2ws"]).to(device)

            gt_imgs = render_data["imgs"]
            gt_masks = render_data["fgs"]
            bg_imgs = render_data["bgs"]
            bg_indices = render_data.get("bg_idxs", None)
            centers = render_data['center']
            cams_val = torch.tensor(render_data["cam_idxs"]) if args.opt_framecode else None
            subject_val = render_data.get("subject_idxs", None)
            subject_val = torch.tensor(subject_val) if subject_val is not None else None

            moviebase = os.path.join(basedir, expname, '{}_val_{:06d}_'.format(expname, i))
            # Intentionally not feeding cyl here to speed ikt up
            if bg_indices is not None:
                masked_gts = gt_imgs * gt_masks + (1 - gt_masks) * bg_imgs[bg_indices]
            else:
                masked_gts = gt_imgs * gt_masks + (1 - gt_masks) * bg_imgs


            metrics, rgbs, disps = render_testset(pose_val, render_data["hwf"], args, render_kwargs_test, cams=cams_val,
                                                  kps=kp_val, skts=skt_val, bones=bone_val, subject_idxs=subject_val,
                                                  gt_imgs=masked_gts, gt_masks=gt_masks, vid_base=moviebase, centers=centers,
                                                  bg_imgs=bg_imgs, bg_indices=bg_indices, eval_metrics=True, eval_both=True)

            fps  = 5
            writer.add_video("Val/ValRGB", torch.tensor(rgbs).permute(0, 3, 1, 2)[None], i, fps=fps)
            writer.add_video("Val/ValDIPS", torch.tensor(disps).permute(0, 3, 1, 2)[None], i, fps=fps)
            if args.opt_pose:
                RH, RW, Rfocals = render_data["hwf"]
                # set the resolution right!
                if args.render_factor != 0:
                    RH, RW = RH // args.render_factor, RW // args.render_factor
                    Rfocals = Rfocals / args.render_factor
                skel_imgs = draw_skeletons_3d((rgbs * 255).astype(np.uint8), kp_val.cpu().numpy(), pose_val.cpu().numpy(),
                                              RH, RW, Rfocals)
                writer.add_video("Val/Skeleton", torch.tensor(skel_imgs).permute(0, 3, 1, 2)[None], i, fps=fps)
            writer.add_scalar("Val/PSNR", metrics["psnr"], i)
            writer.add_scalar("Val/SSIM", metrics["ssim"], i)

        if i % args.i_print == 0:
            mem = torch.cuda.max_memory_allocated() / 1024. / 1024.
            loss, psnr, alpha, total_norm = loss_dict['total_loss'], stats['psnr'], stats['alpha'], stats['total_norm']
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr}, Alpha: {alpha}, GradNorm {total_norm}, Mem: {mem}")
            for k in loss_dict:
                if k == 'total_loss':
                    writer.add_scalar(f'Loss/{k}_{args.loss_fn}', loss_dict[k], i)
                else:
                    writer.add_scalar(f'Loss/{k}', loss_dict[k], i)

            for k in stats:
                writer.add_scalar(f'Stats/{k}', stats[k], i)


        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.multiprocessing.set_start_method('spawn')

    train()
