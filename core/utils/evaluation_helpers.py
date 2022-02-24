import glob
import torch
import imageio
import os, cv2
import numpy as np
from smplx import SMPL
from smplx.lbs import vertices2joints
import torch.nn.functional as F
from pytorch_msssim import SSIM
from PIL import Image, ImageFont, ImageDraw
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from .ray_utils import kp_to_valid_rays
from .run_nerf_helpers import to8b
from .skeleton_utils import axisang_to_rot, nerf_bones_to_smpl, world_to_cam


DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
}



# READERS
def read_tfevent(path, guidance=DEFAULT_SIZE_GUIDANCE):
    event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
    event_acc.Reload()
    return event_acc

def read_tag_scalars(tags, events):

    if not isinstance(events, list):
        events = [events]
    if not isinstance(tags, list):
        tags = [tags]

    return_dict = {}
    for tag in tags:
        return_dict[tag] = []
        return_dict[tag + "_steps"] = []
    return_dict["num_events"] = len(events)

    for event in events:
        for tag in tags:
            data_list = event.Scalars(tag)
            values = list(map(lambda x: x.value, data_list))
            steps = list(map(lambda x: x.step, data_list))
            return_dict[tag].append(values)
            return_dict[tag + "_steps"].append(steps)

    return return_dict

def read_events_from_paths(log_paths):
    events = []
    for log_path in log_paths:
        event_paths = glob.glob(os.path.join(log_path, "events.*"))
        event = None
        for event_path in event_paths:
            e = read_tfevent(event_path)
            if event is None:
                event = e
            # TODO: handle cases that have multiple events
        events.append(event)
    return events

def read_eval_result(log_path, dir_name="val_*_val", step=10000, prefix="Val"):
    """
    dir_name: directory name in the log path
    step: interval between the logged numbers
    """
    num_events = 0
    nonempty_paths = []
    scalar_dict = {f"{prefix}/PSNR": [], f"{prefix}/PSNR_steps": [],
                   f"{prefix}/SSIM": [], f"{prefix}/SSIM_steps": []}
    psnr_path = glob.glob(os.path.join(log_path, dir_name, "psnr.txt"))
    ssim_path = glob.glob(os.path.join(log_path, dir_name, "ssim.txt"))

    if len(psnr_path) < 1:
        return None

    psnr_path = psnr_path[0]
    ssim_path = ssim_path[0]

    with open(psnr_path, "r") as f:
        psnrs = []
        steps = []
        for i, line in enumerate(f.readlines()):
            psnrs.append(float(line))
            steps.append(step * (i + 1))
        scalar_dict[f"{prefix}/PSNR"].append(psnrs)
        scalar_dict[f"{prefix}/PSNR_steps"].append(steps)

    with open(ssim_path, "r") as f:
        ssims = []
        steps = []
        for i, line in enumerate(f.readlines()):
            ssims.append(float(line))
            steps.append(step * (i + 1))
        scalar_dict[f"{prefix}/SSIM"].append(ssims)
        scalar_dict[f"{prefix}/SSIM_steps"].append(steps)

    num_events += 1
    scalar_dict["num_events"] = num_events
    return scalar_dict

def get_best_values_n_steps(scalar_dict, tag, maximum=True):

    reduce_fn = np.argmax if maximum else np.argmin

    n_return = len(scalar_dict[tag])
    values = np.array(scalar_dict[tag])
    best_idx = reduce_fn(values, axis=-1)
    best_values = values[np.arange(len(values)), best_idx]
    best_steps = np.array(scalar_dict[tag+"_steps"])[np.arange(len(values)), best_idx]

    return best_values, best_steps

def retrieve_best_vid_files(log_paths, best_steps, keyword_str="_%06d", postfix="rgb.mp4"):
    vid_names = []

    for log_path, best_step in zip(log_paths, best_steps):
        search_path = os.path.join(log_path, (f"*{keyword_str}*{postfix}") % best_step)
        fn = glob.glob(search_path)
        if len(fn) > 1:
            for f in fn:
                if "text_" in f and ".mp4" in f:
                    os.remove(f)
            fn = [f for f in fn if "text_" not in f]
        try:
            assert len(fn) == 1, "Bad keyword string, multiple files are found!"
        except:
            import pdb; pdb.set_trace()
            print(0)
        vid_names.append(fn[0])
    return vid_names

def concat_vid(vid_names, output_name, nrows=1, ncols=None, texts=None,
               base_cmd="ffmpeg -y"):
    if texts is not None:
        if len(texts) != len(vid_names):
            import pdb; pdb.set_trace()
            print()
        assert len(texts) == len(vid_names), "Text lists should be as the same length as vid_names!"
        tmp_vid_names = []
        for vid_name, text in zip(vid_names, texts):
            tmp_vid_name = add_text_to_vid(vid_name, text)
            tmp_vid_names.append(tmp_vid_name)
        vid_names = tmp_vid_names
    if ncols is None:
        ncols = len(vid_names) // nrows
    vid_names = np.array(vid_names).reshape(nrows, ncols)

    # concat horizontally
    for j, row in enumerate(vid_names):
        cmd = base_cmd
        for name in row:
            cmd += f" -i {name}"
        if nrows == 1:
            # back out here if we only have one row
            cmd += f" -filter_complex hstack={len(row)} {output_name}"
            os.system(cmd)

            # clean up temporary text file
            if texts is not None:
                for vid_name in vid_names.reshape(-1):
                    os.remove(vid_name)
            return

        cmd += f" -filter_complex hstack={len(row)} {j}__tmp.mp4"
        os.system(cmd)

    # concat vertically
    cmd = base_cmd
    for j in range(len(vid_names)):
        cmd += f" -i {j}__tmp.mp4"
    cmd += f" -filter_complex vstack={len(vid_names)} {output_name}"
    os.system(cmd)

    # Clean up temporary videos
    for i in range(len(vid_names)):
        if os.path.exists(f"{i}__tmp.mp4"):
            os.remove(f"{i}__tmp.mp4")

    if texts is not None:
        for vid_name in vid_names.reshape(-1):
            if os.path.exists(vid_name):
                os.remove(vid_name)

def add_text_to_vid(vid_name, text,
                    font_type="DejaVuSans-Bold", font_size=16,
                    text_loc=(10, 30)):

    font = ImageFont.truetype(font_type, font_size)


    # setup name
    pd = os.path.dirname(vid_name)
    new_name = os.path.join(pd, "text_" + vid_name.split("/")[-1])

    # setup video read/write
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid = cv2.VideoWriter(new_name, fourcc, 14, (512, 512))
    reader = cv2.VideoCapture(vid_name)
    while reader.isOpened():
        ret, frame = reader.read()
        if ret:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            draw.text(text_loc, text, font=font)
            text_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            vid.write(text_frame)
        else:
            break
    vid.release()
    reader.release()
    return new_name

def txt_to_argstring(path, ignore_config=False):
    import ast

    argstr = []
    with open(path, 'r') as f:
        for line in f.readlines():
            arg_val = line.strip().split(' = ')
            if len(arg_val) < 2:
                continue
            arg, val = arg_val
            try:
                literal = ast.literal_eval(val)
            except:
                literal = val

            if literal is None:
                continue
            if arg == 'config' and ignore_config:
                continue

            argstr.append(f'--{arg}')
            if isinstance(literal, list):
                val = [f'{v}' for v in literal]
                argstr.extend(val)
            elif isinstance(literal, str) and literal[0] == '[' and literal[-1] == ']':
                # might be non-string?
                list_str = literal[1:-1].split(',')
                list_str = [s.strip() for s in list_str]
                argstr.extend(list_str)
            elif not isinstance(literal, bool):
                argstr.append(val)
            elif not literal:
                argstr.pop(-1)

    return argstr

def evaluate_metric(rgbs, gt_imgs, disps=None, gt_masks=None, valid_idxs=None, poses=None,
                    kps=None, hwf=None, centers=None, ext_scale=None, rgb_vid="rgb.mp4", disp_vid="disp.mp4",
                    vid_base=None, eval_postfix="", eval_both=False, white_bkgd=False,
                    render_factor=0):

    if eval_both and (valid_idxs is None or render_factor != 0):
        RH, RW, Rfocal = hwf
        print("Valid idxs not provided or is calculated at different resolution. Calculate them from keypoints ...")
        _, valid_idxs, _, _ = kp_to_valid_rays(poses, RH, RW, Rfocal,
                                               centers=centers, kps=kps, ext_scale=ext_scale)


    # trim out nan to get visible outcomes

    if vid_base is not None:
        os.makedirs(os.path.dirname(vid_base), exist_ok=True)
        imageio.mimwrite(vid_base + rgb_vid, to8b(rgbs), fps=14, quality=8)
        if disps is not None:
            disps_nan = np.isnan(disps)
            disps[disps_nan] = 0.
            imageio.mimwrite(vid_base + disp_vid, to8b(disps / np.max(disps)), fps=14, quality=8)

    test_psnr = None
    test_ssim = None

    #TODO: all the ops below are arry ops, can't generalize to different resolution
    H, W, _ = hwf
    H = H if isinstance(H, int) else H[0]
    W = W if isinstance(W, int) else W[0]

    if eval_both:
        valid_masks = np.zeros((len(valid_idxs), H * W, 1), dtype=np.float32)
        for i in range(len(valid_idxs)):
            valid_idx = valid_idxs[i].cpu()
            valid_masks[i, valid_idx] = 1
        valid_masks = np.array(valid_masks).reshape(-1, H, W, 1)
    else:
        valid_masks = None

    if gt_masks is not None:
        # remove images without any person in it
        valid_imgs = np.where(gt_masks.reshape(gt_masks.shape[0], -1).sum(-1) > 0)[0]

        rgbs = rgbs[valid_imgs]
        gt_imgs = gt_imgs[valid_imgs]

        gt_masks = gt_masks[valid_imgs]
        valid_masks = valid_masks[valid_imgs] if valid_masks is not None else None
        poses = poses[valid_imgs]

    ssim_eval = SSIM(size_average=False)
    th_rgbs = torch.tensor(rgbs).permute(0, 3, 1, 2)#.cpu()
    if render_factor > 0:
        th_rgbs = F.interpolate(th_rgbs, size=gt_imgs.shape[1:3], mode='bilinear',
                                align_corners=False).cpu()
        rgbs = th_rgbs.permute(0, 2, 3, 1).numpy()
    else:
        th_rgbs = th_rgbs.cpu()
    th_gt = torch.tensor(gt_imgs).permute(0, 3, 1, 2).cpu()
    try:
        th_ssim = ssim_eval(th_rgbs, th_gt)
    except:
        import pdb; pdb.set_trace()
        print()
    test_ssim = th_ssim.permute(0, 2, 3, 1).cpu().numpy()
    sqr_diff = np.square(gt_imgs - rgbs)

    if gt_masks is not None:
        denom = np.maximum(gt_masks.reshape(len(poses), -1).sum(-1) * 3., 1.)# avoid dividing by zero

        fg_psnr = -10. * np.log10((sqr_diff * gt_masks[..., :1]).reshape(len(poses), -1).sum(-1) / denom)
        fg_psnr[fg_psnr == np.inf] = 0.
        fg_psnr = fg_psnr.mean()

        fg_ssim = (test_ssim * gt_masks[..., :1]).reshape(len(poses), -1).sum(-1) / denom
        fg_ssim[fg_ssim == np.inf] = 0.
        fg_ssim = fg_ssim.mean()

    if valid_masks is not None:
        denom = np.maximum(valid_masks.reshape(len(poses), -1).sum(-1) * 3., 1.)# avoid dividing by zero

        valid_psnr = -10. * np.log10((sqr_diff * valid_masks[..., :1]).reshape(len(poses), -1).sum(-1) / denom)
        valid_psnr[valid_psnr == np.inf] = 0.
        valid_psnr = valid_psnr.mean()

        valid_ssim = (test_ssim * valid_masks[..., :1]).reshape(len(poses), -1).sum(-1) / denom
        valid_ssim[valid_ssim == np.inf] = 0.
        valid_ssim = valid_ssim.mean()

    if valid_masks is None and gt_masks is None:
        # average the values over all pixels
        test_psnr  = -10. * np.log10(np.mean(sqr_diff.reshape(len(poses), -1), axis=-1))
        test_psnr[test_psnr == np.inf] = 0.
        test_psnr = test_psnr.mean()

        test_ssim[test_ssim == np.inf] = 0.
        test_ssim = test_ssim.mean()

        print(f"Evaluate PSNR: {test_psnr}, SSIM: {test_ssim}")
        with open(vid_base + f"psnr{eval_postfix}.txt", "a") as f:
            f.write(f"{test_psnr}\n")
        with open(vid_base + f"ssim{eval_postfix}.txt", "a") as f:
            f.write(f"{test_ssim}\n")

    elif not eval_both:
        # only foreground is evaluated
        test_psnr = fg_psnr
        test_ssim = fg_ssim

        print(f"Evaluate PSNR: {test_psnr}, SSIM: {test_ssim}")
        with open(vid_base + f"psnr{eval_postfix}.txt", "a") as f:
            f.write(f"{test_psnr}\n")
        with open(vid_base + f"ssim{eval_postfix}.txt", "a") as f:
            f.write(f"{test_ssim}\n")
    else:
        test_psnr = valid_psnr
        test_ssim = valid_ssim

        print(f"Evaluate PSNR: {test_psnr} ({fg_psnr}), SSIM: {test_ssim} ({fg_ssim})")
        with open(vid_base + f"psnr{eval_postfix}.txt", "a") as f:
            f.write(f"{test_psnr}\n")
        with open(vid_base + f"ssim{eval_postfix}.txt", "a") as f:
            f.write(f"{test_ssim}\n")
        with open(vid_base + f"psnr{eval_postfix}_fg.txt", "a") as f:
            f.write(f"{fg_psnr}\n")
        with open(vid_base + f"ssim{eval_postfix}_fg.txt", "a") as f:
            f.write(f"{fg_ssim}\n")

    return {"psnr": test_psnr, "ssim": test_ssim, "psnr_fg": fg_psnr, "ssim_fg": fg_ssim}

def procrustes(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.
    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.
        d, Z, [tform] = procrustes(X, Y)
    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.
    scaling
        if False, the scaling component of the transformation is forced
        to 1
    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.
    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()
    Z
        the matrix of transformed Y-values
    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y
    """
    n,m = X.shape
    ny,my = Y.shape
    muX = X.mean(0)
    muY = Y.mean(0)
    X0 = X - muX
    Y0 = Y - muY
    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()
    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)
    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY
    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)
    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)
    if reflection !=  'best':
        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0
        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)
    traceTA = s.sum()
    if scaling:
        # optimum scaling of Y
        b = traceTA * normX / normY
        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2
        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX
    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX
    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)
    #transformation values
    tform = {'rotation':T, 'scale':b, 'translation':c}
    return d, Z, tform

class Criterion_MPJPE(torch.nn.Module):

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_batch, label_batch):
        diff = torch.norm(pred_batch - label_batch, p=2, dim=-1)
        if self.reduction == 'mean':
            metric = diff.mean()
        elif self.reduction == 'sum':
            metric = diff.sum()
        else:
            metric = diff
        return metric

class Criterion3DPose_ProcrustesCorrected(torch.nn.Module):
    """
    Normalize translaion, scale and rotation in the least squares sense, then apply the specified criterion
    """
    def __init__(self, criterion):
        super(Criterion3DPose_ProcrustesCorrected, self).__init__()
        self.criterion = criterion
    def forward(self, pred_batch, label_batch):
        #Optimal scale transform
        preds_procrustes = []
        batch_size = pred_batch.size()[0]
        num_joints = pred_batch.size()[-2]
        num_dim = pred_batch.size()[-1]
        assert num_dim == 3
        for i in range(batch_size):
            d, Z, tform = procrustes(label_batch[i].data.cpu().numpy().reshape(num_joints, num_dim),
                                     pred_batch[i].data.cpu().numpy().reshape(num_joints, num_dim))
            preds_procrustes.append(Z.reshape((num_joints, num_dim)))
        pred_batch_aligned = torch.tensor(np.stack(preds_procrustes)).to(pred_batch.device)
        return self.criterion(pred_batch_aligned, label_batch), pred_batch_aligned

class Criterion3DPose_leastQuaresScaled(torch.nn.Module):
    """
    Normalize the scale in the least squares sense, then apply the specified criterion
    """
    def __init__(self, criterion):
        super(Criterion3DPose_leastQuaresScaled, self).__init__()
        self.criterion = criterion
    def forward(self, pred, label):
        #Optimal scale transform
        batch_size = pred.size()[0]
        pred_vec = pred.view(batch_size,-1)
        gt_vec = label.view(batch_size,-1)
        dot_pose_pose = torch.sum(torch.mul(pred_vec,pred_vec),1,keepdim=True)
        dot_pose_gt   = torch.sum(torch.mul(pred_vec,gt_vec),1,keepdim=True)
        s_opt = dot_pose_gt / dot_pose_pose
        s_opt = s_opt[..., None]
        return self.criterion.forward(s_opt*pred, label), s_opt*pred


class SMPLEvalHelper(SMPL):
    # steal from SPIN
    def __init__(self, *args, **kwargs):
        super(SMPLEvalHelper, self).__init__(*args, **kwargs)
        J_regressor_extra = np.load("smpl/data/J_regressor_h36m.npy")
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))

    def forward(self, *args, **kwargs):
        smpl_output = super(SMPLEvalHelper, self).forward(*args, **kwargs)
        h36m_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
        return smpl_output, h36m_joints


SPIN_TO_CANON = [10, 8, 14, 15, 16, 11, 12, 13, 4, 5, 6, 1, 2, 3, 0, 7, 9]
H36M_TO_17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_14 = H36M_TO_17[:14]
@torch.no_grad()
def evaluate_pampjpe_from_smpl_params(gt_kps, betas, bones, bone_orders="xyz",
                                      ret_kp=False, ret_pck=False,
                                      align_kp=False, pck_threshold=150,
                                      reduction="mean",
                                      use_normalize=False):

    assert betas.dim() == 2
    if betas.shape[0] == 1:
        betas = betas.expand(len(gt_kps), -1)

    rots = axisang_to_rot(bones.view(-1, 3)).view(*bones.shape[:2], 3, 3)


    smpl = SMPLEvalHelper("smpl/SMPL_NEUTRAL.pkl").to(rots.device)
    _, pred_kps = smpl(betas=betas,
                       body_pose=rots[:, 1:],
                       global_orient=rots[:, :1],
                       pose2rot=False)
    pred_kps = pred_kps[:, SPIN_TO_CANON] # to the same scale
    # H36M TO 14
    #pred_kps = pred_kps[:, H36M_TO_14] # to the same scale
    #print("TO 14")
    #gt_kps = gt_kps[:, :14]

    mpjpe_crit = Criterion_MPJPE(reduction=reduction).to(rots.device)
    pampjpe_crit = Criterion3DPose_ProcrustesCorrected(mpjpe_crit).to(rots.device)

    if use_normalize:
        mpjpe_crit = Criterion3DPose_leastQuaresScaled(mpjpe_crit)

    pampjpe, aligned_kps = pampjpe_crit(pred_kps,
                           torch.FloatTensor(gt_kps).to(pred_kps.device))

    gt_kps_trans = gt_kps.copy()
    pred_kps_trans = pred_kps.clone()
    gt_kps_trans = gt_kps_trans - gt_kps_trans[:, 14:15]
    pred_kps_trans = pred_kps_trans - pred_kps_trans[:, 14:15]
    #gt_kps_trans = gt_kps_trans - gt_kps_trans[:, :1]
    #pred_kps_trans = pred_kps_trans - pred_kps_trans[:, :1]
    mpjpe = mpjpe_crit(pred_kps_trans,
                       torch.FloatTensor(gt_kps_trans / 1000).to(pred_kps.device))
    if use_normalize:
        mpjpe = mpjpe[0]

    mpjpe  = mpjpe * 1000

    if not ret_kp:
        return pampjpe, mpjpe
    if align_kp:
        return pampjpe, mpjpe, aligned_kps
    if ret_pck:
        # in /1000 scale to avoid numerical issue
        pck_threshold = pck_threshold
        pck = (pampjpe < pck_threshold).float().mean()

        thresholds = torch.linspace(0, 150, 31).tolist()
        auc = []
        for i, t in enumerate(thresholds):
            pck_at_t = (pampjpe < t).float().mean().item()
            auc.append(pck_at_t)

        return pampjpe, mpjpe, pck, np.mean(auc)


    def pck(actual, expected, included_joints=None, threshold=150):
        dists = euclidean_losses(actual, expected)
        if included_joints is not None:
            dists = dists.gather(-1, torch.LongTensor(included_joints))
        return (dists < threshold).double().mean().item()

    return pampjpe, mpjpe, pred_kps

def estimates_to_kp2ds(kps, exts, img_height, img_width, focals,
                       pose_scale=1.0, pelvis_locs=None,
                       pelvis_order="xyz", our_exts=True):
    """
    our_exts: if the extrinsic is in our coordinate system
    """

    assert kps.shape[-2] == 17

    if pelvis_locs is not None:
        if pelvis_order == "xyz":
            kps = kps.copy()
            pelvis_locs = pelvis_locs.copy()
            pelvis_locs[..., 1:] *= -1
        kps[..., 14, :] = pelvis_locs[:, 0]

    kps = kps * pose_scale
    if our_exts:
        kps[..., 1:] *= -1
    kp2ds = np.array(
                [world_to_cam(kp, ext, img_height, img_width, focal)
                 for (kp, ext, focal) in zip(kps, exts, focals)]
            )

    return kp2ds

