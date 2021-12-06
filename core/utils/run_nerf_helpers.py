import torch
import numpy as np

# Misc
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
def load_ckpt_from_path(ray_caster, optimizer, ckpt_path,
                        finetune=False):
    ckpt = torch.load(ckpt_path)

    global_step = ckpt["global_step"]
    ray_caster.load_state_dict(ckpt)
    if optimizer is not None and not finetune:
        print("load optimizer from ckpt")
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    return global_step, ray_caster, optimizer, ckpt

def filter_state_dict(current_state_dict, state_dict):

    filtered_state_dict = {}
    for local_key in current_state_dict:
        local_val = current_state_dict[local_key]
        loaded_val = state_dict[local_key]
        if local_val.shape != loaded_val.shape:
            print(f'!!!WARNING!!!: size mismatch for {local_key}: current model is {local_val.shape} '+
                  f'while the size in ckpt is {loaded_val.shape}')
            print(f'!!!WARNING!!!: Automatically omit loading {local_key}. If this is not intented, stop the program now!')
        else:
            filtered_state_dict[local_key] = loaded_val

    return filtered_state_dict

# TODO: naming not accurate
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

def imgs_to_grid(imgs, nrols=4, ncols=3):
    N, H, W, C = imgs.shape
    n_grids = int(np.ceil(N / (nrols * ncols)))
    grids = np.zeros((n_grids, nrols * H, ncols * W, C), dtype=np.uint8)

    for i, img in enumerate(imgs):
        grid_loc = i // (nrols * ncols)
        col = i % ncols
        row = (i // ncols) % nrols
        grids[grid_loc, row*H:(row+1)*H, col*W:(col+1)*W, :] = img
    return grids

