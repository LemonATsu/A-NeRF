import os
import glob
import pickle
import pathlib
import numpy as np
from scipy.io import loadmat

# hard coded

closer_sets = []

def sorted_glob(path):
    return sorted((glob.glob(path)))

def save_pkl(path, d):
    with open(path, "wb") as f:
        pickle.dump(d, f)

dirs = sorted_glob("surreal_mix/*_*/")
print(dirs)

for d in dirs:
    subsets = sorted_glob(os.path.join(d, "*-*"))
    sname = pathlib.PurePath(d).name

    cams = []
    img_paths = []
    for subset in subsets:
        info_dict = loadmat(glob.glob(os.path.join(subset, "*_info.mat"))[0])
        cam = info_dict["c2ws"]
        N_kp = info_dict["joints3D"].shape[0]

        cams.append(cam)
        img_paths.append(np.array(sorted_glob(os.path.join(subset, "imageSequences", "*.png"))).reshape(-1, N_kp))

    # TODO: check this based on info.mat
    if not "render_type" in info_dict:
        render_type = "dome_closer" if sname in closer_sets else "dome"
    else:
        render_type = info_dict["render_type"][0]

    cams = np.array(cams).reshape(-1, 4, 4)
    img_paths = np.concatenate(img_paths, axis=0)

    meta = {"cams": cams,
            "N_cam_per_subdir": cam.shape[0],
            "N_cams": cams.shape[0],
            "N_kp": info_dict["joints3D"].shape[0],
            "focal": info_dict["focal"].item(),
            "int_scale": info_dict["int_scale"][0][0],
            "img_paths": img_paths,
            "joints3D": info_dict["joints3D"],
            "poses": info_dict["poses"],
            "render_type": render_type,
           }

    meta_fn = os.path.join(d, "metadata.pkl")
    print(f"{sname} {render_type} {cams.shape} {meta_fn}")
    save_pkl(meta_fn, meta)


