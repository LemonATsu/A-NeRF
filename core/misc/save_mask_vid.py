import numpy as np
import argparse
import os
import cv2
import deepdish as dd
import imageio

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Arguments for masks extraction')
    parser.add_argument("-b", "--base_path", type=str,
                        #default="data/h36m/h36m_full",
                        default="data/h36m/",
                        help='base directory')
    parser.add_argument("-c", "--camera_id", type=int, default=None,
                        help='camera to extract')
    parser.add_argument("-s", "--subject", type=str, default="S9",
                        help='subject to extract')
    parser.add_argument("-r", "--res", type=float, default=1.0,
                        help='resize mask')
    args = parser.parse_args()
    base_path = args.base_path #"data/h36m/h36m_full"
    subject = args.subject # "S9"
    cameras = ["54138969", "55011271", "58860488", "60457274"]
    camera_id = args.camera_id

    camera = None
    if camera_id is not None:
        camera = cameras[camera_id]
        if subject != 'S1':
            h5_name = os.path.join(base_path, f"{subject}-camera=[{camera}]-subsample=5.h5")
        else:
            h5_name = os.path.join(base_path, f"{subject}-camera=[{camera}]-subsample=1.h5")
    else:
        h5_name = os.path.join(base_path, f"{subject}_processed.h5")

    print(f"read from {h5_name}")
    _img_paths = dd.io.load(h5_name, "/img_path")
    img_paths = [os.path.join(base_path, img_path) for img_path in _img_paths]
    mask_paths = [img_path.replace(f"{subject}", f"{subject}m_") for img_path in img_paths]

    #masked_imgs = []
    masks = []
    for i, (img_path, mask_path) in enumerate(zip(img_paths, mask_paths)):
        mask = imageio.imread(mask_path)
        mask[mask < 128] = 0
        if mask.shape[0] == 1002:
            #img = img[1:-1]
            mask = mask[1:-1]
        if args.res != 1.0:
            H, W = mask.shape[:2]
            new_W, new_H = int(W * args.res), int(H * args.res)
            mask = cv2.resize(mask, (new_W, new_H), interpolation=cv2.INTER_NEAREST)
        masks.append(mask)

    if camera is None:
        h5_name = os.path.join(base_path, f"{subject}_mask_deeplab_crop.h5")
    else:
        h5_name = os.path.join(base_path, f"{subject}_{camera}_mask_deeplab_crop.h5")

    save_dict = {"masks": np.array(masks), "index": _img_paths}
    if args.res != 1.0:
        save_dict['res'] = args.res
    print(f"saving mask to {h5_name}")
    dd.io.save(h5_name, save_dict)

    #imageio.mimwrite(f"vid_{subject}_cam_{camera_id}.mp4", masked_imgs, fps=25)
    #imageio.mimwrite(f"vid_{subject}_small.mp4", masked_imgs, fps=25)
