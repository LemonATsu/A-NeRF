## Pre-processed data
We provide pre-processed `.h5` files for SURREAL and Mixamo dataset. You can find the processed `.zip` files [here](https://drive.google.com/drive/folders/1QXli-2BS3xyxK7CYweT1cP7xAgVjcKM9?usp=sharing). 

The data folder (i.e., `data/`) should be organized as the following:
```                                                                                      
├── data
│   ├── surreal
│   │   ├── surreal_train_h5py.h5
│   │   │
│   │   ├── surreal_val_h5py.h5 
│   │   │
│   │   └── surreal_val_idxs.npy # selected views/poses indices for evaluation 
│   │   
│   ├── mixamo 
│   │   ├── James_processed_h5py.h5
│   │   │
│   │   ├── James_selected.npy # selected indices used in the paper
│   │   │
│   │   ├── Archer_processed_h5py.h5
│   │   │
│   │   └── Archer_selected.npy # selected indices used in the paper
```
Note that Surreal dataset has ground truth pose and camera data, while Mixamo dataset only has [SPIN](https://github.com/nkolot/SPIN) estimated ones.

## Pre-trained weights
We also provide pre-trained characters for SURREAL and Mixamo. You can download the models [here]().

For rendering, you can simply use the config files in `configs/` for `--nerf_args`.