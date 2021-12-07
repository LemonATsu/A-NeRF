## Pre-processed data
We provide pre-processed `.h5` files for SURREAL and Mixamo dataset. You can find the processed `.zip` files [here](https://drive.google.com/drive/folders/1hFw2OGaqBgxaithmLP5gOTOMgrgJb7xG?usp=sharing). 

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
We also provide pre-trained characters for SURREAL and Mixamo. You can download the models [here](https://drive.google.com/drive/folders/1g0FvSqx9VpC2olhnnuhAnU3Mtlk0is8M?usp=sharing).

For rendering, you can simply use the config files in `configs/` for `--nerf_args`.

Belows are the brief summaries for the pretrained models:

`surreal.tar`: A model trained on SURREAL full dataset for 150k iterations with A-NeRF.

`{james,archer}_ft.tar`: Models that are trained with pose refinement for 500k (with `--opt_pose_stop 200000`), and then finetune with config file `configs/mixamo/mixamo_finetunes.txt` for 200k.

`{james,archer}_ft_tv.tar`: Same as the above, but use `--use_temp_loss --temp_coef 0.05` during the pose refinement phase.
