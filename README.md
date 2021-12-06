# A-NeRF: Articulated Neural Radiance Fields for Learning Human Shape, Appearance, and Pose
### [Paper](https://arxiv.org/abs/2102.06199) | [Website](https://lemonatsu.github.io/anerf/) | [Data(Coming soon)]()
![](imgs/teaser.gif)
>**A-NeRF: Articulated Neural Radiance Fields for Learning Human Shape, Appearance, and Pose**
>
>[Shih-Yang Su](https://lemonatsu.github.io/), [Frank Yu](https://yu-frank.github.io/), [Michael Zollhöfer](https://zollhoefer.com/), and [Helge Rhodin](http://helge.rhodin.de/)
>
>Thirty-Fifth Conference on Neural Information Processing Systems (NeurIPS 2021)


## Setup
TODO: fill in the exact things here
```
conda create -n anerf python=3.8
conda activate anerf
conda install --file requirements.txt
```

TODO: mention how to download SMPL

TODO: mention how to install pytorch-ssim
## Training
We provide template training configurations in `configs/` for different settings. 

To train A-NeRF on our pre-processed SURREAL dataset,
```
python run_nerf.py --config configs/surreal/surreal.txt --basedir logs  --expname surreal_model
```
The trained weights and log can be found in ```logs/surreal_model```.

To train A-NeRF on our pre-processed Mixamo dataset with estimated poses, run
```
python run_nerf.py --config configs/mixamo/mixamo.txt --basedir log_mixamo/ --num_workers 8 --subject archer --expname mixamo_archer
```
This will train A-NeRF on Mixamo Archer with pose refinement for 500k iterations, with 8 worker threads for the dataloader.

To finetune the learned model, run
```
python run_nerf.py --config configs/mixamo/mixamo_finetune.txt --finetune --ft_path log_mixamo/mixamo_archer/500000.tar --expname mixamo_archer_finetune
```
This will finetune the learned Mixamo Archer for 200k with the already refined poses. Note that the pose will not be updated during this time.


## Testing
You can use [`run_render.py`](run_render.py) to render the learned models under different camera motions, or retarget the character to different poses by
```
python run_render.py --nerf_args logs/surreal_model/args.txt --ckptpath logs/surreal_model/150000.tar \ 
				     --dataset surreal --entry hard --render_type bullet --render_res 512 512\
                     --white_bkgd --runname surreal_bullet
```
Here, 
- `--dataset` specifies the data source for poses, 
- `--entry` specifices the particular subset from the dataset to render, 
- `--render_type` defines the camera motion to use, and
- `--render_res` specifies the height and width of the rendered images.

Therefore, the above command will render 512x512 the learned SURREAL character with bullet-time effect. The output can be found in `render_output/surreal_bullet/`
TODO: add example output

You can also extract mesh for the learned character:
```
python run_render.py --nerf_args logs/surreal_model/args.txt --ckptpath logs/surreal_model/150000.tar \ 
				     --dataset surreal --entry hard --render_type mesh --runname surreal_mesh
```
You can find the extracted `.ply` files in `render_output/surreal_mesh/meshes/`.

To render the mesh as in the paper, run
```
python render_mesh.py --expname surreal_mesh 
```
which will output the rendered images in `render_output/surreal_mesh/mesh_render/`
TODO: add example output

You can change the setting in [`run_render.py`](run_render.py) to create your own rendering configuration.


## Citation
```
@inproceedings{su2021anerf,
    title={A-NeRF: Articulated Neural Radiance Fields for Learning Human Shape, Appearance, and Pose},
    author={Su, Shih-Yang and Yu, Frank and Zollh{\"o}fer, Michael and Rhodin, Helge},
    booktitle = {Advances in Neural Information Processing Systems},
    year={2021}
}
```
## Acknowledgements
- The code is built upon [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch).
- We use [SPIN]() for estimating the initial 3D poses for our Mixamo dataset.
