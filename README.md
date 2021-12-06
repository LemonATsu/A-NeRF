# A-NeRF: Articulated Neural Radiance Fields for Learning Human Shape, Appearance, and Pose
### [Paper](https://arxiv.org/abs/2102.06199) | [Website](https://lemonatsu.github.io/anerf/) | [Data(Coming soon)]()
![](imgs/teaser.gif)
>**A-NeRF: Articulated Neural Radiance Fields for Learning Human Shape, Appearance, and Pose**
>[Shih-Yang Su](https://lemonatsu.github.io/), [Frank Yu](https://yu-frank.github.io/), [Michael Zollhöfer](https://zollhoefer.com/), and [Helge Rhodin](http://helge.rhodin.de/)
>Thirty-Fifth Conference on Neural Information Processing Systems (NeurIPS 2021)


## Setup
TODO: fill in the exact things here
```
conda create -n anerf python=3.8
conda activate anerf
conda install --file requirements.txt
```
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
