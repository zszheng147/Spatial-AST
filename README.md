# Spatial-AST

This repo hosts the code and models of "[BAT: Learning to Reason about Spatial Sounds with Large Language Models](https://arxiv.org/abs/2402.01591)" [Accepted by ICML 2024 [bib](https://github.com/zszheng147/Spatial-AST#citation)].


<p align="center">
  <img align="middle" height="450" src="assets/Spatial-AST.png"/>
</p>

## Installation
```
conda env create -f environment.yml
bash timm_patch/patch.sh
```

## Data Preparation
### AudioSet (Anechoic Audio Source)
We provide `Balanced train` and `Evaluation` set for your convenience. You can download from [SpatialSound](https://huggingface.co/datasets/zhisheng01/SpatialSounds/tree/main/AudioSet). 
For the `Unbalanced train` set, please refer to [Official AudioSet](https://research.google.com/audioset/download.html).

Metadata can be downloaded from [metadata](https://huggingface.co/datasets/zhisheng01/SpatialSounds/tree/main/AudioSet/metadata).
```
AudioSet
├── balanced_train
│   └── audio
│   │   ├── Y00M9FhCet6s.wav
│   │   ├── Y00mE-lhe_R8.wav
│   │   ├── ...
├── eval
│   └── audio
│   │   ├── Y007P6bFgRCU.wav
│   │   ├── Y00AGIhlv-w0.wav
│   │   ├── ...
```
### Reverberation
Please visit [mp3d_reverberation](https://huggingface.co/datasets/zhisheng01/SpatialSounds/blob/main/mp3d_reverb.zip) and download manually. Below is an example of the directory structure of the reverberation data.
```bash
/path/to/reverb_root
├── train_reverberation.json
├── eval_reverberation.json
├── binaural
│   ├── 17DRP5sb8fy
│   │   ├── 0.npy
│   │   ├── 10.npy
│   │   ├── 17DRP5sb8fy.json
│   │   ├── ...
│   ├── 1LXtFkjw3qL
│   │   ├── 0.npy
│   │   ├── 10.npy
│   │   ├── 1LXtFkjw3qL.json
│   │   ├── ...
├── mono
│   ├── 17DRP5sb8fy
│   ├── ...
```

## Train a new model
Training from scratch is pretty simple and easy. 
```bash
reverb_type=binaural # or mono / ambisonics (will be supported soon)
bash scripts/finetune-2m.sh $reverb_type
```

## Inference
We provide pretrained [checkpoint](https://huggingface.co/zhisheng01/Bat/blob/main/spatial-ast.pth).
You can do inference basically by 
```bash
# remember to replace `ckpt` variable with your local path
bash scripts/inf.sh
```
## TODO
The TODOs left will be completed before the end of June 2024.
- [x] Environment setup
- [x] Upload pretrained weights
- [x] Fix numba output bug
- [x] Update training data
- [ ] Replace tensorboard with W&B
- [ ] Inference colab

## Citation
```
@article{zheng2024bat,
  author    = {Zheng, Zhisheng and Peng, Puyuan and Ma, Ziyang and Chen, Xie and Choi, Eunsol and Harwath, David},
  title     = {BAT: Learning to Reason about Spatial Sounds with Large Language Models},
  journal   = {arXiv preprint arXiv:2402.01591},
  year      = {2024},
}
```
## Reference
The codebase is based on the [Audio-MAE](https://github.com/facebookresearch/AudioMAE/tree/main) repo.

## License
This project is under the CC-BY 4.0 license. See [LICENSE](LICENSE) for details.