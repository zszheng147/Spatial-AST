# Spatial-AST

This repo hosts the code and models of "[BAT: Learning to Reason about Spatial Sounds with Large Language Models](https://arxiv.org/abs/2402.01591)" [Submitted to ICML 2024 [bib](https://github.com/zszheng147/Spatial-AST#citation)].


<p align="center">
  <img align="middle" height="300" src="assets/Spatial-AST.png"/>
</p>

### Installation
```
conda env create -f environment.yml
bash timm_patch/patch.sh
```

### TODO
The TODOs left will be completed before the end of May 2024.
- [x] Environment setup
- [ ] Replace tensorboard with W&B
- [ ] Inference colab
- [ ] Upload training data: SpatialSoundQA
- [ ] Upload pretrained weights
- [ ] Fix numba output bug


### Citation
```
@article{zheng2024bat,
  author    = {Zheng, Zhisheng and Peng, Puyuan and Ma, Ziyang and Chen, Xie and Choi, Eunsol and Harwath, David},
  title     = {BAT: Learning to Reason about Spatial Sounds with Large Language Models},
  journal   = {arXiv preprint arXiv:2402.01591},
  year      = {2024},
}
```

### Reference
The codebase is based on the [Audio-MAE](https://github.com/facebookresearch/AudioMAE/tree/main) repo.

### License
This project is under the CC-BY 4.0 license. See [LICENSE](LICENSE) for details.