# StereOBJ-1M: Large-scale Stereo Image Dataset for 6D Object Pose Estimation

Created by <a href="http://xingyul.github.io">Xingyu Liu</a>, <a href="https://sh8.io/">Shun Iwase</a> and <a href="http://www.cs.cmu.edu/~kkitani/">Kris Kitani</a> from The Robotics Institute of Carnegie Mellon University.

[[arXiv]](https://arxiv.org/abs/2109.10115) [[project]](https://sites.google.com/view/stereobj-1m)

<img src="https://github.com/xingyul/stereobj-1m/blob/master/doc/stereobj_1m_teaser.jpg" width="100%">

## Citation
If you find this work useful in your research, please cite:
```
@inproceedings{liu2021stereobj1m,
title={StereOBJ-1M: Large-scale Stereo Image Dataset for 6D Object Pose Estimation},
author={Xingyu Liu and Shun Iwase and Kris M. Kitani},
booktitle={ICCV},
year={2021}
}
```

## Abstract

We present a large-scale stereo RGB image object pose estimation dataset named the StereOBJ-1M dataset. The dataset is designed to address challenging cases such as object transparency, translucency, and specular reflection, in addition to the common challenges of occlusion, symmetry, and variations in illumination and environments. In order to collect data of sufficient scale for modern deep learning models, we propose a novel method for efficiently annotating pose data in a multi-view fashion that allows data capturing in complex and flexible environments. Fully annotated with 6D object poses, our dataset contains over 393K frames and over 1.5M annotations of 18 objects recorded in 182 scenes constructed in 11 different environments. The 18 objects include 8 symmetric objects, 7 transparent objects, and 8 reflective objects. We benchmark two state-of-the-art pose estimation frameworks on StereOBJ-1M as baselines for future work. We also propose a novel object-level pose optimization method for computing 6D pose from keypoint predictions in multiple images.


## Data Download

The data can be downloaded [here](https://www.dropbox.com/sh/b1e5xuzysyxqg0a/AAANEt13l8zWSxcv7IkIzVEwa?dl=0). You can find the stereo images, 6D pose annotations, pre-generated instance masks, bounding boxes, and dataset split information.

## Data Loader

We provide an implementation of the data loader in [data_loader/](data_loader/). Feel free to adapt it for your own use (e.g. add more augmentation, improve speed etc.). The data loader implmentation is used in the following KeyPose baseline in [baseline_keypose/](baseline_keypose/).

## KeyPose Baseline

We implementated [KeyPose](https://arxiv.org/abs/1912.02805) as a baseline method. The code for the baseline is in [baseline_keypose/](baseline_keypose/). Please refer to [baseline_keypose/README.md](baseline_keypose/README.md) for more details on how to use the code.


## Evaluation Script and File format
The command scripts for launching 6D pose evaluation is located in [evaluation/](evaluation/).
Please refer to [evaluation/README.md](evaluation/README.md) for more details on how to use the evaluation script.

## Test Set Performance and StereOBJ-1M Challenge
The annotations of the test set of our StereOBJ-1M are held out. To obtain test set performance, please submit your method's prediction results on the test set to [StereOBJ-1M Challenge on EvalAI](https://eval.ai/web/challenges/challenge-page/1645).
The for submission instructions, please refer to the description in the challenge or [instructions on our project website](https://sites.google.com/view/stereobj-1m/submission).


## License
Our code is released under MIT License (see LICENSE file for details).


