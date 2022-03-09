## KeyPose Baseline for StereOBJ-1M

This document contains information about running and evaluating the KeyPose baseline.

### Prepare the Data

Please download the image and annotation data, camera parameters, object files and split files from the [Dropbox link](https://www.dropbox.com/sh/b1e5xuzysyxqg0a/AAANEt13l8zWSxcv7IkIzVEwa?dl=0).
The next step is to extract the tar files. The user needs to create the soft link or copy the files such that the file format looks like the following:

```
/path/to/images_annotations
    biolab_scene_10_08212020_1/
    biolab_scene_10_08212020_2/
    biolab_scene_10_08212020_3/
    ...
    mechanics_scene_10_08212020_1/
    mechanics_scene_10_08212020_2/
    mechanics_scene_10_08212020_3/
    ...
    objects/
    split/
    camera.json
```

### Training the Model

The command script for launching the training of KeyPose model is `command_train.sh`.
This script will train a net model of predicting object 2D projected keypoints.

Inside the command file, please set the value of `data` to be the path to be `/path/to/images_annotations`. The user may also choose to use ImageNet pre-trained weights to initialize the ResNet-34 backbone of the KeyPose model.
One choice is to download the pre-trained model from [here](https://github.com/tensorpack/tensorpack/blob/master/examples/ResNet/README.md).
Please set the value of `pretrained_models` flag to be the path to the downloaded net weights npz file.

#### KeyPose Trained Models

We provide trained KeyPose model [here](https://www.dropbox.com/sh/6p3iwfq7ffvzg6f/AABIfFFdgd4JA_0f43Pc2TDFa?dl=0).
These models are trained on **train+val sets**.
The models are used to report the test set numbers in our paper.
The users can try to run these models and get familiar with the evaluation scripts.

### Test a Trained Model

#### Keypoint Prediction
The first step of model evaluation is to predict the object 2D projected keypoints in both left and right stereo images.
The command script is `command_evaluate_lr.sh`, where "lr" means "left and right".
The keypoint results will be saved in `log_lr_${split}_preds/${cls_type}`, where the `${split}` is the dataset split, and `${cls_type}` is the object class name. Available values for split flag are `train`, `val`, `trainval` and `test`.

#### From Keypoint to 6D Pose
We provide three different ways for computing 6D pose from 2D projected keypoint predictions in stereo images:

1) Monocular PnP. The command script for launching monocular PnP is `command_pnp.sh`. The results are stored in `log_pnp_${split}/${cls_type}.json`.

2) Binocular classic triangulation. The command script for launching binocular classic triangulation is `command_classic_triangulation.sh`. The results are stored in `log_classic_triangulation_${split}/${cls_type}.json`.

3) Binocular object triangulation as proposed in our paper. The command script for launching binocular object triangulation is `command_object_triangulation.sh`. The results are stored in `log_object_triangulation_${split}/${cls_type}`. The user needs to run an additional script of `command_combine_pose_prediction.sh` to combine the json files into one `log_object_triangulation_${split}/${cls_type}.json`.

#### Merge Predictions from All Classes
In order to submit the results to EvalAI to get the results on test set, the user needs to merge predictions from all object classes into one single json file.
We provide `command_merge_json_all_classes.sh` for this purpose.
Please refer to `merge_json_all_classes.py` for more details.

### Evaluation Script

The command script for launching 6D pose evaluation is located in `../command_evaluate_add_s.sh`.
Please change the value of `input_json` flag in the script to be the path to the (merged) json file dumped from the previous step.








