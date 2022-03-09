
## Evaluation Scripts

### Compile Module for Nearest Neighbor Index Computation

Evaluation of symmetric object pose needs to compute the index of nearest point neighbor.
The user can choose to use GPU to speed up the computation.
Please first compile the module by `cd` into `nn/` directory and `python setup.py`.


### Evaluation for a Single Object Class

The command script for launching 6D pose evaluation on a single object class is `command_evaluate_add_s_single_class.sh`.
Please change the value of `input_json` flag in the script to be the path to the input json file that contains the pose predictions of an object class.

The example format of the input json file is
```
{
    "cls_type": "blade_razor",
    "split": "test",
    "pred": {
        "mechanics_scene_1_07272020_13": {
            "000000": [
                [1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
            ],
            "000001": [
                [1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
            ],
            ...
        },
        "mechanics_scene_2_08012020_17": {
            "000000": [ ...
            ],
            ...
        },
        ...
    },
}
```
1) The value of the `cls_type` key should be the object class name.

2) The value of the `split` key should be the dataset split (usually "val" or "test").

3) The value of the `pred` key should be the 6D pose predictions of each frame in each sequence, indexed by sequence name (e.g. "mechanics_scene_1_07272020_13") and frame index (e.g. "000000"). The 6D pose should be represented by a 3x4 array of [R|t], where R is the 3x3 rotation matrix and t is the 3x1 translation vector.

### Evaluation for All Object Classes

The command script for launching 6D pose evaluation on all object classes is `command_evaluate_add_s_overall.sh`.
Please change the value of `input_json` flag in the script to be the path to the input json file that contains the pose predictions of an object class.
The python script `evaluate_add_s_overall.py` is also used in our EvalAI challenge.

Different from a single object class, the example format of the input json file that contains pose predictions of all object classes is
```
{
    "split": "test",
    "pred": {
        "blade_razor": {
            "mechanics_scene_1_07272020_13": {
                "000000": [
                    [1., 0., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 0., 1., 0.],
                ],
                "000001": [
                    [1., 0., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 0., 1., 0.],
                ],
                ...
            },
            "mechanics_scene_2_08012020_17": {
                "000000": [ ...
                ],
                ...
            },
            ...
        },
        "hammer": {
            "mechanics_scene_11_08212020_3": {
                ...
            },
            ...
        },
    },
}
```
1) The value of the `split` key should be the dataset split (usually "val" or "test").

2) The value of the `pred` key should be the 6D pose predictions of each frame in each sequence, indexed by object class name (e.g. "blade_razor"), sequence name (e.g. "mechanics_scene_1_07272020_13") and frame index (e.g. "000000"). The 6D pose should be represented by a 3x4 array of [R|t], where R is the 3x3 rotation matrix and t is the 3x1 translation vector.

**The above format for all object classes is the required format for the submission to EvalAI!**





