

gpu=0
data=/path/to/stereobj_1m/

##### for evaluating validation set performance of an object, e.g. microplate
split=val
gt_dir=/path/to/stereobj_1m/
input_json=../baseline_keypose/log_pnp_val/microplate.json
# input_json=../baseline_keypose/log_classic_triangulation_val/microplate.json
# input_json=../baseline_keypose/log_object_triangulation_val/microplate.json

##### INTERNAL USE ONLY: for evaluating test set performance of an object, e.g. microplate
# split=test
# gt_dir=/path/to/test_label # internal test ground truth directory
# input_json=../baseline_keypose/log_pnp_test/microplate.json
# input_json=../baseline_keypose/log_classic_triangulation_test/microplate.json
# input_json=../baseline_keypose/log_object_triangulation_test/microplate.json


python evaluate_add_s_single_class.py \
    --gpu $gpu \
    --data $data \
    --split $split \
    --input_json $input_json \
    --gt_dir $gt_dir
