

gpu=0
object_data=/path/to/images_annotations/objects

##### for evaluating merged validation set performance
gt_json=/path/to/val_label_merged.json # downloaded merged ground truth file
input_json=../baseline_keypose/log_pnp_val/merged.json # user input file
# input_json=../baseline_keypose/log_classic_triangulation_val/merged.json
# input_json=../baseline_keypose/log_object_triangulation_val/merged.json

##### INTERNAL USE ONLY: for evaluating merged test set performance
# gt_json=/path/to/test_label_merged.json # internal merged test set ground truth file
# input_json=../baseline_keypose/log_pnp_test/merged.json
# input_json=../baseline_keypose/log_classic_triangulation_test/merged.json
# input_json=../baseline_keypose/log_object_triangulation_test/merged.json


python evaluate_add_s_overall.py \
    --gpu $gpu \
    --split $split \
    --object_data $object_data \
    --input_json $input_json \
    --gt_json $gt_json
