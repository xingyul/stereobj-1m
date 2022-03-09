

gpu=0
gt_json=/mnt/nas/xyl/stereobj_1m/test_label_merged.json
input_json=../baseline_keypose/log_pnp_test/merged.json
split=test
object_data=/mnt/nas/xyl/stereobj_1m/images_annotations/objects


python evaluate_add_s_overall.py \
    --gpu $gpu \
    --split $split \
    --object_data $object_data \
    --input_json $input_json \
    --gt_json $gt_json
