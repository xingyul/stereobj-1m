

gpu=1
data=/mnt/nas/xyl/stereobj_1m/images_annotations
gt_dir=/mnt/nas/xyl/stereobj_1m/test_label
input_json=../baseline_keypose/log_pnp_test/microplate.json
split=test


python evaluate_add_s_single_class.py \
    --gpu $gpu \
    --data $data \
    --split $split \
    --input_json $input_json \
    --gt_dir $gt_dir
