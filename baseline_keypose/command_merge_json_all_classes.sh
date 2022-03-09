

input_dir=log_object_triangulation_test
output_dir=log_object_triangulation_test
split=test


python merge_json_all_classes.py \
    --input_dir $input_dir \
    --output_dir $output_dir \
    --split $split
