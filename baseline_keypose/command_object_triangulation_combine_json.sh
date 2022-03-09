

cls_type=blade_razor
input_dir=log_object_triangulation_test/${cls_type}
output_dir=log_object_triangulation_test
split=test


python object_triangulation_combine_json.py \
    --input_dir $input_dir \
    --output_dir $output_dir \
    --split $split \
    --cls_type $cls_type
