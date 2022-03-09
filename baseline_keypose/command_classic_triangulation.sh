

data=/mnt/nas/xyl/stereobj_1m/images_annotations
cls_type=blade_razor
image_width=320
split=test
output_dir=log_classic_triangulation_${split}


python classic_triangulation.py \
    --cls_type $cls_type \
    --data $data \
    --image_width $image_width \
    --output_dir $output_dir \
    --split $split \
