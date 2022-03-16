

cls_type=blade_razor
image_width=320
split=test
output_dir=log_pnp_${split}


python pnp.py \
    --cls_type $cls_type \
    --image_width $image_width \
    --output_dir $output_dir \
    --split $split \
