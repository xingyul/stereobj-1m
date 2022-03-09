

num_kp=16
data=/mnt/nas/xyl/stereobj_1m/images_annotations
cls_type=blade_razor
image_width=320
image_height=320
log_dir=log


python object_triangulation.py \
    --cls_type $cls_type \
    --data $data \
    --num_kp $num_kp \
    --image_width $image_width \
    --image_height $image_height \
    > $log_dir.txt 2>&1 &
