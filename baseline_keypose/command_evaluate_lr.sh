

command_file=`basename "$0"`
gpu=0
cls_type=sterile_tip_rack_10
model_path=log_train_sterile_tip_rack_10/model-15.ckpt
split=test
batch_size=1
model=model_res34_backbone
# model_path=pretrained_models/ImageNet-ResNet34.npz
num_kp=16
data=/mnt/nas/xyl/stereobj_1m/images_annotations
dataset=stereobj1m_dataset
image_width=320
image_height=320
debug=0
log_dir=log


python evaluate_lr.py \
    --gpu $gpu \
    --batch_size $batch_size \
    --split $split \
    --model $model \
    --model_path $model_path \
    --cls_type $cls_type \
    --data $data \
    --dataset $dataset \
    --num_kp $num_kp \
    --image_width $image_width \
    --image_height $image_height \
    --command_file $command_file \
    --debug $debug \
    > $log_dir.txt 2>&1 &
