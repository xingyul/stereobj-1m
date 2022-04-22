

command_file=`basename "$0"`
gpu=0
split=train
batch_size=8
learning_rate=0.0001
model=model_res34_backbone
model_path=pretrained_models/ImageNet-ResNet34.npz
num_kp=16
init_epoch=0
data=/path/to/stereobj_1m/
dataset=stereobj1m_dataset
cls_type=blade_razor
image_width=320
image_height=320
symm180=0
decay_step=2
decay_rate=0.5
optimizer=adam
eval=0
num_workers=8
max_epoch=100
debug=0
log_dir=log_${split}_${cls_type}


python train.py \
    --gpu $gpu \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --split $split \
    --model $model \
    --model_path $model_path \
    --init_epoch $init_epoch \
    --cls_type $cls_type \
    --data $data \
    --num_kp $num_kp \
    --dataset $dataset \
    --optimizer $optimizer \
    --eval $eval \
    --num_workers $num_workers \
    --image_width $image_width \
    --image_height $image_height \
    --decay_step $decay_step \
    --decay_rate $decay_rate \
    --symm180 $symm180 \
    --max_epoch $max_epoch \
    --command_file $command_file \
    --debug $debug \
    --log_dir $log_dir \
    > $log_dir.txt 2>&1 &
