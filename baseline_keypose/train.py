'''
    Single-GPU training.
'''
import argparse
import cv2
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import importlib
import os
import sys
import h5py


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(BASE_DIR, '..'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, '..', 'data_loader'))

import make_data_loader
from dict_restore import DictRestore
from saver_restore import SaverRestore



parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='model_res34_backbone', help='Model name [default: model_res34_backbone]')
parser.add_argument('--model_path', default=None, help='Model checkpint path [default: ]')
parser.add_argument('--split', default='train', help='Dataset split [default: train]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--dataset', default='stereobj1m_dataset', help='Dataset name [default: stereobj1m_dataset]')
parser.add_argument('--subset', default='biolab', help='Dataset subset [default: biolab]')
parser.add_argument('--num_kp', type=int, default=64, help='Number of Keypoints [default: 1024]')
parser.add_argument('--num_workers', type=int, default=4, help='Number of multiprocessing workers [default: 1024]')
parser.add_argument('--image_width', type=int, default=768, help='Image width [default: 768]')
parser.add_argument('--image_height', type=int, default=768, help='Image height [default: 768]')
parser.add_argument('--data', default='', help='Data path [default: ]')
parser.add_argument('--cls_type', default='', help='Object class of interest [default: ]')
parser.add_argument('--subsample_ratio', type=int, default=1, help='Data subsample ratio [default: 1]')
parser.add_argument('--multiprocess_workers', type=int, default=0, help='Number of multiprocess workers [default: 0]')
parser.add_argument('--queue_size', type=int, default=0, help='Queue size for data [default: 0]')
parser.add_argument('--skip_frames', type=int, default=1, help='Skip frames [default: 1]')
parser.add_argument('--max_epoch', type=int, default=251, help='Epoch to run [default: 251]')
parser.add_argument('--init_epoch', type=int, default=0, help='Initial epoch [default: 0]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 16]')
parser.add_argument('--eval', type=int, default=0, help='Eval mode [default: 0]')
parser.add_argument('--symm180', type=int, default=0, help='Symmetry 180 degrees [default: 0]')
parser.add_argument('--mask_loss', type=int, default=1, help='Mask loss [default: 1]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='momentum', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=2, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--debug', type=int, default=0, help='Debug mode [default: 0]')
parser.add_argument('--command_file', default=None, help='Command file name [default: None]')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

dataset = importlib.import_module(args.dataset)

BATCH_SIZE = args.batch_size
GPU_INDEX = args.gpu
MOMENTUM = args.momentum
OPTIMIZER = args.optimizer
DECAY_RATE = args.decay_rate
COMMAND_FILE = args.command_file

MODEL = importlib.import_module(args.model) # import network module
MODEL_FILE = os.path.join(args.model+'.py')
LOG_DIR = args.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('mkdir -p {}'.format(os.path.join(LOG_DIR, 'dump_image')))

os.system('cp %s %s' % (os.path.join('models', MODEL_FILE), LOG_DIR)) # bkp of model def
os.system('cp %s %s' % (__file__, LOG_DIR)) # bkp of train script
os.system('cp %s %s' % (COMMAND_FILE, LOG_DIR)) # bkp of command file
LOG_FOUT = open(os.path.join(LOG_DIR, 'log.txt'), 'w')
LOG_FOUT.write(str(args)+'\n')


BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(args.decay_step)
BN_DECAY_CLIP = 0.99


train_data_loader = make_data_loader.make_data_loader(args, lr=False, split=args.split)


DECAY_STEP = args.decay_step * len(train_data_loader) * 3

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        args.learning_rate,       # Base learning rate.
                        batch * args.batch_size,  # Current index into the dataset.
                        DECAY_STEP,                # Decay step.
                        args.decay_rate,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            image_pl, labels_pl = MODEL.placeholder_inputs( \
                    args.batch_size, args.image_height, args.image_width,
                    args.num_kp, debug=args.debug)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            end_points = MODEL.get_model(image_pl, args.num_kp, \
                    is_training=is_training_pl, debug=args.debug)
            if not args.debug:
                MODEL.get_loss(end_points, labels_pl, symm180=args.symm180)
                losses = tf.get_collection('losses')
                loss_uv = tf.get_collection('kp uv loss')[0]
                loss_prob = tf.get_collection('kp prob loss')[0]
                total_loss = tf.add_n(losses, name='total_loss')
                tf.summary.scalar('total_loss', total_loss)
                for l in losses:
                    tf.summary.scalar(l.op.name, l)

                print("--- Get training operator")
                # Get training operator
                # learning_rate = get_learning_rate(batch)
                learning_rate = tf.placeholder(tf.float32, shape=[])
                learning_rate = tf.maximum(learning_rate, 2e-6) # CLIP THE LEARNING RATE!
                tf.summary.scalar('learning_rate', learning_rate)
                if OPTIMIZER == 'momentum':
                    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
                elif OPTIMIZER == 'adam':
                    optimizer = tf.train.AdamOptimizer(learning_rate)
                train_op = optimizer.minimize(total_loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        if args.model_path is not None:
            if 'npz' in args.model_path:
                dict_file = np.load(args.model_path)
                dict_for_restore = {}
                dict_file_keys = dict_file.keys()
                for k in dict_file_keys:
                    dict_for_restore[k] = dict_file[k]
                dict_for_restore = MODEL.name_mapping(dict_for_restore, debug=args.debug)
                dr = DictRestore(dict_for_restore, log_string)
                dr.run_init(sess)
                log_string("npz file restored.")
            elif '.h5' in args.model_path:
                f = h5py.File(args.model_path, 'r')
                dict_for_restore = {}
                for k in f.keys():
                    for group in f[k].items():
                        for g in group[1]:
                            dict_for_restore[os.path.join(k, g)] = group[1][g][:]
                            value = group[1][g][:]
                dict_for_restore = MODEL.name_mapping(dict_for_restore, debug=args.debug)
                dr = DictRestore(dict_for_restore, log_string)
                dr.run_init(sess)
                log_string("h5 file restored.")
            else:
                sr = SaverRestore(args.model_path, log_string) #, ignore=['batch:0'])
                sr.run_init(sess)
                log_string("Model restored.")

        if args.debug:
            im = cv2.imread('green_mamba.jpg').astype('float32')
            if im.shape[0] < im.shape[1]:
                im = cv2.resize(im, (int(256. * float(im.shape[1]) / im.shape[0]), 256))
            else:
                im = cv2.resize(im, (256, int(256. * float(im.shape[0]) / im.shape[1])))
            im = im[int(im.shape[0]/2-112):int(im.shape[0]/2+112), int(im.shape[1]/2-112):int(im.shape[1]/2+112), :]
            im = im / 255
            mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, -1])
            std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, -1])
            im = (im - mean) / std
            wh = args.image_width
            im = cv2.resize(im, (wh, wh))
            im = np.reshape(im, [1, wh, wh, 3])
            pred_np = sess.run(end_points['pred'], feed_dict={image_pl: im, is_training_pl: False})
            pred_np = np.reshape(pred_np, [-1])
            print(pred_np.argsort()[-5:][::-1])
            exit()

        ops = {'learning_rate': learning_rate,
               'image_pl': image_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'end_points': end_points,
               'loss': total_loss,
               'loss_uv': loss_uv,
               'loss_prob': loss_prob,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        for epoch in range(args.init_epoch, args.max_epoch):
            lr = args.learning_rate * np.power(DECAY_RATE, epoch // args.decay_step)
            lr = np.maximum(lr, 3e-7) # CLIP THE LEARNING RATE!
            log_string('**** EPOCH %03d ****' % (epoch))
            log_string('learning_rate: {}'.format(lr))
            # log_string('learning_rate: {}'.format(sess.run(learning_rate)))
            sys.stdout.flush()

            train_one_epoch(sess, ops, lr, train_writer, epoch)

            # Save the variables to disk.
            if epoch % 1 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model-{}.ckpt".format(epoch)))
                log_string("Model saved in file: %s" % save_path)


def train_one_epoch(sess, ops, learning_rate, train_writer, epoch):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    log_string(str(datetime.now()))

    for idx, batch in enumerate(train_data_loader):
        image = batch['inp'].data.numpy()
        mask = batch['mask'].data.numpy()
        kp_uv = batch['uv'].data.numpy()
        prob = batch['prob'].data.numpy()

        view = False
        if view:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_show = image[0]
            kpt_2d = kp_uv[0]

            image_show = image_show * std + mean
            plt.imshow(image_show, cmap='jet')
            plt.plot(kpt_2d[:, 0], kpt_2d[:, 1], 'ro')
            plt.show()
            exit()

        feed_dict = {ops['learning_rate']: learning_rate,
                     ops['image_pl']: image,
                     ops['labels_pl']['kp_prob']: prob,
                     ops['labels_pl']['kp_uv']: kp_uv,
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, loss_uv_val, loss_prob_val = \
                sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], \
                ops['loss_uv'], ops['loss_prob']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)

        if (idx+1) % 1 == 0:
            log_string(' ---- batch: %04d/%4d ----' % (idx+1, len(train_data_loader)))
            log_string('loss: {}, loss uv: {}, loss prob: {}'\
                    .format(loss_val, loss_uv_val, loss_prob_val))


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
