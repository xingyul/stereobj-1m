

import argparse
import cv2
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import json
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
import triangulation_object



parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='model_res34_backbone', help='Model name [default: model_res34_backbone]')
parser.add_argument('--model_path', default=None, help='Model checkpint path [default: ]')
parser.add_argument('--split', default='test', help='Dataset split [default: test]')
parser.add_argument('--dataset', default='stereobj1m_dataset', help='Dataset name [default: stereobj1m_dataset]')
parser.add_argument('--num_kp', type=int, default=64, help='Number of Keypoints [default: 1024]')
parser.add_argument('--num_workers', type=int, default=4, help='Number of multiprocessing workers [default: 1024]')
parser.add_argument('--image_width', type=int, default=768, help='Image width [default: 768]')
parser.add_argument('--image_height', type=int, default=768, help='Image height [default: 768]')
parser.add_argument('--data', default='', help='Data path [default: ]')
parser.add_argument('--cls_type', default='', help='Object class of interest [default: ]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 16]')
parser.add_argument('--debug', type=int, default=0, help='Debug mode [default: 0]')
parser.add_argument('--command_file', default=None, help='Command file name [default: None]')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

dataset = importlib.import_module(args.dataset)

BATCH_SIZE = args.batch_size
GPU_INDEX = args.gpu

MODEL = importlib.import_module(args.model) # import network module
MODEL_FILE = os.path.join(args.model+'.py')


# dataset
test_data_loader = make_data_loader.make_data_loader(args, lr=True, split=args.split)


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

            # Get model and loss
            end_points = MODEL.get_model(image_pl, args.num_kp, \
                    is_training=is_training_pl, debug=args.debug)

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
                dr = DictRestore(dict_for_restore, print)
                dr.run_init(sess)
                print("npz file restored.")
            elif '.h5' in args.model_path:
                f = h5py.File(args.model_path, 'r')
                dict_for_restore = {}
                for k in f.keys():
                    for group in f[k].items():
                        for g in group[1]:
                            dict_for_restore[os.path.join(k, g)] = group[1][g][:]
                            value = group[1][g][:]
                dict_for_restore = MODEL.name_mapping(dict_for_restore, debug=args.debug)
                dr = DictRestore(dict_for_restore, print)
                dr.run_init(sess)
                print("h5 file restored.")
            else:
                sr = SaverRestore(args.model_path, print) #, ignore=['batch:0'])
                sr.run_init(sess)
                print("Model restored.")

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

        ops = {'image_pl': image_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'end_points': end_points,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        eval_one_epoch(sess, ops)


def eval_one_epoch(sess, ops):
    """ ops: dict mapping from string to tf ops """
    is_training = False

    save_dir = os.path.join('log_lr_{}_preds'.format(args.split), args.cls_type)
    if not os.path.exists(save_dir):
        os.system('mkdir -p {}'.format(save_dir))

    for idx, batch in enumerate(test_data_loader):
        image_l = batch['inp_l'].data.numpy()
        image_r = batch['inp_r'].data.numpy()
        baseline = batch['baseline'].data.numpy()
        kpt_3d = batch['kpt_3d'].data.numpy()
        K = batch['K'].data.numpy()
        img_id = batch['img_id']

        img_id = [i[0] for i in img_id]
        K = K[0]
        kpt_3d = kpt_3d[0]
        baseline = baseline[0]

        feed_dict = {ops['image_pl']: image_l,
                     ops['is_training_pl']: is_training,}

        pred_kp_uv_val_l = sess.run(ops['end_points']['pred_kp_uv'], feed_dict=feed_dict)

        feed_dict = {ops['image_pl']: image_r,
                     ops['is_training_pl']: is_training,}

        pred_kp_uv_val_r = sess.run(ops['end_points']['pred_kp_uv'], feed_dict=feed_dict)

        pred_kp_uv_val_l = pred_kp_uv_val_l[:, :16]
        pred_kp_uv_val_r = pred_kp_uv_val_r[:, :16]

        view = False
        if view:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_show_l = image_l[0]
            image_show_r = image_r[0]
            pred_kp_uv_show_l = pred_kp_uv_val_l[0]
            pred_kp_uv_show_r = pred_kp_uv_val_r[0]

            image_show_l = image_show_l * std + mean
            image_show_r = image_show_r * std + mean
            plt.imshow(image_show_l)
            plt.plot(pred_kp_uv_show_l[:, 0], pred_kp_uv_show_l[:, 1], 'ro')
            plt.figure()
            plt.imshow(image_show_r)
            plt.plot(pred_kp_uv_show_r[:, 0], pred_kp_uv_show_r[:, 1], 'ro')
            plt.show()

        pred_kp_uv_val_l_ = np.copy(pred_kp_uv_val_l[0])
        pred_kp_uv_val_r_ = np.copy(pred_kp_uv_val_r[0])

        save_dict = {'pred_kp_uv_l': pred_kp_uv_val_l_.tolist(), \
                'pred_kp_uv_r': pred_kp_uv_val_r_.tolist(), 'kpt_3d': kpt_3d.tolist(), \
                'K': K.tolist(), 'baseline': baseline}
        save_path = os.path.join(save_dir, img_id[0] + '__' + img_id[1] + '.json')
        with open(save_path, 'w') as f:
            json.dump(save_dict, f, indent=4)


if __name__ == "__main__":
    train()
