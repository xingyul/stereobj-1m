

import os
import sys
import tensorflow as tf
import numpy as np
import copy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, '..', '..', 'utils'))
sys.path.append(os.path.join(BASE_DIR, '..', '..'))

import tf_util

def name_mapping(var_dict, debug=False):
    keys = var_dict.keys()
    mapped_dict = {}
    for k in keys:
        key = k.split(':0')[0]
        new_key = key
        if '/W' in key:
            new_key = key.replace('/W', '/weights')
        elif '/mean/EMA' in key:
            new_key = key.replace('/mean/EMA', '/moving_mean')
        elif '/variance/EMA' in key:
            new_key = key.replace('/variance/EMA', '/moving_variance')
        mapped_dict[new_key] = var_dict[k]
    if debug:
        mapped_dict['fc/biases'] = var_dict['linear/b:0']
        mapped_dict['fc/weights'] = var_dict['linear/W:0']
    return mapped_dict


def placeholder_inputs(batch_size, image_height, image_width, num_kp, debug=False):
    image_pl = tf.placeholder(shape=[batch_size, image_height, image_width, 3], dtype=tf.float32)
    label_kp_prob_pl = tf.placeholder(shape=[batch_size, image_height, image_width, num_kp], \
            dtype=tf.float32)
    label_kp_uv_pl = tf.placeholder(shape=[batch_size, num_kp, 2], dtype=tf.float32)
    label_pl = {'kp_prob': label_kp_prob_pl, 'kp_uv': label_kp_uv_pl}

    if debug:
        label_pl = tf.placeholder(shape=[batch_size], dtype=tf.int32)

    return image_pl, label_pl

def get_model(image, num_kp, is_training, bn_decay=0.999, weight_decay=0.0001, \
        early_exit=None, eval=False, seg_only=False, debug=False):
    """ ResNet, input is BxHxWx3, output  """
    batch_size = image.get_shape()[0].value
    image_height = image.get_shape()[1].value
    image_width = image.get_shape()[2].value
    end_points = {}

    channel_stride = [(64, 1), (128, 2), (256, 2), (512, 2)]
    # res block options
    num_blocks = [3, 4, 6, 3]
    # num_blocks = [2, 2, 2, 2]

    net = tf_util.conv2d(image, 64, [7, 7], stride=[2, 2], bn=True, bn_decay=bn_decay, is_training=is_training, weight_decay=weight_decay, scope='conv0')
    # net = tf_util.max_pool2d(net, [3, 3], stride=[2, 2], scope='pool0', padding='SAME')

    end_points['skip_0'] = net

    for gp, cs in enumerate(channel_stride):
        n_channels = cs[0]
        stride = cs[1]
        with tf.variable_scope('group{}'.format(gp)):
            for i in range(num_blocks[gp]):
                with tf.variable_scope('block{}'.format(i)):
                    if i == 0:
                        net_bra = tf_util.conv2d(net, n_channels, [3, 3], stride=[stride, stride], \
                                bn=True, bn_decay=bn_decay, is_training=is_training, \
                                weight_decay=weight_decay, scope='conv1')
                    else:
                        net_bra = tf_util.conv2d(net, n_channels, [3, 3], stride=[1, 1], \
                                bn=True, bn_decay=bn_decay, is_training=is_training, \
                                weight_decay=weight_decay, scope='conv1')
                    net_bra = tf_util.conv2d(net_bra, n_channels, [3, 3], stride=[1, 1], \
                            bn=True, bn_decay=bn_decay, is_training=is_training, \
                            activation_fn=None, weight_decay=weight_decay, scope='conv2')
                    if net.get_shape()[-1].value != n_channels:
                        net = tf_util.conv2d(net, n_channels, [1, 1], stride=[stride, stride], \
                                bn=True, bn_decay=bn_decay, is_training=is_training, \
                                activation_fn=None, weight_decay=weight_decay, scope='convshortcut')
                    net = net + net_bra
                    if i == (num_blocks[gp] - 1):
                        end_points['skip_{}'.format(gp+1)] = net
                        if early_exit is not None:
                            if 'skip_{}'.format(gp+1) == early_exit:
                                return end_points
                    net = tf.nn.relu(net)

    if debug:
        net = tf.reduce_mean(net, [1,2])
        net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp')
        net = tf_util.fully_connected(net, 1000, activation_fn=None, weight_decay=weight_decay, scope='fc')
        end_points['pred'] = net
        return end_points

    deconv_channels = [256, 256]
    deconv_strides = [2, 2] #, 2]
    for i, c in enumerate(deconv_channels):
        net = tf_util.conv2d(net, deconv_channels[i], [5, 5], \
                stride=[1, 1], activation_fn=tf.nn.relu, \
                bn=True, bn_decay=bn_decay, is_training=is_training, scope='deconv{}'.format(i))
        if deconv_strides[i] > 1:
            height = net.get_shape()[1].value
            width = net.get_shape()[2].value
            net = tf.image.resize_images(net, [height*deconv_strides[i], width*deconv_strides[i]])

        # print(net, end_points['skip_{}'.format(3-i)])
        net = tf.concat([net, end_points['skip_{}'.format(3-i)]], axis=-1)
        # net = net + end_points['skip_{}'.format(3-i)]
        net = tf_util.conv2d(net, deconv_channels[i], [1, 1], \
                stride=[1, 1], activation_fn=None, \
                bn=True, bn_decay=bn_decay, is_training=is_training, scope='deconv_conv{}'.format(i))

    net = tf_util.conv2d(net, 256, [5, 5], stride=[1, 1], activation_fn=tf.nn.relu, \
            bn=True, bn_decay=bn_decay, is_training=is_training, scope='deconv3')
    end_points['return_1'] = net

    ##### kp prob
    net_kp = tf_util.conv2d(net, 128, [5, 5], stride=[1, 1], activation_fn=tf.nn.relu, \
            bn=True, bn_decay=bn_decay, is_training=is_training, scope='deconv_kp_1')

    height = net_kp.get_shape()[1].value
    width = net_kp.get_shape()[2].value
    net_kp = tf.image.resize_images(net_kp, [height*2, width*2])

    net_kp = tf_util.conv2d(net_kp, 128, [5, 5], stride=[1, 1], activation_fn=tf.nn.relu, \
            bn=True, bn_decay=bn_decay, is_training=is_training, scope='deconv_kp_2')

    net_kp = tf.image.resize_images(net_kp, [image_height, image_width])

    net_kp = tf_util.conv2d(net_kp, num_kp, [1, 1], stride=[1, 1], \
            bn=False, activation_fn=None, weight_decay=weight_decay, scope='final_kp')

    net_kp_prob = tf.reshape(net_kp, [batch_size, -1, num_kp])
    net_kp_prob = tf.nn.softmax(net_kp_prob, axis=1)
    net_kp_prob = tf.reshape(net_kp_prob, [batch_size, image_height, image_width, num_kp])
    ##### ~ kp prob

    ##### kp uv
    xx, yy = np.meshgrid(np.arange(image_width), np.arange(image_height))
    xx = tf.constant(np.reshape(xx, [1, image_height, image_width, 1]).astype('float32'))
    yy = tf.constant(np.reshape(yy, [1, image_height, image_width, 1]).astype('float32'))

    x = tf.reduce_sum(net_kp_prob * xx, axis=[1,2])
    y = tf.reduce_sum(net_kp_prob * yy, axis=[1,2])
    xy = tf.stack([x, y], axis=-1)
    ##### ~ kp uv

    end_points['pred_kp'] = net_kp
    end_points['pred_kp_prob'] = net_kp_prob
    end_points['pred_kp_uv'] = xy

    return end_points


def get_loss(end_points, labels, symm180=False):

    batch_size = end_points['pred_kp_prob'].get_shape()[0].value
    image_height = end_points['pred_kp_prob'].get_shape()[1].value
    image_width = end_points['pred_kp_prob'].get_shape()[2].value

    pred_kp_prob = end_points['pred_kp_prob']
    pred_kp = end_points['pred_kp']
    pred_xy = end_points['pred_kp_uv']
    label_kp_prob = labels['kp_prob']
    label_kp_uv = labels['kp_uv']

    num_kp = end_points['pred_kp_prob'].get_shape()[-1].value

    permutations = []
    permutations.append(tf.constant(np.arange(num_kp).astype('int32')))
    if symm180:
        perm = np.zeros([num_kp]).astype('int32')
        perm[::2] = np.arange(num_kp)[1::2]
        perm[1::2] = np.arange(num_kp)[::2]
        permutations.append(tf.constant(perm))

    ##### kp prob loss
    kp_prob_losses = []
    for i in range(len(permutations)):
        labels = tf.gather(label_kp_prob, permutations[i], axis=-1)
        kp_prob_loss = tf.nn.softmax_cross_entropy_with_logits(\
                logits=pred_kp, labels=labels)
        kp_prob_loss = tf.reduce_mean(kp_prob_loss, axis=[1,2])
        kp_prob_losses.append(kp_prob_loss)
    kp_prob_loss = tf.stack(kp_prob_losses, -1)
    kp_prob_loss = tf.reduce_min(kp_prob_loss, -1)
    kp_prob_loss = tf.reduce_mean(kp_prob_loss)

    ##### kp dist loss
    kp_uv_losses = []
    for i in range(len(permutations)):
        labels = tf.gather(label_kp_uv, permutations[i], axis=-2)
        kp_uv_loss = tf.norm((labels - pred_xy) / 10, axis=-1)
        # kp_uv_loss = tf.norm((labels - pred_xy), axis=-1)
        kp_uv_loss = tf.reduce_mean(kp_uv_loss, axis=-1)
        kp_uv_losses.append(kp_uv_loss)
    kp_uv_loss = tf.stack(kp_uv_losses, -1)
    kp_uv_loss = tf.reduce_min(kp_uv_loss, -1)
    kp_uv_loss = tf.reduce_mean(kp_uv_loss)

    tf.add_to_collection('kp prob loss', kp_prob_loss)
    tf.add_to_collection('kp uv loss', kp_uv_loss)

    loss = kp_prob_loss + kp_uv_loss
    tf.add_to_collection('losses', loss)
    return


if __name__=='__main__':
    with tf.Graph().as_default():
        batch_size=32
        image_height=320
        image_width=320
        symm180=1
        num_kp=16

        image_pl, label_pl = placeholder_inputs(batch_size, image_height, image_width, num_kp)
        end_points = get_model(image_pl, num_kp, is_training=tf.constant(True))
        get_loss(end_points, label_pl, symm180)
