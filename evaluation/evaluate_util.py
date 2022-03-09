

import numpy as np
import os
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'nn'))

from nn import nn_utils


def diameter(obj_points):
    '''
        obj_points: (N, 3) array
    '''
    dist_sq = -np.dot(obj_points, obj_points.T) * 2 + \
            np.square(np.linalg.norm(obj_points, axis=-1, keepdims=True)) + \
            np.square(np.linalg.norm(obj_points.T, axis=0, keepdims=True))
    dist_max = np.max(dist_sq)
    diameter = np.sqrt(dist_max)
    return diameter

def find_nn_idx(pc_src, pc_target):
    '''
        pc_src: (N1, 3) array
        pc_target: (N2, 3) array
    '''
    dist_sq = -np.dot(pc_src, pc_target.T) * 2 + \
            np.square(np.linalg.norm(pc_src, axis=-1, keepdims=True)) + \
            np.square(np.linalg.norm(pc_target.T, axis=0, keepdims=True))
    idx_min = np.argmin(dist_sq, axis=0)
    return idx_min


def add_metric(pose_pred, pose_targets, obj_points, diameter, symm=False, percentage=0.1, gpu=True):

    diam = diameter * percentage
    model_pred = np.dot(obj_points, pose_pred[:, :3].T) + pose_pred[:, 3]
    model_targets = np.dot(obj_points, pose_targets[:, :3].T) + pose_targets[:, 3]

    if symm:
        if gpu:
            idxs = nn_utils.find_nearest_point_idx(model_pred, model_targets)
        else:
            idxs = find_nn_idx(model_pred, model_targets)
        mean_dist = np.mean(np.linalg.norm(model_pred[idxs] - model_targets, 2, 1))
    else:
        mean_dist = np.mean(np.linalg.norm(model_pred - model_targets, axis=-1))

    return mean_dist < diam, (mean_dist, diam)

