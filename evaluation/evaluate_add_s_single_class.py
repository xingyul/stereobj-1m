'''
    Single-GPU training.
'''
import argparse
import cv2
import numpy as np
import json
import os
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(BASE_DIR, '..'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

import evaluate_util



parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='', help='GPU to use [default: ]')
parser.add_argument('--data', default='', help='Data path [default: ]')
parser.add_argument('--split', default='val', help='Dataset split [default: val]')
parser.add_argument('--input_json', default='', help='Pose prediction save filename [default: ]')
parser.add_argument('--gt_dir', default='', help='GT directory [default: ]')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


if __name__ == "__main__":

    with open(args.input_json, 'r') as f:
        result_dict = json.load(f)

    cls_type = result_dict['cls_type']
    pred_dict = result_dict['pred']

    with open(os.path.join(args.data, 'split', '{}_{}.json'.format(args.split, cls_type)), 'r') as f:
        all_ids = json.load(f)

    obj_points_fname = os.path.join(args.data, 'objects/', cls_type + '.xyz')
    with open(obj_points_fname, 'r') as f:
        data = f.read().rstrip().split()
        data = [float(d) for d in data]
    obj_points = np.reshape(np.array(data), [-1, 3])
    obj_diameter = evaluate_util.diameter(obj_points)

    add_results = []

    for seq_id in all_ids:
        for frame_id in all_ids[seq_id]:
            try:
                assert seq_id in pred_dict
                assert frame_id in pred_dict[seq_id]
            except:
                print('ERROR in {}'.format(seq_id + ',' + frame_id))
                mean_diff = np.inf
                add_results.append(mean_diff)
                continue

            pose_pred = pred_dict[seq_id][frame_id]
            pose_pred = np.array(pose_pred)

            gt_file = os.path.join(args.gt_dir, seq_id, frame_id + '_rt_label.json')
            with open(gt_file, 'r') as f:
                gt = json.load(f)
                pose_rt_gt = [gt['rt'][obj] for obj in gt['class'] if gt['class'][obj] == cls_type]
            pose_rt_gt = pose_rt_gt[0]
            pose_gt = np.concatenate([np.array(pose_rt_gt['R']), \
                    np.array(pose_rt_gt['t']).reshape([3,1])], axis=-1)

            if cls_type in ['centrifuge_tube', 'microplate', 'needle_nose_pliers', \
                    'screwdriver', 'side_cutters', 'tube_rack_1.5_2_ml', 'tube_rack_50_ml', \
                    'wire_stripper']:
                ret, (mean_diff, diameter) = evaluate_util.add_metric(pose_pred, pose_gt, obj_points, \
                                 obj_diameter, symm=True, percentage=0.1, gpu=(args.gpu!=''))
            else:
                ret, (mean_diff, diameter) = evaluate_util.add_metric(pose_pred, pose_gt, obj_points, \
                                 obj_diameter, symm=False, percentage=0.1)

            add_results.append(mean_diff)
            print(seq_id+','+frame_id+','+str(ret)+','+str(mean_diff)+','+str(diameter))


    max_thres = 0.1
    print('0.1 m threshold add auc: {}'.format(np.mean((max_thres - np.array(add_results)) / max_thres * (np.array(add_results) < max_thres)) * 100.))
    print('0.1 m add accuracy: {}'.format(np.mean(np.array(add_results) < max_thres) * 100.))
    max_thres = 0.1 * obj_diameter
    print('0.1 diameter threshold add auc: {}'.format(np.mean((max_thres - np.array(add_results)) / max_thres * (np.array(add_results) < max_thres)) * 100.))
    print('0.1 diameter add accuracy: {}'.format(np.mean(np.array(add_results) < max_thres) * 100.))



