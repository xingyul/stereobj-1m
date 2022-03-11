'''
    Single-GPU training.
'''
import argparse
import tabulate
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
parser.add_argument('--object_data', default='', help='Object data directory [default: ]')
parser.add_argument('--split', default='val', help='Dataset split [default: val]')
parser.add_argument('--input_json', default='', help='Input json filename [default: ]')
parser.add_argument('--gt_json', default='', help='GT json filename [default: ]')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


if __name__ == "__main__":

    with open(args.input_json, 'r') as f:
        input_dict = json.load(f)
    pred_dict = input_dict['pred']

    with open(args.gt_json, 'r') as f:
        gt_dict = json.load(f)

    assert input_dict['split'] == gt_dict['split']
    gt_dict = gt_dict['pred']

    result_dict = {}

    for cls_type in gt_dict:
        obj_points_fname = os.path.join(args.object_data, cls_type + '.xyz')
        with open(obj_points_fname, 'r') as f:
            data = f.read().rstrip().split()
            data = [float(d) for d in data]
        obj_points = np.reshape(np.array(data), [-1, 3])
        obj_diameter = evaluate_util.diameter(obj_points)
        ##### optional: use fewer objects points to speed up evaluation
        obj_points = obj_points[:1024]

        add_results = []

        for seq_id in gt_dict[cls_type]:
            for frame_id in gt_dict[cls_type][seq_id]:
                try:
                    assert seq_id in pred_dict[cls_type]
                    assert frame_id in pred_dict[cls_type][seq_id]
                except:
                    print('ERROR in {}'.format(seq_id + ',' + frame_id))
                    mean_diff = float(2**31)
                    add_results.append(mean_diff)
                    continue

                pose_pred = pred_dict[cls_type][seq_id][frame_id]
                pose_pred = np.array(pose_pred)

                pose_gt = gt_dict[cls_type][seq_id][frame_id]
                pose_gt = np.array(pose_gt)

                if cls_type in ['centrifuge_tube', 'microplate', 'needle_nose_pliers', \
                        'screwdriver', 'side_cutters', 'tube_rack_1.5_2_ml', \
                        'tube_rack_50_ml', 'wire_stripper']:
                    ret, (mean_diff, diameter) = evaluate_util.add_metric(pose_pred, pose_gt, \
                        obj_points, obj_diameter, symm=True, percentage=0.1, gpu=(args.gpu!=''))
                else:
                    ret, (mean_diff, diameter) = evaluate_util.add_metric(pose_pred, pose_gt, \
                        obj_points, obj_diameter, symm=False, percentage=0.1)

                print(seq_id+','+frame_id+','+str(ret)+','+str(mean_diff)+','+str(diameter))
                add_results.append(mean_diff)

        print(cls_type)

        max_thres = 0.1
        adds_auc_01m_thres = np.mean((max_thres - np.array(add_results)) / max_thres * (np.array(add_results) < max_thres)) * 100.
        adds_acc_01m_thres = np.mean(np.array(add_results) < max_thres) * 100.
        print('0.1 m threshold add auc: {}'.format(adds_auc_01m_thres))
        print('0.1 m add accuracy: {}'.format(adds_acc_01m_thres))

        max_thres = 0.1 * obj_diameter
        adds_auc_01d_thres = np.mean((max_thres - np.array(add_results)) / max_thres * (np.array(add_results) < max_thres)) * 100.
        adds_acc_01d_thres = np.mean(np.array(add_results) < max_thres) * 100.
        print('0.1 diameter add auc: {}'.format(adds_auc_01d_thres))
        print('0.1 diameter add accuracy: {}'.format(adds_acc_01d_thres))

        result_dict[cls_type] = {'0.1 m': {'accuracy': adds_acc_01m_thres,
                                           'auc': adds_auc_01m_thres},
                                 '0.1 diameter': {'accuracy': adds_acc_01d_thres,
                                                  'auc': adds_auc_01d_thres}, }

result_dict['Average'] = {}
for thres in ['0.1 m', '0.1 diameter']:
    result_dict['Average'][thres] = {}
    for metric in ['accuracy', 'auc']:
        result_dict['Average'][thres][metric] = \
                np.mean([result_dict[cls_type][thres][metric] for cls_type in result_dict if 'Average' not in cls_type])


head = ['', '0.1 m accuracy', '0.1 m auc', '0.1 d accuracy', '0.1 d auc']

cls_types = list(result_dict.keys())
cls_types.sort()

result_tabulate = [head]
for cls_type in cls_types + ['Average']:
    line = []
    for key_1 in ['0.1 m', '0.1 diameter']:
        for key_2 in ['accuracy', 'auc']:
            line.append(result_dict[cls_type][key_1][key_2])
    result_tabulate.append([cls_type] + line)

tab = tabulate.tabulate(result_tabulate)
print(tab)



