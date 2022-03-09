'''
    Single-GPU training.
'''
import argparse
import glob
import cv2
import numpy as np
import json
import os
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(BASE_DIR, '..', 'evaluation'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

import evaluate_util



parser = argparse.ArgumentParser()
parser.add_argument('--split', default='test', help='Dataset split [default: test]')
parser.add_argument('--data', default='', help='Data path [default: ]')
parser.add_argument('--cls_type', default='', help='Object class of interest [default: ]')
parser.add_argument('--image_width', type=int, default=768, help='Image width [default: 768]')
parser.add_argument('--output_dir', default='', help='Output directory [default: ]')
args = parser.parse_args()


if not os.path.exists(args.output_dir):
    os.system('mkdir -p {}'.format(args.output_dir))


if __name__ == "__main__":

    obj_points_fname = os.path.join(args.data, 'objects/', args.cls_type + '.xyz')
    with open(obj_points_fname, 'r') as f:
        data = f.read().rstrip().split()
        data = [float(d) for d in data]
    obj_points = np.reshape(np.array(data), [-1, 3])
    obj_diameter = evaluate_util.diameter(obj_points)

    add_results = []
    proj_results = []

    pred_dir = os.path.join('log_lr_{}_preds'.format(args.split), args.cls_type)
    all_saved_jsons = glob.glob(os.path.join(pred_dir, '*.json'))

    pose_preds = {'pred': {}}
    for idx, filename in enumerate(all_saved_jsons):
        with open(filename, 'r') as f:
            data = json.load(f)
        K = np.array(data['K'])
        kpt_3d = np.array(data['kpt_3d'])
        pred_kp_uv_l = np.array(data['pred_kp_uv_l'])

        pred_kp_uv_l = pred_kp_uv_l / args.image_width * 1440

        # try:
        ret, rvec, tvec, _ = cv2.solvePnPRansac(kpt_3d, pred_kp_uv_l, K, distCoeffs=np.zeros([4]))
        '''
        except:
            mean_diff = np.inf
            ret = False
            continue
        '''

        pred_R, _ = cv2.Rodrigues(rvec)
        pred_t = tvec
        pose_pred = np.concatenate([pred_R, pred_t], axis=-1)
        pose_pred = pose_pred.tolist()

        sequence_id = os.path.basename(filename).split('__')[0]
        frame_id = os.path.basename(filename).split('.')[0].split('__')[1]
        if sequence_id not in pose_preds['pred']:
            pose_preds['pred'][sequence_id] = {}
        pose_preds['pred'][sequence_id][frame_id] = pose_pred

    pose_preds['cls_type'] = args.cls_type
    pose_preds['split'] = args.split

    save_filename = os.path.join(args.output_dir, '{}.json'.format(args.cls_type))
    with open(save_filename, 'w') as f:
        json.dump(pose_preds, f, indent=4)



