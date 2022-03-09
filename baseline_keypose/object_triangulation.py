'''
    Single-GPU training.
'''
import argparse
import glob
import numpy as np
import json
import os
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

import triangulation_object



parser = argparse.ArgumentParser()
parser.add_argument('--num_kp', type=int, default=64, help='Number of Keypoints [default: 1024]')
parser.add_argument('--image_width', type=int, default=768, help='Image width [default: 768]')
parser.add_argument('--image_height', type=int, default=768, help='Image height [default: 768]')
parser.add_argument('--split', default='test', help='Dataset split [default: test]')
parser.add_argument('--data', default='', help='Data path [default: ]')
parser.add_argument('--cls_type', default='', help='Object class of interest [default: ]')
args = parser.parse_args()


result_save_dir = os.path.join('log_object_triangulation_{}'.format(args.split), args.cls_type)
if not os.path.exists(result_save_dir):
    os.system('mkdir -p {}'.format(result_save_dir))


if __name__ == "__main__":

    add_results = []

    pred_dir = os.path.join('log_lr_{}_preds'.format(args.split), args.cls_type)
    all_saved_jsons = glob.glob(os.path.join(pred_dir, '*.json'))

    for idx, filename in enumerate(all_saved_jsons):
        save_filename = os.path.join(result_save_dir, os.path.basename(filename))
        if os.path.exists(save_filename):
            continue

        with open(filename, 'r') as f:
            data = json.load(f)
        K = np.array(data['K'])
        kpt_3d = np.array(data['kpt_3d'])
        # pose_gt = np.array(data['pose_gt'])
        baseline = np.array(data['baseline'])

        pred_kp_uv_l = np.array(data['pred_kp_uv_l'])
        pred_kp_uv_r = np.array(data['pred_kp_uv_r'])
        # pose_gt = np.array(data['pose_gt'])

        total_num_points = 2 * args.num_kp
        kps_2ds = np.concatenate([pred_kp_uv_l, pred_kp_uv_r], axis=0)
        # kps_2ds = np.concatenate([kp_uv_l, kp_uv_r], axis=0)
        kpt_3ds = np.concatenate([kpt_3d] * 2, axis=0)
        proj_mats = np.tile(np.expand_dims(K, 0), [total_num_points, 1, 1])

        tvecs = np.concatenate([np.zeros([args.num_kp, 3]), \
                np.tile(np.array([[-baseline, 0, 0]]), [args.num_kp, 1])], 0)

        R, t, cost = triangulation_object.triangulation_kp_dist_ransac(kps_2ds, kpt_3ds, proj_mats, \
                tvecs, reprojection_error_type='linear', \
                resolution_scale=args.image_width/1440., return_cost=True)
        pred_R = R
        pred_t = np.expand_dims(t, -1)
        pose_pred = np.concatenate([pred_R, pred_t], axis=-1)
        pose_pred = pose_pred.tolist()

        with open(save_filename, 'w') as f:
            json.dump(pose_pred, f, indent=4)



