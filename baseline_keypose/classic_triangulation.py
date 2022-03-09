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
sys.path.append(os.path.join(BASE_DIR, '..'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

import evaluate_util
import orthogonal_procrustes



parser = argparse.ArgumentParser()
parser.add_argument('--split', default='test', help='Dataset split [default: test]')
parser.add_argument('--data', default='', help='Data path [default: ]')
parser.add_argument('--cls_type', default='', help='Object class of interest [default: ]')
parser.add_argument('--image_width', type=int, default=768, help='Image width [default: 768]')
parser.add_argument('--output_dir', default='', help='Output directory [default: ]')
args = parser.parse_args()


if not os.path.exists(args.output_dir):
    os.system('mkdir -p {}'.format(args.output_dir))


def ransac_orthogonal_procrustes(xyz, kpt_3d):
    num_min_sample = 4
    num_not_yet_inlier = 5
    best_R, best_t, best_error = None, None, 10000000
    num_kp = xyz.shape[0]
    for i in range(100):
        maybe_inliers_idx = np.random.choice(num_kp, num_min_sample)
        kpt_3d_ = kpt_3d[maybe_inliers_idx]
        xyz_ = xyz[maybe_inliers_idx]
        try:
            R, t, error = orthogonal_procrustes.orthogonal_procrustes(kpt_3d_, xyz_)
        except:
            continue
        if error < best_error:
            best_R, best_t, best_error = R, t, error

        not_yet_inlier_idx = np.delete(np.arange(num_kp), maybe_inliers_idx)
        np.random.shuffle(not_yet_inlier_idx)
        not_yet_inlier_idx = not_yet_inlier_idx[:num_not_yet_inlier]

        for idx in not_yet_inlier_idx:
            merged_inliers_idx = np.concatenate([maybe_inliers_idx, np.array([idx])])
            kpt_3d_ = kpt_3d[maybe_inliers_idx]
            xyz_ = xyz[maybe_inliers_idx]
            try:
                R, t, error = orthogonal_procrustes.orthogonal_procrustes(kpt_3d_, xyz_)
            except:
                continue
            if error < best_error:
                best_R, best_t, best_error = R, t, error
    return best_R, best_t, best_error


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
        print(idx, '/', len(all_saved_jsons))
        with open(filename, 'r') as f:
            data = json.load(f)
        K = np.array(data['K'])
        kpt_3d = np.array(data['kpt_3d'])
        baseline = np.array(data['baseline'])
        pred_kp_uv_l = np.array(data['pred_kp_uv_l'])
        pred_kp_uv_r = np.array(data['pred_kp_uv_r'])
        num_kp = kpt_3d.shape[0]

        pred_kp_uv_l = pred_kp_uv_l / args.image_width * 1440
        pred_kp_uv_r = pred_kp_uv_r / args.image_width * 1440

        disparity = (pred_kp_uv_l - pred_kp_uv_r)[:, 0]
        depth = K[0, 0] * baseline / disparity
        uv = (pred_kp_uv_l + pred_kp_uv_r) / 2
        xy = (uv - np.array([K[0, -1], K[1, -1]])) / K[0, 0] * np.expand_dims(depth, -1)
        xyz = np.concatenate([xy, np.expand_dims(depth, -1)], -1)

        ##### symmetric objects
        if args.cls_type in ['centrifuge_tube', 'microplate', 'needle_nose_pliers', \
                'screwdriver', 'side_cutters', 'tube_rack_1.5_2_ml', 'tube_rack_50_ml', \
                'wire_stripper']:

            R, t, cost = ransac_orthogonal_procrustes(xyz, kpt_3d)

            if cost > 5:
                perm = np.zeros([num_kp]).astype('int32')
                perm[::2] = np.arange(num_kp)[1::2]
                perm[1::2] = np.arange(num_kp)[::2]

                pred_kp_uv_r_ = pred_kp_uv_r[perm]
                disparity = (pred_kp_uv_l - pred_kp_uv_r_)[:, 0]
                depth = K[0, 0] * baseline / disparity
                uv = (pred_kp_uv_l + pred_kp_uv_r_) / 2
                xy = (uv - np.array([K[0, -1], K[1, -1]])) / K[0, 0] * np.expand_dims(depth, -1)
                xyz = np.concatenate([xy, np.expand_dims(depth, -1)], -1)

                R_, t_, cost_ = ransac_orthogonal_procrustes(xyz, kpt_3d)

                if cost_ < cost:
                    R = R_
                    t = t_
            if R is None:
                continue
        else:
            R, t, cost = ransac_orthogonal_procrustes(xyz, kpt_3d)

            if R is None:
                continue

        pred_R = R
        pred_t = np.expand_dims(t, -1)
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



