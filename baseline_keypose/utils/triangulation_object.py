

import os
import sys
import time
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import pathos.multiprocessing as  multiprocessing

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '..'))

import reprojection_error
import pnp_obj


def triangulation_kp_dist_ransac(xy_maxs, obj_kps, proj_mats, tvecs, \
        reprojection_error_type='linear', resolution_scale=1024/1440., return_cost=False):

    assert(len(proj_mats) == len(xy_maxs))
    assert(len(tvecs) == len(proj_mats))

    ##### debug
    # xy_maxs = xy_maxs[:16]
    # obj_kps = obj_kps[:16]
    # proj_mats = proj_mats[:16]
    # tvecs = tvecs[:16]

    num_kp = xy_maxs[0].shape[0]
    num_min_sample = 7
    num_not_yet_inlier = 5
    max_iter = 50
    per_pixel_threshold = 10

    best_R, best_t, best_cost = None, None, 100000000

    def process_one_sample(args):
        xy_maxs, obj_kps, proj_mats, tvecs, reprojection_error_type, multiprocessing_idx = args

        best_R, best_t, best_cost = None, None, 100000000
        # np.random.seed(int(time.time()) + multiprocessing_idx)
        np.random.seed(multiprocessing_idx)

        maybe_inliers_idx = np.random.choice( \
                xy_maxs.shape[0], num_min_sample, replace=False)

        xy_maxs_input = xy_maxs[maybe_inliers_idx]
        obj_kps_input = obj_kps[maybe_inliers_idx]
        proj_mats_input = proj_mats[maybe_inliers_idx]
        tvecs_input = tvecs[maybe_inliers_idx]

        maybe_R, maybe_t, maybe_cost = pnp_obj.pnp_obj(xy_maxs_input, obj_kps_input, \
                proj_mats_input, tvecs_input, \
                reprojection_error_type=reprojection_error_type, return_cost=True, \
                resolution_scale=resolution_scale)
        if maybe_cost > per_pixel_threshold: return maybe_R, maybe_t, maybe_cost
        if maybe_cost < best_cost:
            best_R, best_t, best_cost = maybe_R, maybe_t, maybe_cost

        not_yet_inlier_idx = np.delete(np.arange(num_kp), maybe_inliers_idx)
        np.random.shuffle(not_yet_inlier_idx)
        not_yet_inlier_idx = not_yet_inlier_idx[:num_not_yet_inlier]

        for idx in not_yet_inlier_idx:
            merged_inliers_idx = np.concatenate([maybe_inliers_idx, np.array([idx])])

            xy_maxs_input = xy_maxs[merged_inliers_idx]
            obj_kps_input = obj_kps[merged_inliers_idx]
            proj_mats_input = proj_mats[merged_inliers_idx]
            tvecs_input = tvecs[merged_inliers_idx]

            R, t, cost = pnp_obj.pnp_obj(xy_maxs_input, obj_kps_input, \
                    proj_mats_input, tvecs_input, \
                    reprojection_error_type=reprojection_error_type, \
                    resolution_scale=resolution_scale, return_cost=True)
            if cost < best_cost:
                best_R, best_t, best_cost = R, t, cost
            else:
                break
        return best_R, best_t, best_cost

    pool_size = 24
    pool = multiprocessing.Pool(pool_size)
    best_results = pool.map(process_one_sample, [(xy_maxs, obj_kps, \
            proj_mats, tvecs, reprojection_error_type, i) for i in range(max_iter)])
    # best_results = process_one_sample((xy_maxs, obj_kps, \
    #         proj_mats, tvecs, reprojection_error_type, 0))
    pool.close()
    pool.join()
    del pool
    # best_results = process_one_sample((xy_maxs, obj_kps, proj_mats, tvecs, reprojection_error_type, 0, ))
    # best_results = [best_results]

    best_costs = [b[-1] for b in best_results]
    best_idx = np.argmin(best_costs)

    best_R, best_t, best_cost = best_results[best_idx]

    if return_cost:
        return best_R, best_t, best_cost
    else:
        return best_R, best_t


def triangulation_kp_dist(kp_dist_probs, obj_kps, proj_mats, base_sizes, tvecs, masks, sample_idx, \
        reprojection_error_type='linear', resolution_scale=1024/1440., return_cost=False):
    '''
        list of kp_dist_probs: [H, W, num_kp]
        list of obj_kps: [num_kp, 3]
        list of proj_mat: [3, 3]
        list of base_size: tuple, (x_base, y_base, dim_x, dim_y)
        reprojection_error_type: string
    '''

    xy_max_input = []
    obj_kps_input = []
    proj_mat_input = []
    tvec_input = []
    base_size_input = []
    for cam_id in range(len(kp_dist_probs)):
        input_height, input_width = kp_dist_probs[cam_id].shape[:2]
        num_kp = kp_dist_probs[cam_id].shape[-1]
        xx, yy = np.meshgrid(np.arange(input_width), np.arange(input_height))
        coordinate_map = np.stack([xx, yy], axis=-1)
        coordinate_map = np.reshape(coordinate_map, [-1, 2])
        for i in range(num_kp):
            hw = np.argmax(kp_dist_probs[cam_id][:, :, i] * masks[cam_id])
            # hw = np.argmax(kp_dist_probs[cam_id][:, :, i])
            xy = coordinate_map[hw]
            xy = xy.astype('float32')

            x_base, y_base, dim_x, dim_y = base_sizes[cam_id]
            xy[0] = xy[0] / float(input_width) * dim_x
            xy[1] = xy[1] / float(input_height) * dim_y
            xy_max_input.append(xy)

            obj_kps_input.append(obj_kps[cam_id][i])
            proj_mat_input.append(proj_mats[cam_id])
            tvec_input.append(tvecs[cam_id])
            base_size_input.append(base_sizes[cam_id])

    xy_max_input = [xy_max_input[idx] for idx in sample_idx]
    obj_kps_input = [obj_kps_input[idx] for idx in sample_idx]
    proj_mat_input = [proj_mat_input[idx] for idx in sample_idx]
    base_size_input = [base_size_input[idx] for idx in sample_idx]
    tvec_input = [tvec_input[idx] for idx in sample_idx]

    xy_max_input = np.stack(xy_max_input, axis=0)
    obj_kps_input = np.stack(obj_kps_input, axis=0)

    R, t, cost = pnp_obj.pnp_obj(xy_max_input, obj_kps_input, proj_mat_input, \
            tvecs=tvec_input, base_sizes=base_size_input, \
            reprojection_error_type=reprojection_error_type, \
            resolution_scale=resolution_scale, return_cost=True)

    if return_cost:
        return R, t, cost
    else:
        return R, t


if __name__ == '__main__':
    import skimage.transform
    import json
    import argparse

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = BASE_DIR
    sys.path.append(os.path.join(ROOT_DIR, '..', 'utils'))
    sys.path.append(os.path.join(ROOT_DIR, 'datasets'))

    import viz_tool

    parser = argparse.ArgumentParser()
    parser.add_argument('--class_oi', type=str, default='pipette_0.5_10', help='Class of interest [default: pipette_0.5_10]')
    parser.add_argument('--eval_result_file', type=str, default='test.npz', help='File name for evaluation results [default: test.npz]')
    parser.add_argument('--dump_dir', type=str, default='test.npz', help='Directory for dump results [default: test.npz]')
    FLAGS = parser.parse_args()


    DUMP_DIR = os.path.join(FLAGS.dump_dir, FLAGS.class_oi)
    if not os.path.exists(DUMP_DIR):
        os.system('mkdir -p {}'.format(DUMP_DIR))

    full_data = np.load(FLAGS.eval_result_file)['save'].item()

    xy_maxs = []
    proj_mats = []
    tvecs = []
    obj_kps_list = []

    class_id = full_data['class_id']
    obj_kps = full_data['obj_kps']
    filename = str(full_data['filename'])

    # for left_right in ['right']:
    for left_right in ['left', 'right']:

        data = full_data[left_right]

        xy_max = data['xy_max']
        proj_mat = data['proj_mat']

        num_kp = len(xy_max)

        tvec = np.zeros([3])
        tvec[0] = 0. if left_right == 'left' else \
            -abs(proj_mat[0, -1] / proj_mat[0, 0])

        proj_mat = proj_mat[:, :3]

        view = False
        if view:
            filename_split = os.path.basename(filename).split('.json')[0].split('-')
            label_filename = os.path.join( \
                    '../dataset_processing/data_processed/cam_3/after_subsampling/', \
                    filename_split[0], filename_split[1] + '_rt_label.json')
            with open(label_filename, 'r') as f:
                label_data = json.load(f)

            img_path = os.path.basename(FLAGS.eval_result_file).split('-')[:2]
            img_path = os.path.join('../dataset_processing/data_processed/cam_3/image_extraction/', \
                    img_path[0], img_path[1] + '.png')
            image_full = cv2.imread(img_path)[:, :, ::-1]

            R_label = np.array(label_data['rt']['Object 2']['R'])[0]
            t_label = np.array(label_data['rt']['Object 2']['t'])

            if left_right == 'left':
                image_show = image_full[:, :(image_full.shape[1] // 2)]
            else:
                image_show = image_full[:, (image_full.shape[1] // 2):]

            viz_tool.viz_seg(image, None, None, \
                    pred_mask, pred_kp_dist_prob[:, :, :5], \
                    center_field_label=None, subset='biolab')
            cv2.imshow('mask', (pred_mask * 255).astype('uint8'))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            continue

        xy_maxs.append(xy_max)
        proj_mats.append([proj_mat] * num_kp)
        tvecs.append([tvec] * num_kp)
        obj_kps_list.append(obj_kps[:num_kp])

    xy_maxs = np.stack(xy_maxs, axis=0).reshape([-1, 2])
    obj_kps = np.stack(obj_kps_list, axis=0).reshape([-1, 3])
    proj_mats = np.stack(proj_mats, axis=0).reshape([-1, 3, 3])
    tvecs = np.stack(tvecs, axis=0).reshape([-1, 3])

    start = time.time()
    R, t, cost = triangulation_kp_dist_ransac(xy_maxs, obj_kps, proj_mats, \
            tvecs, reprojection_error_type='linear', \
            resolution_scale=1024/1440., return_cost=True)
    end = time.time()
    print('time elapse:', end - start)
    print('cost:', cost)

    save_filename = os.path.join(DUMP_DIR, \
            os.path.basename(FLAGS.eval_result_file).split('.npz')[0] + '.json')
    with open(save_filename, 'w') as f:
        json.dump({'R': R.tolist(), 't': t.tolist(), 'cost': cost}, f, indent=4)
    # np.savez_compressed(save_filename, results={'R': R, 't': t, 'cost': cost})



