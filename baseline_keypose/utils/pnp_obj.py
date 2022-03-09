

import os
import sys
import numpy as np
import cv2
import scipy.optimize
import pyquaternion

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '..'))

import reprojection_error

def obj_reprejection_error(Q_t, xy_coords, obj_coords, proj_mats, tvecs, \
        reprojection_error_type='linear', resolution_scale=1024/1440., all_same_proj=True):
    Q = Q_t[:4]
    t = Q_t[4:]
    # rotation_matrix = pyquaternion.Quaternion(Q[0], Q[1], Q[2], Q[3]).rotation_matrix
    Q = Q / np.linalg.norm(Q)
    rotation_matrix = np.array([\
            [1-2*(Q[2]*Q[2]+Q[3]*Q[3]), 2*(Q[1]*Q[2]-Q[3]*Q[0]), 2*(Q[1]*Q[3]+Q[2]*Q[0])], \
            [2*(Q[1]*Q[2]+Q[3]*Q[0]), 1-2*(Q[1]*Q[1]+Q[3]*Q[3]), 2*(Q[2]*Q[3]-Q[1]*Q[0])], \
            [2*(Q[1]*Q[3]-Q[2]*Q[0]), 2*(Q[2]*Q[3]+Q[1]*Q[0]), 1-2*(Q[1]*Q[1]+Q[2]*Q[2])], \
            ])

    if all_same_proj: # acceleration by vectorization
        obj_coords_3d = np.dot(obj_coords, rotation_matrix.T) + (t + tvecs)

        proj_2d = np.dot(obj_coords_3d, proj_mats[0].T)
        proj_2d = proj_2d[:, :2] / proj_2d[:, 2, None]

        # proj_2d, _ = cv2.projectPoints(objectPoints=obj_coords_3d, rvec=np.zeros([3]), \
        #         tvec=np.zeros([3]), cameraMatrix=proj_mats[0], distCoeffs=np.zeros([4]))
        # proj_2d = proj_2d[:, 0]

        proj_2d = proj_2d * resolution_scale
        error = np.linalg.norm(proj_2d - xy_coords, axis=-1)
    else:
        obj_coords_3d = np.dot(obj_coords, rotation_matrix.T) + t

        N = len(xy_coords)
        errors = []
        for i in range(N):
            xy_coord = xy_coords[i]
            obj_coord = obj_coords_3d[i]
            proj_mat = proj_mats[i]
            tvec = tvecs[i]

            proj_2d, _ = cv2.projectPoints(objectPoints=obj_coord, rvec=np.zeros([3]), \
                    tvec=tvec, cameraMatrix=proj_mat, distCoeffs=np.zeros([4]))
            proj_2d = proj_2d * resolution_scale

            if reprojection_error_type == 'linear':
                errors.append(np.linalg.norm(proj_2d[:, 0] - xy_coord))
            elif reprojection_error_type == 'square':
                errors.append(np.square(np.linalg.norm(proj_2d[:, 0] - xy_coord)))
        error = np.array(errors)

    return error


def pnp_obj(xy_coords, obj_coords, proj_mats, tvecs, \
        reprojection_error_type='linear', resolution_scale=1024/1440., return_cost=False):
    '''
        reprojection_error_type: string
    '''
    Q_init = np.array([1., 0, 0, 0])
    t_init = np.array([0, 0, 0.5])
    Q_t_init = np.concatenate([Q_init, t_init])

    all_same_proj = True
    for i in range(1, len(proj_mats)):
        if not np.array_equal(proj_mats[0], proj_mats[i]):
            all_same_proj = False

    if all_same_proj: # acceleration by vectorization
        xy_coords = np.stack(xy_coords, axis=0)
        tvecs = np.stack(tvecs, axis=0)

    optim_result = scipy.optimize.least_squares(obj_reprejection_error, Q_t_init, \
            args=(xy_coords, obj_coords, proj_mats, tvecs, reprojection_error_type, \
            resolution_scale, all_same_proj), \
            method='lm', max_nfev=100*7*8)
            # ftol=1e-9, xtol=1e-10, gtol=1e-9)

    Q = optim_result['x'][:4]
    t = optim_result['x'][4:]

    R = pyquaternion.Quaternion(Q[0], Q[1], Q[2], Q[3]).rotation_matrix

    if return_cost:
        cost = optim_result['cost']
        cost = np.sqrt(cost*2 / len(xy_coords))
        return R, t, cost
    else:
        return R, t



