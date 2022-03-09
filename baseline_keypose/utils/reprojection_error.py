


import cv2
import numpy as np

def reprojection_error(vec_3d, vecs_2d, rvecs, tvecs, cam_mtxs, dist=None, cost='linear'):
    '''
        vec_3d: (N, 3) array
        vec_2d: (N, 2) array
        rvec: 3 array
    '''
    vec_3d = np.expand_dims(vec_3d, -1)
    num_cams = len(rvecs)
    errors = []
    for i in range(num_cams):
        vec_2d_proj, _ = cv2.projectPoints(objectPoints=vec_3d, rvec=rvecs[i], tvec=tvecs[i], cameraMatrix=cam_mtxs[i], distCoeffs=dist[i])
        if cost == 'linear':
            errors.append(np.linalg.norm(vec_2d_proj[:, 0] - vecs_2d[i]))
        elif cost == 'square':
            errors.append(np.square(np.linalg.norm(vec_2d_proj[:, 0] - vecs_2d[i])))
    return np.array(errors)



