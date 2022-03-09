

import numpy as np

def orthogonal_procrustes_rotation(canonical_points, observed_points):
    '''
        canonical_points: (N, 3) array
        observed_points: (N, 3) array
    '''
    M_matrix = np.dot(observed_points.T, canonical_points)
    U, S, Vh = np.linalg.svd(M_matrix)
    R = np.dot(U, Vh)

    if np.linalg.det(R) < 0:

        S_mod = np.eye(3)
        S_mod[-1, -1] = -1
        R = np.dot(np.dot(U, S_mod), Vh)

    return R

def orthogonal_procrustes(canonical_points, observed_points):
    '''
        canonical_points: (N, 3) array
        observed_points: (N, 3) array
    '''
    canonical_points_mean = np.mean(canonical_points, axis=0)
    observed_points_mean = np.mean(observed_points, axis=0)

    observed_points_ = observed_points - observed_points_mean
    canonical_points_ = canonical_points - canonical_points_mean
    R = orthogonal_procrustes_rotation(canonical_points_, observed_points_)
    t = observed_points_mean - np.dot(R, canonical_points_mean.T)

    mean_error = np.mean(np.linalg.norm(observed_points - (np.dot(R, canonical_points.T).T + t), axis=-1))

    return R, t, mean_error

def batch_orthogonal_procrustes_rotation(canonical_points, observed_points, num_valid):
    '''
        canonical_points: (B, N, 3) array
        observed_points: (B, N, 3) array
        num_valid: (B, ) array
    '''

    M_matrix = np.matmul(np.transpose(observed_points, [0, 2, 1]), canonical_points)
    print(M_matrix)
    U, S, Vh = np.linalg.svd(M_matrix)
    R = np.matmul(U, Vh)

    if np.linalg.det(R) < 0:

        S_mod = np.eye(3)
        S_mod[-1, -1] = -1
        R = np.dot(np.dot(U, S_mod), Vh)

    return R

def batch_orthogonal_procrustes(canonical_points, observed_points):
    '''
        canonical_points: (B, N, 3) array
        observed_points: (B, N, 3) array
    '''
    canonical_points_mean = np.mean(canonical_points, axis=1, keepdims=True)
    observed_points_mean = np.mean(observed_points, axis=1, keepdims=True)

    observed_points_ = observed_points - observed_points_mean
    canonical_points_ = canonical_points - canonical_points_mean
    R = orthogonal_procrustes_rotation(canonical_points_, observed_points_)
    t = observed_points_mean - np.dot(R, canonical_points_mean.T)

    mean_error = np.mean(np.linalg.norm(observed_points - (np.dot(R, canonical_points.T).T + t), axis=-1))

    return R, t, mean_error


if __name__ == '__main__':
    from pyquaternion import Quaternion

    A = np.array([[0, 0, 0], \
                  [0, 1., 0], \
                  [1., 0, 0], \
                  [1., 1., 0], \
                  ])

    rot_axis = np.random.uniform(-1, 1, size=(3,))
    rot_axis = rot_axis / np.linalg.norm(rot_axis)
    angle = np.random.uniform(0, 2*np.pi)

    rot_matrix = Quaternion(axis=rot_axis, angle=angle).rotation_matrix

    B = np.dot(rot_matrix, A.T).T

    R_ = orthogonal_procrustes_rotation(A, B)
    print(rot_matrix)
    print(R_)


    A -= A.mean(axis=0) + 0.02

    translation = np.random.uniform(-1, 1, size=(3,))
    B = np.dot(rot_matrix, A.T).T + translation
    R_, t_, error = orthogonal_procrustes(A, B)
    print(rot_matrix)
    print(R_)
    print(translation)
    print(t_)
    print(error)




