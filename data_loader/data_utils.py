import pycocotools.mask as mask_utils
import numpy as np
from PIL import Image


def compute_prob(mask, kpt_2d, gaussian_sigma=10.):

    h, w = mask.shape[:2]
    k = kpt_2d.shape[0]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    coordinate_map = np.stack([xx, yy], axis=-1)

    eps = 1e-6

    uv_diff = np.expand_dims(coordinate_map, -2) - kpt_2d
    uv_dist = np.linalg.norm(uv_diff, axis=-1)

    uv_prob = np.exp(-np.square(uv_dist) / gaussian_sigma)

    return uv_prob


