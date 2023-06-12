"""
Example 2. Drawing a mask and a depth map
"""
import os
import argparse

import torch
from PIL import Image
import numpy as np

import renderer as r

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
FARTHEST_DEPTH = 100.
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, 'data')


def get_inside_percentage(masks):
    area_all = masks.sum(axis=1).sum(axis=1)
    area_inside = masks[:, IMAGE_HEIGHT:2 * IMAGE_HEIGHT,
                        IMAGE_WIDTH:2 * IMAGE_WIDTH].sum(axis=1).sum(axis=1)
    percentage = area_inside / area_all
    return percentage


def get_front_mask(depth_maps):
    # Add 1 to make it 1-based indexing
    depth_dummy = np.ones((1, IMAGE_HEIGHT, IMAGE_WIDTH)) * FARTHEST_DEPTH
    depth_maps = np.concatenate((depth_dummy, depth_maps), axis=0)
    front_mask = np.argmin(depth_maps, axis=0)
    front_mask = front_mask.astype(np.uint8)
    return front_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--models', nargs='+', type=str, default=[])
    args = parser.parse_args()

    num_models = len(args.models)
    masks = np.zeros((num_models, 3 * IMAGE_HEIGHT, 3 * IMAGE_WIDTH))
    depth_maps = np.zeros((num_models, IMAGE_HEIGHT, IMAGE_WIDTH))

    # create renderer
    renderer = r.Renderer(image_height=IMAGE_HEIGHT,
                          image_width=IMAGE_WIDTH,
                          camera_mode='projection',
                          render_outside=True)

    for i, m in enumerate(args.models):
        model_dir = os.path.join(DATA_DIR, f'{m}.obj')
        # load .obj
        vertices, faces = r.load_obj(model_dir, load_textures=False)
        vertices = vertices[None, :, :]  # [bs, num_vertices, 3]
        faces = faces[None, :, :]  # [bs, num_faces, 3]
        # textures = textures[None, :, :]  # [bs, num_vertices, 3]
        # if an object is texture-less, please use this line
        textures = torch.ones(1, vertices.shape[1], 3,
                              dtype=torch.float32).cuda()

        # [bs, 3, 3]
        K = torch.Tensor([[[572.4114, 0., 325.2611], [0., 573.5704, 242.0490],
                           [0., 0., 1.]]]).cuda()
        # [bs, 3, 3]
        R = torch.Tensor([[[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]]).cuda()
        # [bs, 1, 3]
        t = torch.Tensor([[[0.2, 0.0, 0.4]]]).cuda()

        # Output an rgb image and mask
        _, mask, depth_map = renderer(vertices, faces, textures, K, R, t)

        # Range of an output rgb image and mask is [0, 1]
        mask = (mask[0] * 255.0).astype(np.uint8)
        depth_inside_map = depth_map[0, IMAGE_HEIGHT:2 * IMAGE_HEIGHT,
                                     IMAGE_WIDTH:2 * IMAGE_WIDTH]
        masks[i] = mask
        depth_maps[i] = depth_inside_map

        mask_inside = mask[IMAGE_HEIGHT:2*IMAGE_HEIGHT, IMAGE_WIDTH:2*IMAGE_WIDTH]
        mask_inside = Image.fromarray(mask_inside, mode='L')
        mask_inside.save(os.path.join(DATA_DIR, f'mask_{args.models[i]}.png'))

    percentages = get_inside_percentage(masks)
    print(percentages)
    front_mask = get_front_mask(depth_maps)
    front_mask = Image.fromarray(front_mask, mode='L')
    front_mask.save(os.path.join(DATA_DIR, 'front_mask.png'))


if __name__ == '__main__':
    main()
