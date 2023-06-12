"""
Example 1. Drawing a cat rgb image and mask
"""
import os
import argparse

import torch
from PIL import Image
import numpy as np

import renderer as r

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--filename_input',
                        type=str,
                        default=os.path.join(data_dir, 'cat.obj'))
    args = parser.parse_args()

    # load .obj
    vertices, faces, textures = r.load_obj(args.filename_input,
                                           load_textures=True)
    vertices = vertices[None, :, :]  # [bs, num_vertices, 3]
    faces = faces[None, :, :]  # [bs, num_faces, 3]
    textures = textures[None, :, :]  # [bs, num_vertices, 3]
    # if an object is texture-less, please use this line
    # textures = torch.ones(1, vertices.shape[1],
    #                       3, dtype=torch.float32).cuda()

    # create renderer
    renderer = r.Renderer(image_height=480,
                          image_width=640,
                          camera_mode='projection')
    # [bs, 3, 3]
    K = torch.Tensor([[[572.4114, 0., 325.2611], [0., 573.5704, 242.0490],
                       [0., 0., 1.]]]).cuda()
    # [bs, 3, 3]
    R = torch.Tensor([[[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]]).cuda()
    # [bs, 1, 3]
    t = torch.Tensor([[[0.323, 0.0, 0.4]]]).cuda()

    # Output an rgb image and mask
    rgb, mask, depth = renderer(vertices, faces, textures, K, R, t)

    # Range of an output rgb image and mask is [0, 1]
    rgb = (rgb[0] * 255.0).astype(np.uint8)
    mask = (mask[0] * 255.0).astype(np.uint8)
    depth = (depth[0] * 255.0).astype(np.uint8)

    rgb = Image.fromarray(rgb, mode='RGB')
    mask = Image.fromarray(mask, mode='L')
    depth = Image.fromarray(depth, mode='L')
    rgb.save(os.path.join(data_dir, 'rgb.png'))
    mask.save(os.path.join(data_dir, 'mask.png'))
    depth.save(os.path.join(data_dir, 'depth.png'))


if __name__ == '__main__':
    main()
