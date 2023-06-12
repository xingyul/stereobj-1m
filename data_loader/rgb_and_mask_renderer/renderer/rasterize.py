import torch
import torch.nn as nn
from torch.autograd import Function

import renderer.cuda.rasterize as rasterize_cuda

DEFAULT_IMAGE_SIZE = 256
DEFAULT_NEAR = 0.1
DEFAULT_FAR = 100
DEFAULT_BACKGROUND_COLOR = (0, 0, 0)


class RasterizeFunction(Function):
    '''
    Definition of differentiable rasterize operation
    Some parts of the code are implemented in CUDA
    Currently implemented only for cuda Tensors
    '''
    @staticmethod
    def forward(ctx, faces, textures, image_height, image_width, near, far,
                background_color):
        '''
        Forward pass
        '''

        ctx.image_height = image_height
        ctx.image_width = image_width
        ctx.near = near
        ctx.far = far
        ctx.background_color = background_color

        ctx.device = faces.device
        ctx.num_faces = faces.shape[1]

        rgb_map, mask_map, depth_map = \
            rasterize_cuda.forward(faces, textures, ctx.image_height,
                                   ctx.image_width, ctx.near, ctx.far)

        return rgb_map, mask_map, depth_map

    @staticmethod
    def backward():
        pass


class Rasterize(nn.Module):
    '''
    Wrapper around the autograd function RasterizeFunction
    Currently implemented only for cuda Tensors
    '''
    def __init__(self, image_height, image_width, near, far, background_color):
        super(Rasterize, self).__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.near = near
        self.far = far
        self.background_color = background_color

    def forward(self, faces, textures):
        if faces.device == "cpu" or (textures is not None
                                     and textures.device == "cpu"):
            raise TypeError('Rasterize module supports only cuda Tensors')
        return RasterizeFunction.apply(faces, textures, self.image_height,
                                       self.image_width, self.near, self.far,
                                       self.background_color)


def rasterize(
    faces,
    textures=None,
    image_height=DEFAULT_IMAGE_SIZE,
    image_width=DEFAULT_IMAGE_SIZE,
    near=DEFAULT_NEAR,
    far=DEFAULT_FAR,
    background_color=DEFAULT_BACKGROUND_COLOR,
):
    """
    Rasterize multi-channel features from faces and textures (features).

    Args:
        faces (torch.Tensor): Faces. The shape is [batch size, number of faces, 3 (vertices), 3 (XYZ)].
        textures (torch.Tensor): Textures.
            The shape is [batch size, number of faces, texture size, 3 (RGB)].
        image_height (int): Height of rendered images.
        image_width (int): Width of rendered images.
        near (float): nearest z-coordinate to draw.
        far (float): farthest z-coordinate to draw.
        background_color (tuple): background color of RGB images.

    Returns:
        rgb_image, mask

    """

    rgb, mask, depth = Rasterize(image_height, image_width, near, far,
                                 background_color)(faces, textures)

    # transpose & vertical flip
    # rgb = rgb.permute((0, 3, 1, 2))

    return rgb, mask, depth
