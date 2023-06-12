from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUDA_FLAGS = []
INSTALL_REQUIREMENTS = []

ext_modules = [
    CUDAExtension('renderer.cuda.load_textures', [
        'renderer/cuda/load_textures_cuda.cpp',
        'renderer/cuda/load_textures_cuda_kernel.cu',
    ]),
    CUDAExtension('renderer.cuda.rasterize', [
        'renderer/cuda/rasterize_cuda.cpp',
        'renderer/cuda/rasterize_cuda_kernel.cu',
    ]),
    CUDAExtension('renderer.cuda.create_texture_image', [
        'renderer/cuda/create_texture_image_cuda.cpp',
        'renderer/cuda/create_texture_image_cuda_kernel.cu',
    ]),
]

setup(
    description='PyTorch implementation of "A 3D mesh renderer for networks"',
    author='Shun Iwase',
    author_email='siwase@andrew.cmu.edu',
    license='MIT License',
    version='1.1.3',
    name='renderer_pytorch',
    packages=['renderer', 'renderer.cuda'],
    install_requires=INSTALL_REQUIREMENTS,
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension})
