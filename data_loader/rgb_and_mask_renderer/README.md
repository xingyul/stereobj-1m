# RGB and Mask Renderer

## Requirements
- Python 3.6+
- PyTorch 1.5.1
- CUDA 10.x

## Installation
You can install the package by running
```
python setup.py install
```

## Running examples

### Render a mask, an RGB, and depth images for a single 3D model

```
python ./examples/example1.py
```

### Render an occlusion-aware mask image and calculate the ratio of its inside area


```
python ./examples/example2.py -m cat ape driller
```
Please put obj files under `./examples/data` directory before running the script.
The occlusion-aware mask contains all objects' masks and each mask has an 1-based unique id. (cat: 1, ape: 2, driller: 3)
If some objects exist in the same pixel the frontmost object's id will be given to the pixel. 
The script also returns the ratio of the inside area for each object in the order of given parameters.
