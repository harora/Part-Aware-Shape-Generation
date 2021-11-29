## Dependencies
Requirements:
- Download binvox.exe to data_preparation: https://www.patrickmin.com/binvox/ 
- Python 3.x with numpy, opencv-python and cython

Build Cython module:
```
python setup.py build_ext --inplace
```

## Usage:
Convert PartNet meshes to voxels
```
python conv2voxels.py /path/to/partnet_dataset
```

## More Details:
https://github.com/czq142857/DECOR-GAN/tree/main/data_preparationxs