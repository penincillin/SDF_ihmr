## Implementation of the SDF Loss

We provide the implementation of Signed Distance Functinon (SDF) to detect collisions between 3D interacting hands.

## Installation instructions

You can install the package by running
```
python setup.py develop
```

# Acknowledgements
The code is adapted from origin [SDF](https://github.com/JiangWenPL/multiperson/tree/master/sdf) implemented by [Wen Jiang](https://github.com/JiangWenPL).

If you find this code useful for your research, please consider citing the following paper:
```
@InProceedings{rong2021ihmr,
    author = {Rong, Yu and Wang, Jingbo and Liu, Ziwei and Loy, Chen Change},
    title = {Monocular 3D Reconstruction of Interacting Handsvia Collision-Aware Factorized Refinements},
    booktitle = {3DV},
    year = {2021}
}

@Inproceedings{jiang2020mpshape,
    author = {Jiang, Wen and Kolotouros, Nikos and Pavlakos, Georgios and Zhou, Xiaowei and Daniilidis, Kostas},
    title = {Coherent Reconstruction of Multiple Humans from a Single Image},
    booktitle = {CVPR},
    year = {2020}
}
```
