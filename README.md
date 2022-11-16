# Reflectional and rotational symmetry detection via Radon transform

Usage:

Symmetry.exe filename flags

Flags are specified as "-char int", possible flags are:

-a number of angles for Radon transform (default: 180)
-s whether shear transform is allowed (default: 0)
-c whether to use CUDA (default: 0)
-m whether to use matrix multiplication for column-by-column comparison (default: 0)
-t problem type, 0 for reflectional symmetry, 1 for rotational symmetry (default: 0)
-d degree of rotational symmetry (default: 0, means to determine degree automatically via Fourier transform of Radon features)
-k number of angle subdivisions when searching for rotational symmetry optimal focus (default: 1)
-v whether to visualize results (default: 0)
-i whether to invert input image (default: 0)
-n number of additional checks for non-diagonal directions when searching for reflectional symmetry optimal axis without shear (default: 0)