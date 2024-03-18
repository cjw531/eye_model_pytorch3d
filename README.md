# eye_model_pytorch3d

## PyTorch3D Dependency Installation
### 1. Create Conda Environmenmt
```
conda create -n eye_model python=3.8
conda activate eye_model
```
### 2. CUDA Toolkit Check
```
nvcc --version

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2020 NVIDIA Corporation
Built on Mon_Nov_30_19:08:53_PST_2020
Cuda compilation tools, release 11.2, V11.2.67
Build cuda_11.2.r11.2/compiler.29373293_0
```
Cuda Toolkit version is 11.2

### 3. Install Runtime Dependencies
```
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
```

### 4. For CUDA < 11.7, Install CUB
```
conda install -c bottler nvidiacub
```

### 5. Tests/Linting and Demos
```
conda install jupyter
pip install scikit-image matplotlib imageio plotly opencv-python
pip install black usort flake8 flake8-bugbear flake8-comprehensions
```

### 6. Install with CUDA support from Anaconda Cloud
```
conda install pytorch3d -c pytorch3d
```

### 7. Install PyTorch3D from GitHub (ver. 0.2.0)
```
FORCE_CUDA=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.2.0"
```
