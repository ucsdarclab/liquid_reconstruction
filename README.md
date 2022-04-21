# Image Based Reconstruction of Liquids from 2D Surface Detections
arxiv: https://arxiv.org/abs/2111.11491

## Still under construction! ##

## Environment set up: ##
Tested with Python 3.8, PyTorch 1.7.1, CUDA 11.0 and PyTorch3D 0.4.0
```
conda create --name liquid_rec python=3.8
conda activate liquid_rec
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d=0.4.0=py38_cu110_pyt171 -c pytorch3d
```
Requires SPNet (original github is: https://github.com/cschenck/SmoothParticleNets) for solving collision constraint. The following fork updated the original SPNet for PyTorch 1.7.1
```
mkdir liquid_reconstruction
cd liquid_reconstruction
gh repo clone bango123/SmoothParticleNets
cd SmoothParticleNets
python setup.py install
```
Install this repo
```
cd ..
gh repo clone ucsdarclab/liquid_reconstruction
```
