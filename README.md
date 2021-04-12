# CNLRN

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/csbhr/CNLRN/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.2.0-%237732a8)](https://pytorch.org/)

#### [Paper](https://github.com/csbhr/CNLRN) | [Discussion](https://github.com/csbhr/CNLRN/issues)
### Learning A Cascaded Non-Local Residual Network for Super-resolving Blurry Images
By [Haoran Bai](https://baihaoran.xyz/about), Songsheng Cheng, Jinhui Tang, [Jinshan Pan](https://jspan.github.io/)

## Updates
[2021-04-12] Pretrained models are available [[Here]](https://drive.google.com/drive/folders/1-JDSZvyQ8wzx5yLou4IEa6TWfQq1dymW?usp=sharing)!  
[2021-04-12] Add training code!  
[2021-04-12] Testing code is available!

## Overview

Deblurring low-resolution images is quite challenging as blur exists in the images and the resolution of the images is low. Existing deblurring methods usually require high-resolution input while the super-resolution methods usually assume that the blur is known or small. Simply applying the deblurring and super-resolution does not solve this problem well.

![top-result](https://z3.ax1x.com/2021/04/12/cBkSDx.png) 

In this work, we jointly solve the image deblurring and super-resolution in a cascaded manner and develop a non-local residual network (NLRN) as the SR module to boost the performance of blurry image SR. In addition, we develop an effective constraint based on image gradients for edge preservation and adopt the progressive upsampling mechanism to better constrain the network and reduce the training difficulty.

Both quantitative and qualitative results on the benchmarks demonstrate the effectiveness of the proposed method, and it achieves top-3 performance on the low-resolution track of the NTIRE 2021 Image Deblurring Challenge

More detailed analysis and experimental results are included in [[Paper]](https://github.com/csbhr/CNLRN).

## Dependencies

- This repository is based on [[EDVR/old_version]](https://github.com/xinntao/EDVR/tree/old_version)
- Linux (Tested on Ubuntu 18.04)
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch 1.2.0](https://pytorch.org/): `conda install pytorch=1.2.0 torchvision cudatoolkit=9.2 -c pytorch`
- numpy: `pip install numpy`
- opencv: `pip install opencv-python`
- lmdb: `pip install lmdb`
- pyyaml: `pip install pyyaml`
- tensorboardX: `pip install tensorboardX`

## Get Started

### Download
- Pretrained models and Datasets can be downloaded [[Here]](https://drive.google.com/drive/folders/1-JDSZvyQ8wzx5yLou4IEa6TWfQq1dymW?usp=sharing).
	- If you have downloaded the pretrained models，please put them to './pretrain_models'.
	- If you have downloaded the datasets，please put them to './dataset'.

### Dataset Organization Form
- If you prepare your own dataset, please follow the following form:
```
|--dataset  
    |--REDS  
        |--lmdb
            |--train_240_sharp.lmdb
            |--train_240_blur_bicubic_X4.lmdb
            |--train_240_sharp_bicubic_X4.lmdb
        |--train_240
            |--sharp
                |--000
                :
                |--239
            |--blur_bicubic_X4
            |--sharp_bicubic_X4
        |--Val300
            |--sharp
                |--000_00000009.png
                :
                |--029_00000099.png
            |--blur_bicubic_X4
            |--sharp_bicubic_X4
```
- We use LDMB to organize the training dataset for faster IO speed. Please use the following script to generate lmdb files:
```
python ./code/create_lmdb.py
```

### Training

#### Pretraining of the deblurring module
- Using the following commands:
```
cd ./code
python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 train.py -opt ./options/train/train_SRN_PreDeblur.yml --launcher pytorch
```

#### Joint training
- Putting the pretrained model of the deblurring module into './pretrain_models'.
- Using the following commands:
```
cd ./code
python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 train.py -opt ./options/train/train_CNLRN.yml --launcher pytorch
```

### Testing

- Using the following commands:
```
cd ./code
python inference_image_with_GT.py --name_flag CNLRN_Val300 --input_path ../dataset/REDS/Val300/blur_bicubic_X4 --gt_path ../dataset/REDS/Val300/sharp --save_imgs
```

## Citation
```
@InProceedings{Bai_2021_CVPRW,
  author = {Bai, Haoran and Cheng, Songsheng and Tang, Jinhui and Pan, Jinshan},
  title = {Learning A Cascaded Non-Local Residual Network for Super-resolving Blurry Images},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition WorkShops (CVPRW)},
  month = {June},
  year = {2021}
}
```
