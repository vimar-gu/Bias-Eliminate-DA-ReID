# Bias Eliminate Domain Adaptive Pedestrian Re-identification

This repo contains our code for [VisDA2020](http://ai.bu.edu/visda-2020) challenge at ECCV workshop. 

## Introduction

This work mainly solve the domain adaptive pedestrian re-identification problem by eliminishing the bias from inter-domain gap and intra-domain camera difference. 

This project is mainly based on [reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline).

## Get Started

1. Clone the repo `git clone https://github.com/vimar-gu/Bias-Eliminate-DA-ReID.git`
2. Install dependencies:
* pytorch >= 1.0.0
* python >= 3.5
* torchvision
* yacs
3. Prepare dataset. We modified the file names in order to read all datasets through one api. You can download the modified version in [here](https://drive.google.com/file/d/1n0UTKs4dq47bpYYHIh6BH1kV5jYdebId/view?usp=sharing). In addition to the original data, we also added CamStyle data to better train the model. 
4. We use [ResNet-ibn](https://github.com/XingangPan/IBN-Net) and [HRNet](https://github.com/HRNet/HRNet-Image-Classification) as backbones. ImageNet pretrained models can be downloaded in [here](https://drive.google.com/drive/folders/1thS2B8UOSBi_cJX6zRy6YYRwz_nVFI_S) and [here](https://onedrive.live.com/?authkey=%21AMkPimlmClRvmpw&cid=F7FD0B7F26543CEB&id=F7FD0B7F26543CEB%21112&parId=F7FD0B7F26543CEB%21105&o=OneUp). 

## Run
If you want to reproduce our results, please refer to [[VisDA.md]](https://github.com/vimar-gu/Bias-Eliminate-DA-ReID/blob/master/VisDA.md)

## Results
The performance on VisDA2020 validation dataset

| Method | mAP | Rank-1 | Rank-5 | Rank-10 |
|  ---   | --- |   ---  |   ---  |   ---   |
| Basline | 30.7 | 59.7 | 77.5 | 83.3 |
| + Domain Adaptation | 44.9 | 75.3 | 86.7 | 91.0 |
| + Finetuning | 48.6 | 79.8 | 88.3 | 91.5 |
| + Post Processing | 70.9 | 86.5 | 92.8 | 94.4 |

## Trained models
The models can be downloaded from:

* ResNet50-ibn-a: [Google Drive](https://drive.google.com/file/d/1ejLJk7sJOWhMD6zwQDWmhzFsli0dcSim/view?usp=sharing)
* ResNet101-ibn-a: [Google Drive](https://drive.google.com/file/d/1AM_xjiu68iaquT0qMpo8TyauuxKj91sh/view?usp=sharing)
* ResNet50-ibn-b: [Google Drive](https://drive.google.com/file/d/1w3NITiq4fnmijynAcJM6J-JcqWspscpI/view?usp=sharing)
* HRNetv2-w18: [Google Drive](https://drive.google.com/file/d/1uiryXdhsH8X4MCIDBafEO9qM7dMekLQS/view?usp=sharing)
* ResNet50-ibn-a-large: [Google Drive](https://drive.google.com/file/d/1mVQeamQGUgSuIr8Y73DGNe1H6GPKjuAo/view?usp=sharing)
* ResNet101-ibn-a-large: [Google Drive](https://drive.google.com/file/d/1jlwwIIGIUwzSaGwP9mc77gMnteTviPiG/view?usp=sharing)
* ResNet50-ibn-b-large: [Google Drive](https://drive.google.com/file/d/1oseEqEPKDx6-1b0h0RyNxR4yKV-t3-Z2/view?usp=sharing)
* HRNetv2-w18-large: [Google Drive](https://drive.google.com/file/d/11_npph5csVOSmn6RL5g_3JthqCEQw3ga/view?usp=sharing)

The camera models can be downloaded from:

* Camera(ResNet101): [Google Drive](https://drive.google.com/file/d/1E-n2iOVwq-3PGv1CUxpEDzjP8Uchn7rR/view?usp=sharing)
* Camera(ResNet152): [Google Drive](https://drive.google.com/file/d/1WLBrxiIWj3FmidCh2notX71nMvQoujUT/view?usp=sharing)
* Camera(ResNet101-ibn-a): [Google Drive](https://drive.google.com/file/d/1tuJZw1DnTQ5B95voUL8bE1akiyrqeK-E/view?usp=sharing)
* Camera(HRNetv2-w18): [Google Drive](https://drive.google.com/file/d/1eC6sqKkefrpl1Bq-2lQ_8ScQTe51jl_K/view?usp=sharing)

### Some tips
* By our experience, there can be a large fluctuation of validation scores which are not completely positive correlated to the scores on testing set. 
* We have fixed the random seed in the updates. But there might still be some difference due to environment. 
* Multiple camera models in the testing phase may boost the performance by a little bit. 
