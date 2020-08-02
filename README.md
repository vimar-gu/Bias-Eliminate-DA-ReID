# Bias Eliminate Domain Adaptive Pedestrain Re-identification

This repo contains our code for [VisDA2020](http://ai.bu.edu/visda-2020) challenge at ECCV workshop. 

## Introduction

This work mainly solve the domain adaptive pedestrain re-identification problem by eliminishing the bias from inter-domain gap and intra-domain camera difference. 

This project is mainly based on [reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline).

## Get Started

1. Clone the repo `git clone https://github.com/vimar-gu/`
2. Install dependencies:
* pytorch >= 1.0.0
* python >= 3.5
* torchvision
* yacs
3. Prepare dataset. We modified the file names in order to read all datasets through one api. You can download the modified version in [here](https://drive.google.com/file/d/1n0UTKs4dq47bpYYHIh6BH1kV5jYdebId/view?usp=sharing). In addition to the original data, we also added CamStyle data to better train the model. 
4. We use [ResNet-ibn](https://github.com/XingangPan/IBN-Net) and [HRNet](https://github.com/HRNet/HRNet-Image-Classification) as backbones. ImageNet pretrained models can be downloaded in [here](https://drive.google.com/drive/folders/1thS2B8UOSBi_cJX6zRy6YYRwz_nVFI_S) and [here](https://onedrive.live.com/?authkey=%21AMkPimlmClRvmpw&cid=F7FD0B7F26543CEB&id=F7FD0B7F26543CEB%21112&parId=F7FD0B7F26543CEB%21105&o=OneUp). 

## Run

1. Modify the path to your datasets and pretrained models in configs.
2. Train the baseline model.
```
python train_baseline.py --config=configs/baseline.yml
```
3. Train domain adaptation.
```
python train_adaptation.py --config=configs/adaptation.yml
```
4. Train the camera model. 
```
python train_camera.py --config=configs/camera.yml
```
5. Finetune the model.
```
python train_adaptation.py --config=configs/finetune.yml
```
6. Validate and test.
```
python validate.py --config=configs/finetune.yml
python test.py --config=configs/finetune.yml
```
7. Model ensemble.
To test multiple model ensemble, put the names of distmats into `utils/ensemble.py` and run it. 

## Results
The performance on VisDA2020 validation dataset

| Method | mAP | Rank-1 | Rank-5 | Rank-10 |
|  ---   | --- |   ---  |   ---  |   ---   |
| Basline | 30.7 | 59.7 | 77.5 | 83.3 |
| + Domain Adaptation | 43.2 | 72.1 | 84.4 | 89.9 |
| + Finetuning | 46.3 | 76.1 | 86.5 | 91.0 |
| + Post Processing | 69.3 | 86.5 | 90.5 | 91.8 |
