# Reproduce instructions

* First please modify the directions to the dataset and pretrained models in the config ymls. 

* Baseline models

```shell
python train_baseline.py --config=configs/baseline.yml MODEL.NAME "resnet50_ibn_a" OUTPUT_DIR "./log/baseline_a"

python train_baseline.py --config=configs/baseline.yml MODEL.NAME "resnet50_ibn_b" OUTPUT_DIR "./log/baseline_b"

python train_baseline.py --config=configs/baseline.yml MODEL.NAME "resnet101_ibn_a" OUTPUT_DIR "./log/baseline_101"

python train_baseline.py --config=configs/baseline.yml MODEL.NAME "hrnetv2_w18" OUTPUT_DIR "./log/baseline_hr"
```

* Camera models

```shell
python train_camera.py --config=configs/camera.yml MODEL.NAME "resnet101" OUTPUT_DIR "./log/camera_101"

python train_camera.py --config=configs/camera.yml MODEL.NAME "resnet152" OUTPUT_DIR "./log/camera_152"

python train_camera.py --config=configs/camera.yml MODEL.NAME "resnet101_ibn_a" OUTPUT_DIR "./log/camera_101_a"

python train_camera.py --config=configs/camera.yml MODEL.NAME "hrnetv2_w18" OUTPUT_DIR "./log/camera_hr"
```

* Adaptation

Please select the best camera models for the following training. In our experience, the accuracy on validation set should be above 97%. Of course, you can directly use the camera model trained by us: [Google Drive](https://drive.google.com/file/d/1tuJZw1DnTQ5B95voUL8bE1akiyrqeK-E/view?usp=sharing). 

```shell
python train_adaptation.py --config=configs/adaptation.yml MODEL.NAME "resnet50_ibn_a" OUTPUT_DIR "./log/adaptation_a" \
MODEL.PRETRAIN_PATH "./log/baseline_a/resnet50_ibn_a_model_60.pth" \
TEST.CAMERA_WEIGHT "./log/camera_101_a/best.pth"

python train_adaptation.py --config=configs/adaptation.yml MODEL.NAME "resnet50_ibn_b" OUTPUT_DIR "./log/adaptation_b" \
MODEL.PRETRAIN_PATH "./log/baseline_b/resnet50_ibn_b_model_60.pth" \
TEST.CAMERA_WEIGHT "./log/camera_101_a/best.pth"

python train_adaptation.py --config=configs/adaptation.yml MODEL.NAME "resnet101_ibn_a" OUTPUT_DIR "./log/adaptation_101" \
MODEL.PRETRAIN_PATH "./log/baseline_101/resnet101_ibn_a_model_60.pth" \
TEST.CAMERA_WEIGHT "./log/camera_101_a/best.pth"

python train_adaptation.py --config=configs/adaptation.yml MODEL.NAME "hrnetv2_w18" OUTPUT_DIR "./log/adaptation_hr" \
MODEL.PRETRAIN_PATH "./log/baseline_hr/hrnetv2_w18_model_60.pth" \
TEST.CAMERA_WEIGHT "./log/camera_101_a/best.pth"
```

* Finetune

Please select the best adaptation model for finetuning. 

```shell
python train_adaptation.py --config=configs/finetune.yml MODEL.NAME "resnet50_ibn_a" OUTPUT_DIR "./log/finetune_a" \
MODEL.PRETRAIN_PATH "./log/adaptation_a/resnet50_ibn_a_model_best.pth" \
TEST.CAMERA_WEIGHT "./log/camera_101_a/best.pth"

python train_adaptation.py --config=configs/finetune.yml MODEL.NAME "resnet50_ibn_b" OUTPUT_DIR "./log/finetune_b" \
MODEL.PRETRAIN_PATH "./log/adaptation_b/resnet50_ibn_b_model_best.pth" \
TEST.CAMERA_WEIGHT "./log/camera_101_a/best.pth"

python train_adaptation.py --config=configs/finetune.yml MODEL.NAME "resnet101_ibn_a" OUTPUT_DIR "./log/finetune_101" \
MODEL.PRETRAIN_PATH "./log/adaptation_101/resnet101_ibn_a_model_best.pth" \
TEST.CAMERA_WEIGHT "./log/camera_101_a/best.pth"

python train_adaptation.py --config=configs/finetune.yml MODEL.NAME "hrnetv2_w18" OUTPUT_DIR "./log/finetune_hr" \
MODEL.PRETRAIN_PATH "./log/adaptation_hr/hrnetv2_w18_model_best.pth" \
TEST.CAMERA_WEIGHT "./log/camera_101_a/best.pth"
```

* Train models with large image size

```shell
python train_adaptation.py --config=configs/finetune.yml MODEL.NAME "resnet50_ibn_a" OUTPUT_DIR "./log/finetune_a_large" \
MODEL.PRETRAIN_PATH "./log/adaptation_a/resnet50_ibn_a_model_best.pth" \
TEST.CAMERA_WEIGHT "./log/camera_101_a/best.pth" \
INPUT.SIZE_TRAIN "[384, 192]" INPUT.SIZE_TEST "[384, 192]"

python train_adaptation.py --config=configs/finetune.yml MODEL.NAME "resnet50_ibn_b" OUTPUT_DIR "./log/finetune_b_large" \
MODEL.PRETRAIN_PATH "./log/adaptation_b/resnet50_ibn_b_model_best.pth" \
TEST.CAMERA_WEIGHT "./log/camera_101_a/best.pth" \
INPUT.SIZE_TRAIN "[384, 192]" INPUT.SIZE_TEST "[384, 192]"

python train_adaptation.py --config=configs/finetune.yml MODEL.NAME "resnet101_ibn_a" OUTPUT_DIR "./log/finetune_101_large" \
MODEL.PRETRAIN_PATH "./log/adaptation_101/resnet101_ibn_a_model_best.pth" \
TEST.CAMERA_WEIGHT "./log/camera_101_a/best.pth" \
INPUT.SIZE_TRAIN "[384, 192]" INPUT.SIZE_TEST "[384, 192]"

python train_adaptation.py --config=configs/finetune.yml MODEL.NAME "hrnetv2_w18" OUTPUT_DIR "./log/finetune_hr_large" \
MODEL.PRETRAIN_PATH "./log/adaptation_hr/hrnetv2_w18_model_best.pth" \
TEST.CAMERA_WEIGHT "./log/camera_101_a/best.pth" \
INPUT.SIZE_TRAIN "[384, 192]" INPUT.SIZE_TEST "[384, 192]"
```

* Test

First execute forward pass for every model. In our experience, the model with an around 78% validation score would perform best on test set. 

```shell
python test.py --config=configs/test.yml MODEL.NAME "resnet50_ibn_a" OUTPUT_DIR "./log/test_a" \
TEST.WEIGHT "./log/finetune_a/resnet50_ibn_a_model_best.pth" \
TEST.CAMERA_WEIGHT "./log/camera_101_a/best.pth"

python test.py --config=configs/test.yml MODEL.NAME "resnet50_ibn_b" OUTPUT_DIR "./log/test_b" \
TEST.WEIGHT "./log/finetune_b/resnet50_ibn_b_model_best.pth" \
TEST.CAMERA_WEIGHT "./log/camera_101_a/best.pth"

python test.py --config=configs/test.yml MODEL.NAME "resnet101_ibn_a" OUTPUT_DIR "./log/test_101" \
TEST.WEIGHT "./log/finetune_101/resnet101_ibn_a_model_best.pth" \
TEST.CAMERA_WEIGHT "./log/camera_101_a/best.pth"

python test.py --config=configs/test.yml MODEL.NAME "hrnetv2_w18" OUTPUT_DIR "./log/test_hr" \
TEST.WEIGHT "./log/finetune_hr/hrnetv2_w18_model_best.pth" \
TEST.CAMERA_WEIGHT "./log/camera_101_a/best.pth"

python test.py --config=configs/test.yml MODEL.NAME "resnet50_ibn_a" OUTPUT_DIR "./log/test_a_large" \
TEST.WEIGHT "./log/finetune_a_large/resnet50_ibn_a_model_best.pth" \
TEST.CAMERA_WEIGHT "./log/camera_101_a/best.pth" \
INPUT.SIZE_TRAIN "[384, 192]" INPUT.SIZE_TEST "[384, 192]"

python test.py --config=configs/test.yml MODEL.NAME "resnet50_ibn_b" OUTPUT_DIR "./log/test_b_large" \
TEST.WEIGHT "./log/finetune_b_large/resnet50_ibn_b_model_best.pth" \
TEST.CAMERA_WEIGHT "./log/camera_101_a/best.pth" \
INPUT.SIZE_TRAIN "[384, 192]" INPUT.SIZE_TEST "[384, 192]"

python test.py --config=configs/test.yml MODEL.NAME "resnet101_ibn_a" OUTPUT_DIR "./log/test_101_large" \
TEST.WEIGHT "./log/finetune_101_large/resnet101_ibn_a_model_best.pth" \
TEST.CAMERA_WEIGHT "./log/camera_101_a/best.pth" \
INPUT.SIZE_TRAIN "[384, 192]" INPUT.SIZE_TEST "[384, 192]"

python test.py --config=configs/test.yml MODEL.NAME "hrnetv2_w18" OUTPUT_DIR "./log/test_hr_large" \
TEST.WEIGHT "./log/finetune_hr_large/hrnetv2_w18_model_best.pth" \
TEST.CAMERA_WEIGHT "./log/camera_101_a/best.pth" \
INPUT.SIZE_TRAIN "[384, 192]" INPUT.SIZE_TEST "[384, 192]"
```

Then get distmats from camera models. 
```shell
python test.py --config=configs/test.yml MODEL.NAME "resnet50_ibn_a" OUTPUT_DIR "./log/test_camera_101" \
TEST.WEIGHT "./log/finetune_a/resnet50_ibn_a_model_best.pth" \
TEST.CAMERA_WEIGHT "./log/camera_101/best.pth"

python test.py --config=configs/test.yml MODEL.NAME "resnet50_ibn_a" OUTPUT_DIR "./log/test_camera_152" \
TEST.WEIGHT "./log/finetune_a/resnet50_ibn_a_model_best.pth" \
TEST.CAMERA_WEIGHT "./log/camera_152/best.pth"

python test.py --config=configs/test.yml MODEL.NAME "resnet50_ibn_a" OUTPUT_DIR "./log/test_camera_101_a" \
TEST.WEIGHT "./log/finetune_a/resnet50_ibn_a_model_best.pth" \
TEST.CAMERA_WEIGHT "./log/camera_101_a/best.pth"

python test.py --config=configs/test.yml MODEL.NAME "resnet50_ibn_a" OUTPUT_DIR "./log/test_camera_hr" \
TEST.WEIGHT "./log/finetune_a/resnet50_ibn_a_model_best.pth" \
TEST.CAMERA_WEIGHT "./log/camera_hr/best.pth"
```

In the end, run model ensemble. 
```shell
python utils/ensemble.py
```