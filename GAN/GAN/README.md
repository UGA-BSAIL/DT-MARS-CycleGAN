# Pytorch-CycleGAN

## 1.Data

The simulation and real data is availible at [kaggle](https://www.kaggle.com/datasets/zhengkunli3969/dtmars-cyclegan).


## 2. Train Detection Model for Spatial Consistency GAN Loss

   `python Detector/train.py`

## 3. Train on Multi-GPUs

1. Train DT-CycleGAN

   `sh scripts/dtgan.sh`

2. Train CycleGAN

   `sh scripts/cyclegan.sh`

3. Train RetinaGAN

   `sh scripts/retinagan.sh`


## 4. GAN Data Augment

The sim2real models play roles as a data augment methods. Therefor, we need generate the augment images and train with crop detection model (YOLOv8).

   `sh scripts/gen_img.sh`

## 5.Crop Detection

Fine-tune the YOLOv8 trained on sim-only with the sim2real transferred annotated data. 