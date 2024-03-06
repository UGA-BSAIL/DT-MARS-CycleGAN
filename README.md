# DT/MARS-CycleGAN
The repository is for the paper: [Digital Twin/MARS-CycleGAN: Improving Crop Row Detection for MARS Phenotyping Robot with Simulated and Generative Images](https://arxiv.org/abs/2310.12787), including the code and dataset for reproducing. 

The paper is under revision on Journal of Field Robotics.

## Pipeline of DT/MARS-CycleGAN
<img src="figures/fig1.png"/>
Fig. 1: Illustration of the physical/DT robots (left panel), the DT/MARS-CycleGAN model (middle panel), and the crop object/row detection network (right panel).


<p align="center">
  <img src="figures/fig2.png" alt="Fig. 2: Simulation-to-Real synthetic images generated by different models." style="width: 70%;">
</p>
<p align="center"><i>Fig. 2: Simulation-to-Real synthetic images generated by different models.</i></p>


<p align="center">
  <img src="figures/fig5.png" alt="Fig. 3. Zero-shot crop detection on [Sugar Beets Dataset from University of Bonn](https://www.ipb.uni-bonn.de/data/sugarbeets2016/index.html) with more diverse situation, including different growing crops and growing stage, soil background, illumination, and shadow, weeds, and occlusion." style="width: 70%;">
</p>
<p align="center"><i>Fig. 3. Zero-shot crop detection on <a href="https://www.ipb.uni-bonn.de/data/sugarbeets2016/index.html">Sugar Beets Dataset from University of Bonn</a> with more diverse situation, including different growing crops and growing stage, soil background, illumination, and shadow, weeds, and occlusion.</i></p>



## Prerequisites

[YOLOv8](https://github.com/ultralytics/ultralytics) repository for crop detection: 
```
 pip install ultralytics
 pip install pybullet torch accelerate timm
```


## Getting Started
See [GANs](GAN/GAN)


## Dataset Download
The dataset has been released on [Kaggle](https://www.kaggle.com/datasets/zhengkunli3969/dtmars-cyclegan). Also, you can download on [Google Drive](https://drive.google.com/drive/folders/12pb47Zl1j285z5AXG8F77oASkfYqSbA0?usp=sharing).



## Pretrained models
The pre-trained models are available at [weight](weight). All the weights are YOLOv8n trained with different synthesized data.  
    - best.pt: trained with images generated from DT/MARs-CycleGAN (ours).  
    - CYC.pt: trained with  images generated from original CycleGAN.  
    - RetinaGAN.pt: trained with images generated from RetinaGAN.  
    - pure_sim.pt: trained with images generated from Publlet.  

## References
If you find this work or code useful, please cite:

```
@article{liu2023dt,
  title={Dt/mars-cyclegan: Improved object detection for mars phenotyping robot},
  author={Liu, David and Li, Zhengkun and Wu, Zihao and Li, Changying},
  journal={arXiv preprint arXiv:2310.12787},
  year={2023}
}
```
