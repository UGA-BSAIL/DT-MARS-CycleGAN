#!/bin/bash

python data_gan.py \
    --img_outfolder 'sim2real/dtgan_pretrainDet' \
    --generator_A2B '/home/myid/zw63397/Projects/Crop_Detect/DT/GAN/output/dtgan_pretrainDet/netG_dt_A2B_withdet_200.pth' \
    --generator_B2A '/home/myid/zw63397/Projects/Crop_Detect/DT/GAN/output/dtgan_pretrainDet/netG_dt_B2A_withdet_200.pth' \
    --cuda

# output/retina/netG_retina_B2A_withdet_200.pth'
# output/dtgan/netG_dt_B2A_withdet_200.pth
# output/dtgan_pretrainDet/netG_dt_B2A_withdet_200.pth