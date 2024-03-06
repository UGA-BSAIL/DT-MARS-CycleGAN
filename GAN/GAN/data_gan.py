#!/usr/bin/python3

import argparse
import sys
import os
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from models import Generator
from datasets import ImageDataset, CropDataset

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='/home/myid/zw63397/Projects/Crop_Detect/data', help='root directory of the dataset')
parser.add_argument('--img_outfolder', type=str, default='sim2real/cycle', help='output img dir')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_A2B', type=str, default='/home/myid/zw63397/Projects/Crop_Detect/DT/GAN/output/netG_retina_A2B_withdet_200.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='/home/myid/zw63397/Projects/Crop_Detect/DT/GAN/output/netG_retina_B2A_withdet_200.pth', help='B2A generator checkpoint file')
opt = parser.parse_args()
opt.size = (640, 640)
print(opt)
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()

# Load state dicts
state_dict_with_prefix = torch.load(opt.generator_A2B)
state_dict = {k.replace('module.', ''): v for k, v in state_dict_with_prefix.items()}
netG_A2B.load_state_dict(state_dict)

state_dict_with_prefix = torch.load(opt.generator_B2A)
state_dict = {k.replace('module.', ''): v for k, v in state_dict_with_prefix.items()}
netG_B2A.load_state_dict(state_dict)

# Set model's test mode
netG_A2B.eval()
netG_B2A.eval()

device = torch.device('cuda' if opt.cuda else "cpu")
input_A = torch.empty(opt.batchSize, opt.input_nc, *opt.size, device=device)
input_B = torch.empty(opt.batchSize, opt.output_nc, *opt.size, device=device)

# Dataset loader
transforms_ = transforms.Compose([
        transforms.Resize(opt.size),
        transforms.ToTensor()
    ])
dataloader = DataLoader(CropDataset(opt.dataroot, transforms_=transforms_, rate=1.0), 
                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
###################################

path_name = os.path.join('/home/myid/zw63397/Projects/Crop_Detect/data', opt.img_outfolder, 'images')
if not os.path.exists(path_name):
    os.makedirs(path_name)

for batch in tqdm(dataloader):
    # Set model input
    real_B = input_B.copy_(batch['B'])
    fname = batch['B_fname'][0].replace('sim_data', opt.img_outfolder)

    # Generate output
    fake_A = 0.5*(netG_B2A(real_B).data + 1.0)
    # fake_A = netG_B2A(real_B).data

    # Save image files
    save_image(fake_A, fname, normalize=True)

print('Generated {} images in {}'.format(len(dataloader), opt.img_outfolder))