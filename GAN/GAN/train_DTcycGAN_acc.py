#!/usr/bin/python3

import argparse
import itertools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import os
from accelerate import Accelerator
import logging
from accelerate.logging import get_logger
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)
logging.getLogger('timm.models').setLevel(logging.WARNING)

from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import weights_init_normal
from datasets import ImageDataset, CropDataset
from model import DetLineModel



def train(opt, accelerator):

    device = accelerator.device
    ###### Definition of variables ######
    # Networks
    netG_A2B = Generator(opt.input_nc, opt.output_nc)
    netG_B2A = Generator(opt.output_nc, opt.input_nc)
    netD_A = Discriminator(opt.input_nc)
    netD_B = Discriminator(opt.output_nc)

    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)

    netDet = DetLineModel() # vit_small_patch16_224
    netDet.load_state_dict(torch.load('/home/myid/zw63397/Projects/Crop_Detect/DT/Detector/models/box/best_model_46_0.0105.pth',\
                                      map_location=torch.device('cpu')))

    # Lossess
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    criterion_consistency = torch.nn.L1Loss()
    criterion_operation = torch.nn.L1Loss()

    # Optimizers & LR schedulers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                    lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_Det = torch.optim.SGD(netDet.parameters(), lr=opt.lr)

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_Det = torch.optim.lr_scheduler.LambdaLR(optimizer_Det, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

    # Inputs & targets memory allocation
    input_A = torch.empty(opt.batchSize, opt.input_nc, *opt.size, device=device)
    input_B = torch.empty(opt.batchSize, opt.output_nc, *opt.size, device=device)
    target_real = torch.ones(opt.batchSize, 1, device=device, requires_grad=False)
    target_fake = torch.zeros(opt.batchSize, 1, device=device, requires_grad=False)
    # print(input_A.shape, target_real.shape)
    # print(target_real, target_fake)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Dataset loader
    transforms_ = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2))
    ])

    transforms_det = transforms.Resize((224, 224))

    train_set = CropDataset(opt.dataroot, transforms_=transforms_, rate=1.0)
    dataloader = DataLoader(train_set, batch_size=opt.batchSize, drop_last=True)
    logger.info('dataset len: {}'.format(len(dataloader)), main_process_only=True)

    netG_A2B, netG_B2A, netD_A, netD_B, netDet, optimizer_G, optimizer_D_A, optimizer_D_B, optimizer_Det, dataloader = accelerator.prepare(
        netG_A2B, netG_B2A, netD_A, netD_B, netDet, optimizer_G, optimizer_D_A, optimizer_D_B, optimizer_Det, dataloader
    )

    # Loss plot
    if accelerator.is_main_process:
        writer = SummaryWriter('./runs/DT_GAN')


    ###### Training ######
    step = 0
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):
            step += 1
            # Set model input: A for real, B for sim
            # print(input_A.shape, batch['A'].shape)
            real_A = input_A.copy_(batch['A'])
            real_B = input_B.copy_(batch['B'])
            label = batch['B_label']

            ###### Generators A2B and B2A ######

            # Identity loss
            # G_A2B(B) should equal B if real B is fed
            same_B = netG_A2B(real_B) # B -> B
            loss_identity_B = criterion_identity(same_B, real_B)*2.0
            # G_B2A(A) should equal A if real A is fed
            same_A = netG_B2A(real_A) # A -> A
            loss_identity_A = criterion_identity(same_A, real_A)*2.0

            # GAN loss
            fake_B = netG_A2B(real_A) # A -> B
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

            fake_A = netG_B2A(real_B) # B -> A
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

            # Consist loss
            y_real = netDet(transforms_det(real_B))
            y_fake = netDet(transforms_det(fake_A))
            z_real = netDet(transforms_det(real_A))
            z_fake = netDet(transforms_det(fake_B))
            loss_consist = criterion_consistency(y_fake, y_real) * 10.0 + criterion_consistency(z_fake, z_real) * 10.0

            # Cycle loss
            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*5.0

            recovered_B = netG_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*5.0

            # Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + loss_consist
            
            optimizer_G.zero_grad()
            accelerator.backward(loss_G)
            optimizer_G.step()
            ###################################

            y_real = netDet(transforms_det(real_B))
            y_fake = netDet(transforms_det(fake_A.detach()))
            loss_det = criterion_operation(y_fake,label)*5 +criterion_operation(y_real,label)*5
            optimizer_Det.zero_grad()
            accelerator.backward(loss_det)
            optimizer_Det.step()

            ###### Discriminator A ######

            # Real loss
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake)*0.5

            optimizer_D_A.zero_grad()
            accelerator.backward(loss_D_A)
            optimizer_D_A.step()
            ###################################

            ###### Discriminator B ######

            # Real loss
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake)*0.5

            optimizer_D_B.zero_grad()
            accelerator.backward(loss_D_B)
            optimizer_D_B.step()
            ###################################

            if (step)%10 == 0:
                logger.info('Epoch {} Step {}, Loss: {}'.format(epoch+1,step,loss_G), main_process_only=True)
                # accelerator.print(loss_G, loss_identity_A + loss_identity_B, \
                # loss_GAN_A2B + loss_GAN_B2A, loss_cycle_ABA + loss_cycle_BAB, \
                # loss_D_A + loss_D_B, loss_consist, loss_det)
            if accelerator.is_main_process:
                writer.add_scalar('Loss_G',loss_G,step)
                writer.add_scalar('Loss_G_identity',loss_identity_A + loss_identity_B,step)
                writer.add_scalar('Loss_G_GAN',loss_GAN_A2B + loss_GAN_B2A,step)
                writer.add_scalar('Loss_G_cycle',loss_cycle_ABA + loss_cycle_BAB,step)
                writer.add_scalar('Loss_D',loss_D_A + loss_D_B,step)
                writer.add_scalar('Loss_G_consistent',loss_consist,step)
                writer.add_scalar('Loss_Det',loss_det,step)

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()
        lr_scheduler_Det.step()

        # Save models checkpoints
        if (epoch+1)%10 == 0 and accelerator.is_main_process:
            if not os.path.exists(opt.outdir):
                os.makedirs(opt.outdir)
            accelerator.save(netG_A2B.module.state_dict(), '{}/netG_dt_A2B_withdet_{}.pth'.format(opt.outdir, epoch+1))
            accelerator.save(netG_B2A.module.state_dict(), '{}/netG_dt_B2A_withdet_{}.pth'.format(opt.outdir, epoch+1))
            accelerator.save(netD_A.module.state_dict(), '{}/netD_dt_A_withdet_{}.pth'.format(opt.outdir, epoch+1))
            accelerator.save(netD_B.module.state_dict(), '{}/netD__dt_B_withdet_{}.pth'.format(opt.outdir, epoch+1))
            accelerator.save(netDet.module.state_dict(), '{}/netDet_{}.pth'.format(opt.outdir, epoch+1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
    parser.add_argument('--batchSize', type=int, default=8, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='/home/myid/zw63397/Projects/Crop_Detect/data', help='root directory of the dataset')
    parser.add_argument('--outdir', type=str, default='/home/myid/zw63397/Projects/Crop_Detect/DT/GAN/output', help='output dir')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    opt = parser.parse_args()
    opt.size = (224,224)
    # print(opt)

    # Initialize the Accelerator
    accelerator = Accelerator()
    train(opt, accelerator)


if __name__ == '__main__':
    main()
