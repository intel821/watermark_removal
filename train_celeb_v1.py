# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 15:33:34 2021

@author: MinYoung
"""


import torch, torchvision, torchsummaryX
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import time, random
from tqdm import tqdm
from matplotlib import pyplot as plt

from model import GeneratorResNet, Discriminator, MobileNetV2
from utils import gradient_penalty
from dataset import WaterMarkCelebDataset


# Path
_SAVE_PATH          =       'weights/CelebA/WMR_'
_LOAD_PATH          =       'weights/CelebA/WMR_0009.pt'
_TRAIN_SET_PATH     =       'img/CelebA/train/'

# For Device
CUDA                =                           torch.cuda.is_available()
DEVICE              =             torch.device('cuda' if CUDA else 'cpu')

# Coefficient
LAMBDA_GP           =            1
LAMBDA_PL           =            1
LAMBDA_L1           =            1
LIP_CONST           =          1.0

# For Model
IMG_RESOLUTION      =          128
GEN_BASE_CHANNELS   =           64
DISC_BASE_CHANNELS  =           64

# For Training
BATCH_SIZE          =            8
LEARNING_RATE       =         2e-4
P_TRAIN_GEN         =         0.25
NUM_EPOCHS          =         int(5/P_TRAIN_GEN)


SCHEDULE_G          =           []#[NUM_EPOCHS // 2]
SCHEDULE_D          =           []#[NUM_EPOCHS // 2]

# To continue training
LOAD                =         True
EPOCH_OFFSET        =            0
STEP                =            1

gen = GeneratorResNet(num_resample= 2, num_residual_per_block= 2, base_channels= GEN_BASE_CHANNELS).to(DEVICE)
disc = Discriminator(num_downsample= 4, resolution= IMG_RESOLUTION, base_channels= DISC_BASE_CHANNELS, snorm= True).to(DEVICE)
pretrained_mobilenet_v2 = MobileNetV2().to(DEVICE)

perceptual_criterion = nn.MSELoss()
l1_criterion = nn.L1Loss()

gen.train(), disc.train(), pretrained_mobilenet_v2.eval()


# lr_g_list, lr_c_list = [], []


if LOAD:
    checkpoint = torch.load(_LOAD_PATH)
    EPOCH_OFFSET = checkpoint['epoch']
    STEP = checkpoint['step']
    # lr_g_list = checkpoint['lr_g_list']
    # lr_c_list = checkpoint['lr_c_list']
    gen.load_state_dict(checkpoint['gen_dict'])
    disc.load_state_dict(checkpoint['disc_dict'])
    
print(f'Dataset Loading Start')
start = time.time()
train_set = WaterMarkCelebDataset(_TRAIN_SET_PATH, IMG_RESOLUTION)
train_loader = torch.utils.data.DataLoader(train_set, batch_size= BATCH_SIZE, shuffle= True)
print(f'Dataset Loading Completed : {time.time() - start:0.2f} seconds')

# Optimizers & LearningRate Schedulers
opt_g = optim.Adam(gen.parameters(), lr= LEARNING_RATE, betas= (0.0, 0.9))
opt_d = optim.Adam(disc.parameters(), lr= 2 * LEARNING_RATE, betas= (0.0, 0.9))
scheduler_g = torch.optim.lr_scheduler.MultiStepLR(opt_g, gamma= 0.5, milestones= SCHEDULE_G)
scheduler_d = torch.optim.lr_scheduler.MultiStepLR(opt_d, gamma= 0.5, milestones= SCHEDULE_D)

time.sleep(1)

for epoch in range(EPOCH_OFFSET, NUM_EPOCHS + EPOCH_OFFSET):

    loop = tqdm(train_loader)
    loop.set_description(f'Epoch [{epoch + 1}/{NUM_EPOCHS + EPOCH_OFFSET}]')

    for batch_idx, (x, label) in enumerate(loop):

        x, label = x.to(DEVICE), label.to(DEVICE)

        fake = gen(x)
        disc_fake, disc_label = disc(fake), disc(label)
        
        gp = gradient_penalty(disc, label, fake, k= LIP_CONST, device= DEVICE)
        
        # -(E[min(0, -K + D(x_real))] + E[min(0, -K - D(x_fake))]) + LAMBDA_GP * Gradient Panelty     --> -() For gradient descent
        loss_disc = (
                         -(     torch.mean( torch.min(torch.zeros_like(disc_label),  - LIP_CONST + disc_label))      # E[min(0, - Lip Const + D(x_real))]
                              + torch.mean( torch.min(torch.zeros_like(disc_fake),   - LIP_CONST - disc_fake))       # E[min(0, - Lip Const - D(x_fake))]
                          )
                         + LAMBDA_GP * gp
                      )     
        
        disc.zero_grad()
        loss_disc.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm= 1)
        opt_d.step()

        if (epoch == EPOCH_OFFSET and batch_idx == 0) or random.uniform(0, 1) < P_TRAIN_GEN:

            gen.zero_grad()
            
            fake = gen(x)
            gen_fake = disc(fake)
            gan_loss = -torch.mean(gen_fake)
            
            perceptual_fake, perceptual_label = pretrained_mobilenet_v2(fake), pretrained_mobilenet_v2(label)
            perceptual_loss = perceptual_criterion(perceptual_fake, perceptual_label)  # Criterion -> MSE Loss
            
            l1_loss = l1_criterion(fake, label)

            loss_gen = gan_loss + LAMBDA_PL * perceptual_loss + LAMBDA_L1 * l1_loss
            
            pretrained_mobilenet_v2.zero_grad()
            
            loss_gen.backward()
            opt_g.step()
            
        loop.set_postfix(Discriminator_loss = loss_disc.item(), GAN_loss = gan_loss.item(), L1_loss = LAMBDA_L1 * l1_loss.item(), Perceptual_loss = LAMBDA_PL * perceptual_loss.item())
        
        
        if batch_idx % 50 == 0:
            gen.eval()
            
            with torch.no_grad():
                
                input = x[0:4]
                fake = gen(x[0:4])
                real = label[0:4]
                
                compare = torch.cat([input.data, fake.data, real.data], dim= 0)
                img_grid = torchvision.utils.make_grid(compare, normalize=False, padding = 3, nrow= 4)

                torchvision.utils.save_image(img_grid, f'./img/CelebA/compare/{STEP:04d}.png')
                
                STEP += 1
                
                torch.save({
                    'gen_dict' : gen.state_dict(),
                    'disc_dict' : disc.state_dict(),
                    'epoch' : epoch + 1,
                    'step' : STEP,        
                    
                    # 'lr_g_list' : lr_g_list,
                    # 'lr_c_list' : lr_c_list,
                    
                    }, _SAVE_PATH + 'temp.pt')
                
                gen.train()
  
    # lr_g_list.append(scheduler_g.get_last_lr()[0])
    # lr_c_list.append(scheduler_c.get_last_lr()[0])
    scheduler_g.step()
    scheduler_d.step()
    
    torch.save({
        'gen_dict' : gen.state_dict(),
        'disc_dict' : disc.state_dict(),
        'epoch' : epoch + 1,
        'step' : STEP,        
        
        # 'lr_g_list' : lr_g_list,
        # 'lr_c_list' : lr_c_list,
        
        }, _SAVE_PATH + f'{epoch:04d}.pt')

