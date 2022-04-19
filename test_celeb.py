# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 14:31:53 2021

@author: MinYoung
"""



import torch, torchvision

import time, random
from tqdm import tqdm

from model import GeneratorResNet as Gen01
from model_v2 import GeneratorResNet as Gen02
from dataset import WaterMarkCelebDataset
from utils import merge_img

# Path
_LOAD_PATH_01       =       'weights/CelebA/WMR_0008.pt'
_LOAD_PATH_02       =       'weights/CelebA/WMR_Version_02_0025.pt'
_LOAD_PATH_03       =       'weights/CelebA/WMR_Version_02_1_0006.pt'

_TEST_SET_PATH      =       'img/CelebA/test/'

# For Device
CUDA                =                           torch.cuda.is_available()
DEVICE              =             torch.device('cuda' if CUDA else 'cpu')

# For Model Version 01
IMG_RESOLUTION      =          128
GEN_BASE_CHANNELS   =           64

gen01 = Gen01(num_resample= 2, num_residual_per_block= 2, base_channels= GEN_BASE_CHANNELS).to(DEVICE)
gen01.eval()

# For Model Version 02
GEN_BASE_CHANNELS   =           32

gen02 = Gen02(num_resample= 2, num_residual_per_block= 1, base_channels= GEN_BASE_CHANNELS).to(DEVICE)
gen02.eval()

# For Model Version 02_1
GEN_BASE_CHANNELS   =           64

gen03 = Gen02(num_resample= 3, num_residual_per_block= 2, base_channels= GEN_BASE_CHANNELS).to(DEVICE)
gen03.eval()

# For Testing
BATCH_SIZE          =            1


# Loading Process
checkpoint = torch.load(_LOAD_PATH_01)
gen01.load_state_dict(checkpoint['gen_dict'])

checkpoint = torch.load(_LOAD_PATH_02)
gen02.load_state_dict(checkpoint['gen_dict'])

checkpoint = torch.load(_LOAD_PATH_03)
gen03.load_state_dict(checkpoint['gen_dict'])


print(f'Dataset Loading Start')
start = time.time()
test_set = WaterMarkCelebDataset(_TEST_SET_PATH, IMG_RESOLUTION)
test_loader = torch.utils.data.DataLoader(test_set, batch_size= BATCH_SIZE, shuffle= False)
print(f'Dataset Loading Completed : {time.time() - start:0.2f} seconds')

time.sleep(1)

compare_list = []
step = 1
with torch.no_grad():
    
    loop = tqdm(test_loader)
    for batch_idx, (x, label) in enumerate(loop):
    
        x, label = x.to(DEVICE), label.to(DEVICE)
        N, C, H, W = x.shape
        
        fake_01, fake_02, fake_03 = gen01(x), gen02(x), gen03(x)
        compare = torch.cat([x, torch.zeros([1,3,5,128]).to(DEVICE), fake_01, fake_02, fake_03, torch.zeros([1,3,5,128]).to(DEVICE), label], dim= 2)
        
        compare_list.append(compare)
    
    
    for i in range(16):
        compare = torch.cat(compare_list[i:112:16], dim= 0)
        img_grid = torchvision.utils.make_grid(compare, normalize=False, padding = 0, nrow= 8)
        torchvision.utils.save_image(img_grid, f'./img/CelebA/output/{step:02d}.png')
        step += 1
        
        
