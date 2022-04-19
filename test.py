# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 21:44:54 2021

@author: MinYoung
"""


import torch, torchvision

import time, random
from tqdm import tqdm

from model import GeneratorResNet
from dataset import WaterMarkDataset
from utils import merge_img

# Path
_LOAD_PATH          =       'weights/Places365_val/WMR_temp.pt'
_TEST_SET_PATH     =       'img/Places365_val/test/'

# For Device
CUDA                =                           torch.cuda.is_available()
DEVICE              =             torch.device('cuda' if CUDA else 'cpu')

# For Model
IMG_RESOLUTION      =          128
GEN_BASE_CHANNELS   =           64
DISC_BASE_CHANNELS  =           64

# For Testing
BATCH_SIZE          =            8

gen = GeneratorResNet(num_resample= 3, num_residual_per_block= 2, base_channels= GEN_BASE_CHANNELS).to(DEVICE)
gen.eval()

# Loading Process
checkpoint = torch.load(_LOAD_PATH)
gen.load_state_dict(checkpoint['gen_dict'])


print(f'Dataset Loading Start')
start = time.time()
test_set = WaterMarkDataset(_TEST_SET_PATH, IMG_RESOLUTION)
test_loader = torch.utils.data.DataLoader(test_set, batch_size= BATCH_SIZE, shuffle= False)
print(f'Dataset Loading Completed : {time.time() - start:0.2f} seconds')

time.sleep(1)

with torch.no_grad():
    
    loop = tqdm(test_loader)
    for batch_idx, (x, label) in enumerate(loop):
    
        x, label = x.to(DEVICE), label.to(DEVICE)
        N, C, H, W = x.shape
        num_img = N // 4
    
        fake = gen(x)
        
        compare_list = []
        
        for i in range(num_img):
            merged_input = merge_img(x[ 4 * i : 4 * (i+1) ], 2, 2)
            merged_output = merge_img(fake[ 4 * i : 4 * (i+1) ], 2, 2)
            merged_label = merge_img(label[ 4 * i : 4 * (i+1) ], 2, 2)
            
            compare_list.append(torch.cat([merged_input, merged_output, merged_label], dim= -1))
        
        compare = torch.stack(compare_list, dim= 0)
            
        img_grid = torchvision.utils.make_grid(compare, normalize=False, padding = 5, nrow= 1)
        torchvision.utils.save_image(img_grid, f'./img/Places365_val/output/{batch_idx:04d}.png')
