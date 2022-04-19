# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 16:30:57 2021

@author: MinYoung
"""

import torch, torchvision
from torchvision.io import read_image

import os
import numpy as np
from PIL import Image
from time import time

from utils import split_img, merge_img

# 주의사항 모든 폴더의 이미지는 똑같은 순서대로 있어야하며
# 폴더마다 들어있는 이미지의 갯수는 같아야 한다
class WaterMarkDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, resolution):
        self.root_dir = root_dir
        self.resolution = resolution
        
        self.label_path = self.root_dir + 'labels'
        
        self.dir_list = os.listdir(self.root_dir)
        self.files_list = os.listdir(self.root_dir + self.dir_list[0])
        
        self.n_file = len(self.files_list)
        self.files_list *= (len(self.dir_list) - 1)
        self.label_files_list = os.listdir(self.label_path)
        
        self.images_list = []
        self.label_images_list = []
        
        for idx, file in enumerate(self.files_list):
            img_file = self.root_dir + self.dir_list[idx // self.n_file] + '/' + file
            input_image = torchvision.io.read_image(img_file) / 255.0
            splited_list, _ = split_img(input_image, self.resolution)
            self.images_list += splited_list
        
        for idx, file in enumerate(self.label_files_list):
            label_img_file = self.label_path + '/' + file
            input_image = torchvision.io.read_image(label_img_file) / 255.0
            if input_image.size(0) == 1:
                input_image = torch.cat([input_image.data] * 3, dim= 0)
            splited_list, _ = split_img(input_image, self.resolution)
            self.label_images_list += splited_list
            
        self.n_img = len(self.label_images_list)
            
               
        
    def __len__(self):
        return len(self.images_list)
    
    def __getitem__(self, index):
        return self.images_list[index], self.label_images_list[index % self.n_img]
    
class WaterMarkCelebDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, resolution):
        
        self.root_dir = root_dir
        self.resolution = resolution
        
        self.label_path = self.root_dir + 'labels'
        
        self.dir_list = os.listdir(self.root_dir)
        self.files_list = os.listdir(self.root_dir + self.dir_list[0])
        
        self.n_file = len(self.files_list)
        self.files_list *= (len(self.dir_list) - 1)
        self.label_files_list = os.listdir(self.label_path)
        
        self.images_list = []
        self.label_images_list = []
        
        for idx, file in enumerate(self.files_list):
            self.images_list.append(self.root_dir + self.dir_list[idx // self.n_file] + '/' + file)
        
        for idx, file in enumerate(self.label_files_list):
            self.label_images_list.append(self.label_path + '/' + file)
            
        self.n_img = len(self.label_images_list)
        
        
    def __len__(self):
        return len(self.images_list)
    
    def __getitem__(self, index):
        return read_image(self.images_list[index]) / 255.0, read_image(self.label_images_list[index % self.n_img]) / 255.0 

    
if __name__ == '__main__':
    
    
    dset = WaterMarkCelebDataset('img/Places365_val/train_/', 128)
    
    print(len(dset))
    
    loader = torch.utils.data.DataLoader(dset, batch_size= 1)
    
    for batch_idx, (polluted, label) in enumerate(loader):
        
        if batch_idx % 1000 == 0:
            print(batch_idx)
            
            polluted, label = polluted[0].permute(1,2,0).numpy(), label[0].permute(1,2,0).numpy()
            
            from matplotlib import pyplot as plt
            plt.imshow(polluted)
            plt.show()
            plt.imshow(label)
            plt.show()
        