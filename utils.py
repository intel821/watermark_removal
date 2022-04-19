# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 19:38:27 2021

@author: MinYoung
"""
import torch, torchvision
import torchvision.transforms.functional as TF

def init_weights(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight.data)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight.data)
        
def gradient_penalty(critic, real, fake, k= 1, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - k) ** 2)
    return gradient_penalty

def split_img(img_tensor, split_size):
    
    C, H, W = img_tensor.shape
    
    H_n_split = H // split_size
    W_n_split = W // split_size
    
    splited_img_list = []
    
    for i in range(H_n_split):
        for j in range(W_n_split):
            splited_img_list.append(
                img_tensor[:, i * split_size : (i+1) * split_size, j * split_size : (j+1) * split_size])
    
    return splited_img_list, (len(splited_img_list), H_n_split, W_n_split)


def merge_img(img_tensor, H_n_split, W_n_split):

    idx = 0
    
    col_list = []
    for i in range(H_n_split):
        
        temp = []
        for j in range(W_n_split):
            temp.append(img_tensor[idx])
            idx += 1
        
        # print(temp)    
        col_list.append(torch.cat(temp, dim= -1))
    
    return torch.cat(col_list, dim= -2)

            
        
        
        
        
        
    