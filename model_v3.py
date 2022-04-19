# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 16:09:50 2021

@author: MinYoung
"""

#-------------------'Version_02'-----------------
# Final layers of discriminator are changed
# Avg pooling -> 2 layers of Fully connected Layers for Patch Gan
# elementwise summation -> channel concatenation

#-------------------'Version_03'-----------------
# List -> nn.ModuleList For nn.Sequential
# Upsampling Layer input channel / output channel ratio 4 
# -> adopt 1x1 conv that ratio of 2 for intermediate layer
# Added SE blocks for all layers of generator




import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm

from utils import init_weights

from torchsummary import summary



class _ResidualBlock(nn.Module):
    def __init__(self, in_and_out_channels, norm):
        super(_ResidualBlock, self).__init__()
        
        layers = nn.ModuleList()
        for _ in range(2):
            layers.append(nn.Conv2d(in_and_out_channels, in_and_out_channels, kernel_size= 3, stride= 1, padding= 1, bias= not norm))
            if norm:
                layers.append(nn.BatchNorm2d(in_and_out_channels))
            layers.append(nn.ReLU(inplace= True))
            
        self.block = nn.Sequential(*layers)
        self.block.apply(init_weights)
     
    def forward(self, x):
        return self.block(x) + x
    
class _SEBlock(nn.Module):
    def __init__(self, channels, reduction_ratio= 16):
        super(_SEBlock, self).__init__()
        
        se_layers = nn.ModuleList()
        
        se_layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        se_layers.append(nn.Flatten())
        se_layers.append(nn.Linear(channels, channels // reduction_ratio))
        se_layers.append(nn.ReLU(inplace= True))
        se_layers.append(nn.Linear(channels // reduction_ratio, channels))
        se_layers.append(nn.Sigmoid())
        
        self.se_block = nn.Sequential(*se_layers)
        self.se_block.apply(init_weights)
        
    def forward(self, x):
        return x * self.se_block(x).view(x.size(0), -1, 1, 1)
    
    
class _TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, resample, norm):
        super(_TransitionBlock, self).__init__()

        layers = nn.ModuleList()
        
        # Resample Mode ['down', 'up']
        
        if resample == 'down':
            
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size= 3, stride= 2, padding= 1, bias= not norm))
            if norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace= True))
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size= 3, stride= 1, padding= 1, bias= not norm))
            if norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace= True))
        
        elif resample == 'up':
            
            # Version_03 appended
            layers.append(nn.Conv2d(in_channels, in_channels // 2, kernel_size= 1, padding= 0, bias= False))
            layers.append(nn.ReLU(inplace= True))
            
            in_channels //= 2
            
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size= 4, stride= 2, padding= 1, bias= not norm))
            if norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace= True))
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size= 3, stride= 1, padding= 1, bias= not norm))
            if norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace= True))
            

        else:
            print(f'resample could be "up" and "down" try again')
            exit()
        
        self.block = nn.Sequential(*layers)
        self.block.apply(init_weights)
        
    def forward(self, x):
        return self.block(x)
    
class _GeneratorFirstBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm):
        super(_GeneratorFirstBlock, self).__init__()
        
        layers = nn.ModuleList()
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size= 3, stride= 1, padding= 1, bias= not norm))
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace= True))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size= 3, stride= 1, padding= 1, bias= not norm))
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace= True))
        
        self.block = nn.Sequential(*layers)
        self.block.apply(init_weights)
        
    def forward(self, x):
        return self.block(x)
    
class _GeneratorFinalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm):
        super(_GeneratorFinalBlock, self).__init__()
       
        layers = nn.ModuleList()
        layers.append(nn.Conv2d(in_channels, in_channels, kernel_size= 3, stride= 1, padding= 1, bias= not norm))
        if norm:
            layers.append(nn.BatchNorm2d(in_channels))
        layers.append(nn.ReLU(inplace= True))
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size= 3, stride= 1, padding= 1, bias= not norm))
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace= True))

        self.block = nn.Sequential(*layers)
        self.block.apply(init_weights)
        
    def forward(self, x):
        return self.block(x)

class GeneratorResNet(nn.Module):
    def __init__(self, num_resample, num_residual_per_block, base_channels= 32, norm= True, se_block= True):
        super(GeneratorResNet, self).__init__()
    
        self.num_resample = num_resample
        self.num_residual_per_block = num_residual_per_block

        # First Conv Block
        in_channels = 3
        current_channels = base_channels

        self.conv_first = _GeneratorFirstBlock(in_channels, current_channels, norm= norm)
        if se_block:
            self.conv_first = nn.Sequential(self.conv_first, _SEBlock(current_channels))
        
        # Make Downsampling Layers
        # Residual and Transition Layers
        for i in range(self.num_resample):
            
            block = nn.ModuleList()
            
            for _ in range(self.num_residual_per_block):
                block.append(_ResidualBlock(current_channels, norm= norm))
                if se_block:
                    block.append(_SEBlock(current_channels))
                    
                
            block.append(_TransitionBlock(current_channels, current_channels * 2, resample= 'down', norm= norm))
            current_channels *= 2
            if se_block:
                block.append(_SEBlock(current_channels))
            
            self.add_module('downsample_' + str(i+1), nn.Sequential(*block))
        
        
        # Mid Residual Block
        if se_block:
            block = [_ResidualBlock(current_channels, norm= norm), _SEBlock(current_channels)] * self.num_residual_per_block
        else:
            block = [_ResidualBlock(current_channels, norm= norm)] * self.num_residual_per_block
        self.middle = nn.Sequential(*block)
        
        # Make Upsampling Layers
        # Residual and Transition Layers
        for i in range(self.num_resample):
            
            block = nn.ModuleList()
            
            current_channels *= 2       # Because of Channel Concatenation for using intermediate calculation
            
            block.append(_TransitionBlock(current_channels, current_channels // 4, resample= 'up', norm= norm))
            current_channels = current_channels // 4
            if se_block:
                block.append(_SEBlock(current_channels))
            
            for _ in range(self.num_residual_per_block):
                block.append(_ResidualBlock(current_channels, norm= norm))
                if se_block:
                    block.append(_SEBlock(current_channels))

            self.add_module('upsample_' + str(i+1), nn.Sequential(*block))

        # Final Conv Block
        current_channels *= 2
        self.conv_final = _GeneratorFinalBlock(current_channels, in_channels, norm= norm)
        
    def forward(self, x):
        
        intermediate_result = []
        
        out = self.conv_first(x)
        intermediate_result.append(out)
        
        # Downsampling
        for i in range(self.num_resample):
            
            for name, module in self.named_children():
                if name == 'downsample_' + str(i+1):
                    out = module(out)
                    intermediate_result.append(out)
        
        # Mid Residual Block
        out = self.middle(out)
        intermediate_result.reverse()

        # Upsampling
        for i in range(self.num_resample):
            
            for name, module in self.named_children():
                if name == 'upsample_' + str(i+1):
                    out = module(torch.cat([out, intermediate_result[i]], dim= 1))
                    
        out = self.conv_final(torch.cat([out, intermediate_result[-1]], dim= 1))
        return out





class Discriminator(nn.Module):
    def __init__(self, resolution, num_downsample, base_channels= 32, norm= True, snorm= False):
        super(Discriminator, self).__init__()

        # Output Resolution for PatchGAN
        self.output_resolution = resolution // (2 ** num_downsample)
        
        model = nn.ModuleList()
        current_channels = base_channels
        
        model.append(self._make_conv_layer(in_channels= 3, out_channels= current_channels, kernel_size= 3, stride= 1, padding= 1, norm= norm, snorm= snorm))
        
        for i in range(num_downsample):
            model.append(self._make_conv_layer(current_channels, current_channels * 2, kernel_size= 3, stride= 2, padding= 1, norm= norm, snorm= snorm))
            current_channels *= 2
            
        model.append(self._make_conv_layer(current_channels, base_channels, kernel_size= 1, stride= 1, padding= 0, norm= norm, snorm= snorm))
        
        if snorm:
            model.append(spectral_norm(nn.Conv2d(base_channels, 1, kernel_size= 1, stride= 1, padding= 0, bias= norm)))
        else:
            model.append(nn.Conv2d(base_channels, 1, kernel_size= 1, stride= 1, padding= 0, bias= norm))
        
        self.model = nn.Sequential(*model)
        
        self.fc1 = nn.Linear(self.output_resolution ** 2, self.output_resolution)
        self.fc2 = nn.Linear(self.output_resolution, 1)
        
        self.model.apply(init_weights)

    def forward(self, x):
        
        out = self.model(x).view(x.size(0), self.output_resolution ** 2)
        out = self.fc2(self.fc1(out))
        
        return torch.tanh(out)

    def _make_conv_layer(self, in_channels, out_channels, kernel_size, stride, padding, norm, snorm):
        
        layers = nn.ModuleList()
        
        if snorm:
            layers.append(
                spectral_norm(
                    nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias= not norm)
                              ))
        else:            
            layers.append(
                    nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias= not norm)
                              )
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        out = nn.Sequential(*layers)
 
        return out

class MobileNetV2(nn.Module):
    
    def __init__(self):
        super(MobileNetV2, self).__init__()
        mobilenet_features = torchvision.models.mobilenet_v2(pretrained= True).features
        
        self.model = nn.Sequential()
        
        for i in range(6):
            self.model.add_module(str(i), mobilenet_features[i])

    def forward(self, x):
        return self.model(x)
    


if __name__ == '__main__':
    
    gen = GeneratorResNet(num_resample = 2, num_residual_per_block= 1).to('cuda')
    print(gen)
    summary(gen, (3, 128, 128))
    print(gen(torch.randn(1,3,256,256).to('cuda')).size())
    
    disc = Discriminator(resolution= 128, num_downsample= 4).to('cuda')
    # print(disc)
    summary(disc, (3, 128, 128))
    print(disc(torch.randn(1,3,128,128).to('cuda')).size())
    
    # percep = MobileNetV2().to('cuda')
    # summary(percep, (3, 256, 256))
    # # print(percep)
    # print(percep(torch.randn(1,3,256,256).to('cuda')).size())