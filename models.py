#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 15:46:26 2024

@author: francesco
"""

#import section
import torch
from torch import nn


#create generator model
class Generator(nn.Module):
    def __init__(self,input_channels=10,hidden_channels=64,output_channels=1,kernel_size=4, stride=1):
        super().__init__()
        self.input_channels = input_channels
        #define structure of the generator
        self.gen_model = nn.Sequential(
            self.generate_middle_block(input_channels,hidden_channels*4),
            self.generate_middle_block(hidden_channels*4,hidden_channels*2,kernel_size,stride),
            self.generate_middle_block(hidden_channels*2,hidden_channels),
            self.generate_output_block(hidden_channels,output_channels,kernel_size)
            )
        
    def forward(self,x):
        x = x.view(len(x), self.input_channels, 1, 1)
        return self.gen_model(x)

        
        
    def generate_middle_block(self, input_channels, output_channels,kernel_size=3,stride=2):
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size,stride),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(negative_slope=0.2)
            )

    
    def generate_output_block(self,hidden_channels,output_channels,kernel_size=3,stride=2):
        return nn.Sequential(
            nn.ConvTranspose2d(hidden_channels,output_channels,kernel_size,stride),
            nn.Tanh())
    
#%% 
    
class Discriminator(nn.Module):
    def __init__(self,input_channels=1,hidden_channels=64,output_channels=1):
        super().__init__()
        
        #define structure of the discrinimator
        self.disc_model = nn.Sequential(
            self.generate_middle_block(input_channels,hidden_channels),
            self.generate_middle_block(hidden_channels,hidden_channels*2),
            self.generate_output_block(hidden_channels*2,output_channels)
            )
        
    def forward(self,x):
        return self.disc_model(x)
    
    def generate_middle_block(self,input_channels,output_channels,kernel_size=4,stride=2):
        return nn.Sequential(
            nn.Conv2d(input_channels,output_channels,kernel_size,stride),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(negative_slope=0.2)
            )
    
    def generate_output_block(self,input_channels,output_channels,kernel_size=4,stride=2):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride)
            )
        