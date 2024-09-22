#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 10:27:29 2024

@author: francesco
"""

#import section
from models import Generator, Discriminator
import torch
from torch import nn
from dataset import ImageFolderCustom
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from helper_functions import get_noise, combine_vectors, get_one_hot_labels, show_tensor_images
import matplotlib.pyplot as plt
import numpy as np

#define dataset path
train_data_path = "images/train"

#set up a transformation pipeline to resize images, turn them into tensors and apply normalization
train_transform = transforms.Compose([
    transforms.Resize(size=(28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])



#create the dataset
train_dataset = ImageFolderCustom(train_data_path, transform=train_transform)
#create the dataloader
BATCH_SIZE = 32
NUM_WORKERS = 0
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              num_workers = NUM_WORKERS,
                              shuffle=True)

#set up other params
dim_z = 10
num_classes = len(train_dataset.classes)
channels_images = train_dataset[0][0].shape[0]
disc_lr = 0.0001
gen_lr = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#model definition
gen = Generator(input_channels=dim_z+num_classes,output_channels=train_dataset[0][0].shape[0]).to(device)
disc = Discriminator(input_channels=channels_images+num_classes,output_channels=train_dataset[0][0].shape[0]).to(device)

#optimizer definition
gen_optimizer = torch.optim.Adam(gen.parameters(), lr=gen_lr)
disc_optimizer = torch.optim.Adam(disc.parameters(), lr=disc_lr)

#loss function definition
criterion = nn.BCEWithLogitsLoss()

#training loop
epochs = 200
gen_losses = []
disc_losses = []

for epoch in range(epochs):
    
    ## LOAD THE DATA ##
    temp_gen_losses = []
    temp_disc_losses = []
    for real,label in tqdm(train_dataloader):
        #generator
        noise = get_noise(real.shape[0], dim_z)
        one_hot_encoding = get_one_hot_labels(label, num_classes)
        gen_samples = combine_vectors(noise, one_hot_encoding)
        
        #discriminator
        one_hot_encoding_image = get_one_hot_labels(label, 
                                                    num_classes, 
                                                    shape=(train_dataset[0][0].shape[1],train_dataset[0][0].shape[2]))
        fake = gen(gen_samples)
        #combine output with image level one hot encoding
        disc_samples_fake = combine_vectors(fake.detach(), one_hot_encoding_image)
        disc_samples_real = combine_vectors(real, one_hot_encoding_image)
        #make predictions over true and fake images
        disc_fake_pred = disc(disc_samples_fake)
        disc_true_pred = disc(disc_samples_real)
    
        ### UPDATE DISCRIMINATOR ###
        #do the forward pass
        #calculate the loss
        loss_disc_fake = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        loss_disc_real = criterion(disc_true_pred, torch.ones_like(disc_true_pred))
        loss_disc = (loss_disc_fake + loss_disc_real)/2
        ######################################################################
        ### SAVE LOSS FOR PLOT ###
        temp_disc_losses.append(loss_disc.detach().to('cpu').numpy().item())
        ######################################################################
        #optimized zero grad
        disc_optimizer.zero_grad()
        #loss backward
        loss_disc.backward()
        #optimizer step
        disc_optimizer.step()
        
        
        
        ### UPDATE GENERATOR ###
        #do the forward pass
        gen_sample_fake = combine_vectors(fake, one_hot_encoding_image)
        fake_prediction = disc(gen_sample_fake)
        #calculate the loss
        loss_gen = criterion(fake_prediction,torch.ones_like(fake_prediction))
        ######################################################################
        ### SAVE LOSS FOR PLOT ###
        temp_gen_losses.append(loss_gen.detach().to('cpu').numpy().item())
        ######################################################################
        #optimized zero grad
        gen_optimizer.zero_grad()
        #loss backward
        loss_gen.backward()
        #optimizer step
        gen_optimizer.step()
    
    gen_losses.append(np.array(temp_gen_losses).mean())
    disc_losses.append(np.array(temp_disc_losses).mean())
        
        
    if (epoch%2 == 0) and (epoch != 0):
        #define x data
        x = [i for i in range(epoch+1)]
        plt.plot(x,gen_losses,label="Generator")
        plt.plot(x,disc_losses,label="Discriminator")
        plt.legend()
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title("Generatore vs Discriminator loss")
        plt.show()
        
        show_tensor_images(fake)
        show_tensor_images(real)
        
#%%

show_tensor_images(fake)
            
            
            
            
            
            
            
            
            
            
            
            
            
    
        
        
