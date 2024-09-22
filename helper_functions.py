#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 15:07:22 2024

@author: francesco
"""

#import section
import torch
from torch.utils.data import Dataset
from typing import List, Tuple
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
from torchvision.utils import make_grid


#function taking in a dataset and showing random images
def display_random_images(dataset: torch.utils.data.Dataset, 
                          classes: List[str] = None, 
                          n: int = 10, 
                          display_shape: bool = True, 
                          seed: int = None):
    '''

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        DESCRIPTION. dataset loaded as a dataset pytorch object
    classes : List[str], optional
        DESCRIPTION. The default is None. List containing all the class names
    n : int, optional
        DESCRIPTION. The default is 10. Number of images to be displayed
    display_shape : bool, optional
        DESCRIPTION. The default is True. If the shape of the image has to be displayed in the title
    seed : int, optional
        DESCRIPTION. The default is None. For reproducible sampling strategy

    Returns
    -------
    None.

    '''
        
    #contrain n imagesto display to be max 10
    if n > 10:
        n = 10
        display_shape = False
        print(f"Number of images should not exceed 10. The parameter has been set back to 10")
    
    #if we want to set the random porcess to be repeatable
    if seed:
        random.seed(seed)
        
    #extract set of random images from the entire dataset, k is the number of 
    #images to be sampled and is fixed to n
    random_sample_idx = random.sample(range(len(dataset)), k=n)
    
    #setup plot
    plt.figure(figsize=(16,8))
    
    #loop over the random idxs and plot the related images
    for i, targ_sample in enumerate(random_sample_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]
        
        #adjust tesnor dimension to have channel last (matplotlib requirement)
        targ_image_adjust = targ_image.permute(1,2,0) #[H,W,C]
        
        #plot adjusted images
        plt.subplot(1,n,i+1)
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        if classes:
            title = f"Class: {classes[targ_label]}"
            if display_shape:
                title = title + f"\nshape: {targ_image_adjust.shape}"
        plt.title(title)
        
        

def get_noise(n_samples: int, z_dim: int, device='cpu') -> torch.Tensor:
    '''
    

    Parameters
    ----------
    n_samples : int
        DESCRIPTION. Number of random vector samples to produuce
    z_dim : int
        DESCRIPTION. dimension of the random vector to produce
    device : TYPE, optional
        DESCRIPTION. The default is 'cpu'.

    Returns
    -------
    TYPE
        DESCRIPTION. tensor of noise vectors for the generator

    '''
    return torch.randn(n_samples, z_dim, device=device)


def combine_vectors(x:torch.Tensor,y:torch.Tensor) -> torch.Tensor:
    '''

    Parameters
    ----------
    x : torch.Tensor
        DESCRIPTION. first tensor to be concatenated
    y : torch.Tensor
        DESCRIPTION. second tensor to be concatenated

    Returns
    -------
    TYPE
        DESCRIPTION. concatenation of the two tensors

    '''
    return torch.cat((x.float(),y.float()),1)



def get_one_hot_labels(labels:torch.Tensor,n_classes:int,shape:Tuple = (1,1))->torch.Tensor:
    '''

    Parameters
    ----------
    labels : torch.Tensor
        DESCRIPTION. List of labels that are present in the current batch (there might not be all the classes)
    n_classes : int
        DESCRIPTION. Number of total classes (regardless if are or not present in the batch)
    shape : Tuple, optional
        DESCRIPTION. The default is (1,1). Shape of the one hot encoding result (depending if is for gen or disc)

    Returns
    -------
    one_hot_encodings : TYPE
        DESCRIPTION.

    '''
    
    ohe = F.one_hot(labels,n_classes)
    
    if shape != (1,1):
        # Reshape the tensor to (batch_size, n_classes, 1, 1)
        ohe = ohe.unsqueeze(-1).unsqueeze(-1)
        # Repeat the tensor along the last two dimensions to create nxn grids
        ohe = ohe.repeat(1, 1, shape[0], shape[1])
    
    return ohe



def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28), nrow=5, show=True):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    if show:
        plt.show()
    
        
        



