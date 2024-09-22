#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 20:16:26 2024

@author: francesco
"""

#import section
import os
import pathlib
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple, Dict, List

''' 
1) we want to write a function to 
        - get the classes from the directory where the data is stored
                    ['pizza', 'steak', 'sushi']
        - turn the class into a dictionary of index
                    {'pizza':0, 'steak':1, 'sushi':2}
        
'''

#set up the directory where the data are sotred
train_dir = "images/train"



class ImageFolderCustom(Dataset):
    
    '''
    THIS CLASS HAS TO CONTAIN THE FOLLOWING ATTRIBUTES:
        1) paths: paths to our images
        2) transform:  ..... capire
        3) classes: list of target classes
        4) class_to_idx: mapping between class names and class indices
    
    THIS CLASS HAS TO COINTAIN THE FOLLOWING FUNCTIONS:
        1) load_images(), to open a single image
        2) __len()__ to return the len of out dataset (this function is 
        suggested to be created by pytorch documentation to overwrite the 
        original __len()__ function)
        3) __getitem__(), to return a given sample when passed an index
                                                       
    '''
    
    ### STANDARD INIT METHOD. IT STORES THE DATA PATH
    def __init__(self, 
                 targ_dir:str, 
                 transform=None):
        
        super().__init__()
        
        ### OBTAIN ALL IMAGES FOLDERS ###
        #from the training folde mathc all the images (in each subfolder) that ends with ".jpg"
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg"))
        
        ### SETUP TRANSFORM ###
        #take it from the argument passed to the class
        self.transform = transform 
        
        ### CREATE classes AND class_to_idx ATTRIBUTES
        self.classes, self.class_to_idx = self.find_classes(targ_dir)
        
    ### FUNCTION TO RETURN LIST OF CLASSES AND DICTIONARY NAMES -> INDEX FOR EACH CLASS
    def find_classes(self, directory: str) -> Tuple[List[str],Dict[str,int]]:
        #get list of classes
        class_names = [item.name for item in list(os.scandir(directory)) if item.is_dir()]
        
        #error checking code
        if not class_names:
            raise FileNotFoundError(f"couldn't find any class in {directory}. Please check file structure")
            
        #copy list and make it dictionary
        class_to_idx = {class_name:i for i,class_name in enumerate(class_names)}
        
        return class_names, class_to_idx
    
    ### FUNCTION TO RETURN AN IMAGE GIVEN AN INDEX
    def load_image(self, index:int) -> Image.Image: #THIS RETURNS AN IMAGE (we will call this method in __getitem__())
        #referece the path previously loaded in the init method in self.path and index over them
        image_path = self.paths[index]
        #return the image related to the indexed path
        return Image.open(image_path).convert("RGB") #images have to be converted from CMYK to RGB
    
    ### FUNCTION TO RETURN THE LEN OF OUT DATASET
    # overwriting __len__() method
    def __len__(self) -> int:
        return len(self.paths)
    
    ### FUNCTION TO RETURN A SPECIFIC ITEM FROM THE DATASET
    #overvriting __getitem__() mehtod to return a particular sample
    def __getitem__(self, index:int) -> Tuple[torch.Tensor, int]: #THIS RETURNS A TENSOR AND ITS CLASS
        #get the image
        img = self.load_image(index) 
        #get the class
        class_name = self.paths[index].parent.name #this takes the name of the parent folder of the image
        class_idx = self.class_to_idx[class_name]
        
        #transform if necessary
        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx

    
    
    