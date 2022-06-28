# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 16:58:14 2021

@author: mob
"""

import torch
import os
from random import randint
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F


class chaseTrainDataset(Dataset):
    
    def __init__(self,folder,mask="mask_1"):
        self.mask = mask
        self.train_img_list = os.listdir(folder+"\\image_patches\\train\\")
        self.train_mask_list = os.listdir(folder+"\\"+self.mask+"_patches\\train\\")
        self.length = len(self.train_img_list)
        self.indexes = torch.randperm(self.length)
        self.folder = folder
    def __len__(self):
        return self.length
    
    def __getitem__(self,idx):
        img_name = self.train_img_list[self.indexes[idx]]
        mask_name = self.train_mask_list[self.indexes[idx]]
        img = read_image(self.folder+"\\image_patches\\train\\"+img_name)
        mask = read_image(self.folder+"\\"+self.mask+"_patches\\train\\"+mask_name)
        img = img.float()
        mask = mask.float()
        return data_aug(img,mask)

class chaseValDataset(Dataset):
    
    def __init__(self,folder,mask="mask_1"):
        self.mask = mask
        self.val_img_list = os.listdir(folder+"\\image_patches\\val\\")
        self.val_mask_list = os.listdir(folder+"\\"+self.mask+"_patches\\val\\")
        self.length = len(self.val_img_list)
        self.indexes = torch.randperm(self.length)
        self.folder = folder
    def __len__(self):
        return self.length
    
    def __getitem__(self,idx):
        img_name = self.val_img_list[self.indexes[idx]]
        mask_name = self.val_mask_list[self.indexes[idx]]
        img = read_image(self.folder+"\\image_patches\\val\\"+img_name)
        mask = read_image(self.folder+"\\"+self.mask+"_patches\\val\\"+mask_name)
        img = img.float()
        mask = mask.float()
        return img,mask

def getDataLoader(dataset):
    return DataLoader(dataset,batch_size=64,shuffle=True)

def data_aug(img,mask):
    i = randint(1,4)
    if i==1:
        img = F.vflip(img)
        mask = F.vflip(mask)
    elif i==2:
        img = F.hflip(img)
        mask = F.hflip(mask)
    elif i==3:
        img = F.hflip(img)
        img = F.vflip(img)
        mask = F.hflip(mask)
        mask = F.vflip(mask)
    else:
        None
    return img,mask