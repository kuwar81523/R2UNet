# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 15:19:29 2021

@author: mob
"""

import torch
import os
import tifffile as tiff
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from random import randint
import torchvision.transforms as transforms

class lungTrainDataset(Dataset):
    def __init__(self,folder):
        self.folder = folder
        self.image_folder = self.folder+"\\2d_images\\train\\"
        self.mask_folder = self.folder+"\\2d_masks\\train\\"
        self.file_list = os.listdir(self.image_folder)
        self.length = len(self.file_list)

    def __len__(self):
        return self.length
    def __getitem__(self,idx):
        img = torch.tensor(tiff.imread(self.image_folder+self.file_list[idx]))
        mask = torch.tensor(tiff.imread(self.mask_folder+self.file_list[idx]))
        img = img.float()
        mask = mask.float()
        img = img.unsqueeze(0)
        mask = mask.unsqueeze(0)
        img = transforms.Resize(256)(img)
        mask = transforms.Resize(256)(mask)
        return data_aug(img,mask)
    
class lungValDataset(Dataset):
    def __init__(self,folder):
        self.folder = folder
        self.image_folder = self.folder+"\\2d_images\\val\\"
        self.mask_folder = self.folder+"\\2d_masks\\val\\"
        self.file_list = os.listdir(self.image_folder)
        self.length = len(self.file_list)

    def __len__(self):
        return self.length
    def __getitem__(self,idx):
        img = torch.tensor(tiff.imread(self.image_folder+self.file_list[idx]))
        mask = torch.tensor(tiff.imread(self.mask_folder+self.file_list[idx]))
        img = img.float()
        mask = mask.float()
        img = img.unsqueeze(0)
        mask = mask.unsqueeze(0)
        img = transforms.Resize(256)(img)
        mask = transforms.Resize(256)(mask)
        return img,mask

    
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

def getDataLoader(dataset):
    return DataLoader(dataset,batch_size=16,shuffle=True)