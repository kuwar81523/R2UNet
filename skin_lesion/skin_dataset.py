# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 15:19:29 2021

@author: mob
"""

import torch
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from random import randint
from torchvision.io import read_image
import torchvision.transforms as transforms

class skinTrainDataset(Dataset):
    def __init__(self,folder):
        self.folder = folder
        self.image_folder = self.folder+"\\images\\train\\"
        self.mask_folder = self.folder+"\\masks\\train\\"
        self.file_list = os.listdir(self.image_folder)
        self.length = len(self.file_list)

    def __len__(self):
        return self.length
    def __getitem__(self,idx):
        img = read_image(self.image_folder+self.file_list[idx])
        mask = read_image(self.mask_folder+self.file_list[idx].split(".")[0]+"_Segmentation.png")
        img = img.float()
        mask = mask.float()
        # img = img.unsqueeze(0)
        # mask = mask.unsqueeze(0)
        img = transforms.Resize((128,128))(img)
        mask = transforms.Resize((128,128))(mask)
        return data_aug(img,mask)
    
class skinValDataset(Dataset):
    def __init__(self,folder):
        self.folder = folder
        self.image_folder = self.folder+"\\images\\val\\"
        self.mask_folder = self.folder+"\\masks\\val\\"
        self.file_list = os.listdir(self.image_folder)
        self.length = len(self.file_list)

    def __len__(self):
        return self.length
    def __getitem__(self,idx):
        img = read_image(self.image_folder+self.file_list[idx])
        mask = read_image(self.mask_folder+self.file_list[idx].split(".")[0]+"_Segmentation.png")
        img = img.float()
        mask = mask.float()
        # img = img.unsqueeze(0)
        # mask = mask.unsqueeze(0)
        img = transforms.Resize((128,128))(img)
        mask = transforms.Resize((128,128))(mask)
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
    return DataLoader(dataset,batch_size=32,shuffle=True)