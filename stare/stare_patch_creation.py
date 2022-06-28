# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 16:31:20 2021

@author: mob
"""

from PIL import Image
from tqdm import tqdm

import tifffile as tiff
import os
import numpy as np
import shutil
import torch


def rm_mk_folders(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def make_folders_stare():   
    dataset_folder = r"C:\Users\vamsi\Documents\AI\data\dataset\STARE\\"
    
    training_images_folder = os.path.join(dataset_folder,"stare-images\\")
    training_gt_folder = os.path.join(dataset_folder,"labels-ah\\")
    
    patch_image_folder = r"C:\Users\vamsi\Documents\AI\data\dataset\STARE\training\image_patches\\"
    patch_mask_folder = r"C:\Users\vamsi\Documents\AI\data\dataset\STARE\\training\mask_patches\\"
    
    rm_mk_folders(patch_image_folder)
    rm_mk_folders(patch_mask_folder)

    img_list = os.listdir(training_images_folder)
    patch_size = 48
    total_patches = 100000
    no_of_images = len(img_list)
    patches_per_image = int(total_patches/no_of_images)
    
    count = 1
    
    
    for i,img in tqdm(enumerate(img_list)):
        
        image = Image.open(training_images_folder+img)
        mask = Image.open(training_gt_folder+"\\"+img.split(".")[0]+".ah.ppm")
        w,h = image.size
        
        for patch in tqdm(range(patches_per_image)):
            #print(img+"_"+str(patch))
            x,y = np.random.randint(0,w-patch_size),np.random.randint(0,h-patch_size)
            image_patch = image.crop((x,y,x+patch_size,y+patch_size))
            mask_patch = mask.crop((x,y,x+patch_size,y+patch_size))
            image_patch.save(patch_image_folder+"\\"+str(count)+"_patch.jpg")
            mask_patch.save(patch_mask_folder+"\\"+str(count)+"_mask.jpg")
            count+=1

def split_folders(folder,ratio=(0.80,0.1,0.1)):
    '''
    folder contains both image_patches folder and mask_patches folder
    
    '''
    train_img_folder = folder+"\\image_patches\\train\\"
    train_mask_folder = folder+"\\mask_patches\\train\\"
    val_img_folder = folder+"\\image_patches\\val\\"
    val_mask_folder = folder+"\\mask_patches\\val\\"
    test_img_folder = folder+"\\image_patches\\test\\"
    test_mask_folder = folder+"\\mask_patches\\test\\"
       
    rm_mk_folders(train_img_folder)
    rm_mk_folders(train_mask_folder)
    rm_mk_folders(val_img_folder)
    rm_mk_folders(val_mask_folder)
    rm_mk_folders(test_img_folder)
    rm_mk_folders(test_mask_folder)
    
    l = len(os.listdir(folder+"\\image_patches\\"))-3
    l_perm = np.array(torch.randperm(l))+1
    
    train_list = l_perm[:int(l*ratio[0])]
    val_list = l_perm[int(l*ratio[0]):int(l*ratio[0]+l*ratio[1])]
    test_list = l_perm[int(l*ratio[0]+l*ratio[1]):]    
    
    
    try:
        for train_idx in tqdm(train_list):
            os.replace(folder+"\\image_patches\\"+str(train_idx)+"_patch.jpg",
                       folder+"\\image_patches\\train\\"+str(train_idx)+"_patch.jpg")
            os.replace(folder+"\\mask_patches\\"+str(train_idx)+"_mask.jpg",
                       folder+"\\mask_patches\\train\\"+str(train_idx)+"_mask.jpg")
    except:
        return train_idx,train_list
    
    try:
        for val_idx in tqdm(val_list):
            os.replace(folder+"\\image_patches\\"+str(val_idx)+"_patch.jpg",
                       folder+"\\image_patches\\val\\"+str(val_idx)+"_patch.jpg")
            os.replace(folder+"\\mask_patches\\"+str(val_idx)+"_mask.jpg",
                       folder+"\\mask_patches\\val\\"+str(val_idx)+"_mask.jpg")
    except:
        return val_idx,val_list
    
    try:        
        for test_idx in tqdm(test_list):
            os.replace(folder+"\\image_patches\\"+str(test_idx)+"_patch.jpg",
                       folder+"\\image_patches\\test\\"+str(test_idx)+"_patch.jpg")
            os.replace(folder+"\\mask_patches\\"+str(test_idx)+"_mask.jpg",
                       folder+"\\mask_patches\\test\\"+str(test_idx)+"_mask.jpg")   
    except:
        return test_idx,test_list

