# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 10:09:41 2021

@author: mob
"""

import numpy as np
import torch
import pickle
from torch.nn.functional import sigmoid
from lung_dataset import *
from lung_network import *
from tqdm import tqdm

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
folder = r"C:\Users\vamsi\Documents\AI\data\dataset\Lung2D\training\\"
epochs = 100
model = r2u_net()
 
model = model.cuda()
ds = lungTrainDataset(folder) 
dl = getDataLoader(ds)
val_ds = lungValDataset(folder)
val_dl = getDataLoader(val_ds)

lr = 0.0002
opt = torch.optim.Adam(model.parameters(),lr=lr)


q = input("Make sure that you have changed the PKL and model file names:\n")

def DL(y_preds,y):
    loss_fn = torch.nn.BCELoss()
    return loss_fn(y_preds,y)


def get_DC(SR,GT,threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT == torch.max(GT)
    Inter = torch.sum((SR&GT)==1)
    #print(Inter)
    DC = float(2*Inter)/(float(torch.sum(SR)+torch.sum(GT)) + 1e-6)

    return DC

def get_accuracy(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR==GT)
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr)/float(tensor_size)

    return acc


def loss_batch(model,loss_fn,img_b,mask_b,opt=None,metric=None):
    img_b = img_b
    img_b = img_b.cuda()
    mask_b = mask_b.cuda()
    pred_mask = model(img_b)
    #pred_mask = sigmoid(pred_mask)
    mask_b = mask_b/255
    loss = loss_fn(pred_mask,mask_b)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    
    metric_result = None
    
    if metric is not None:
        metric_result = metric(pred_mask,mask_b)
    #print(metric_result)     
    return loss.item(),metric_result
    #return pred_mask,mask_b

def evaluate(model,loss_fn,valid_dl,metric=None):
    with torch.no_grad():
        results = [loss_batch(model,loss_fn,img_b_val,mask_b_val,metric=metric) for img_b_val,mask_b_val in tqdm(valid_dl)]
        losses,metrics = zip(*results)
        # loss = np.average(losses)
        # metric_result = None
        # if metric is not None:
        #     metric_result = np.average(metrics)
        return losses,metrics

def fit(model,epochs,train_dl,valid_dl,loss_fn,opt=None,metric=None):
    lss_train_epoch = []
    metric_train_epoch = []
    lss_val_epoch = []
    metric_val_epoch = []
    for epoch in tqdm(range(epochs)):
        lss_train_batch = []
        metric_train_batch = []
        # lss_val_batch = []
        # metric_val_batch = []
        for img_b,mask_b in tqdm(train_dl):
            #print("Epoch: {} Shape:{} ".format(epoch,str(img_b.shape)))
            lss,metric_res = loss_batch(model,loss_fn,img_b,mask_b,opt=opt,metric=metric)
            print("Batch Loss: {:.4f} Accuracy: {:.4f}".format(lss,metric_res))
            lss_train_batch.append(lss)
            metric_train_batch.append(metric_res)
        lss_train_epoch.append(lss_train_batch)
        metric_train_epoch.append(metric_train_batch)
        val_loss,val_metric = evaluate(model,loss_fn,valid_dl,metric=metric)
        #print("Train Loss: {:.4f} Train Metric: {:.4f} Val Loss: {:.4f} Val Metric: {:.4f}".format(lss,metric_res,val_loss,val_metric))
        lss_val_epoch.append(val_loss)
        metric_val_epoch.append(val_metric)
        a,b,c,d = lss_train_epoch,metric_train_epoch,lss_val_epoch,metric_val_epoch
        fin = [a,b,c,d]
        file_name = "r2u_net_lung.pkl"
        open_file = open(file_name,"wb")
        pickle.dump(fin,open_file)
        open_file.close()
        torch.save({'model_state_dict':model.state_dict(),'opt_state_dict':opt.state_dict(),'epoch':epoch},"r2u_net_lung_model")

    #return lss_train_epoch,metric_train_epoch,lss_val_epoch,metric_val_epoch
fit(model,epochs,dl,val_dl,DL,opt=opt,metric=get_accuracy)