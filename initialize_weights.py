# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 12:08:49 2021

@author: mob
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

n = 200
gain=1

# Normal Weights
w_normal = torch.empty(n,n)
nn.init.normal_(w_normal,mean=0,std=gain)
w_normal = torch.flatten(w_normal)

# Xavier
w_xavier = torch.empty(n,n)
nn.init.xavier_normal_(w_xavier,gain=gain)
w_xavier = torch.flatten(w_xavier)

# kaiming
w_kaiming = torch.empty(n,n)
nn.init.kaiming_normal_(w_kaiming,a=0,mode='fan_in') 
w_kaiming = torch.flatten(w_kaiming)

# orthogonal
w_orthogonal = torch.empty(n,n)
nn.init.orthogonal_(w_orthogonal,gain=gain)
w_orthogonal = torch.flatten(w_orthogonal)

weights = [w_normal,w_xavier,w_kaiming,w_orthogonal]

titles= ["Normal","Xavier","Kaiming","Orthogonal"]

for i,w in enumerate(weights):
    plt.subplot(2,2,i+1)
    (x,y) = torch.unique(w,return_counts=True)
    plt.scatter([i for i in range(len(x))],x)
    plt.title(titles[i])
plt.show()