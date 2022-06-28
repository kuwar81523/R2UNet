# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 21:57:08 2021

@author: mob
"""

import torch
import torch.nn as nn

from torchsummary import summary

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
                        nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
                        nn.BatchNorm2d(ch_out),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
                        nn.BatchNorm2d(ch_out),
                        nn.ReLU(inplace=True))
                        
    
    def forward(self,x):
        x = self.conv(x)
        return x
    

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up_sample = nn.Sequential(
                            nn.ConvTranspose2d(ch_in,ch_out,kernel_size=2,stride=2),
                            nn.BatchNorm2d(ch_out),
                            nn.ReLU(inplace=True))
    
    def forward(self,x):
        x = self.up_sample(x)
        return x

class recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True))
    
    def forward(self,x):
        for i in range(self.t):
            if i==0:
                x1 = self.conv(x)
            x1 = self.conv(x+x1)
        return x1

class rrcnn_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(rrcnn_block,self).__init__()
        self.rcnn = nn.Sequential(recurrent_block(ch_out,t),recurrent_block(ch_out,t))
        self.conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)
    
    def forward(self,x):
        x = self.conv_1x1(x)
        x1 = self.rcnn(x)
        return x+x1
    
class u_net(nn.Module):
    def __init__(self,img_ch=3,out_ch=1):
        super(u_net,self).__init__()
        
        self.max = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.conv_1 = conv_block(3,16)
        
        self.conv_2 = conv_block(16,32)
        
        self.conv_3 = conv_block(32,64)
        
        self.conv_4 = conv_block(64,128)
        
        self.conv_2_1x1 = nn.Conv2d(16,16,kernel_size=1,stride=1,padding=0)
        
        self.conv_3_1x1 = nn.Conv2d(32,32,kernel_size=1,stride=1,padding=0)
        
        self.conv_4_1x1 = nn.Conv2d(64,64,kernel_size=1,stride=1,padding=0)
        
        
        self.up_samp_4 = up_conv(128,64)
        
        self.up_conv_4 = conv_block(128,64)
        
        self.up_samp_3 = up_conv(64,32)
        
        self.up_conv_3 = conv_block(64,32)
        
        self.up_samp_2 = up_conv(32,16)
        
        self.up_conv_2 = conv_block(32,16)
        
        self.conv_1x1 = nn.Conv2d(16,out_ch,kernel_size=1,stride=1,padding=0)
        
        self.sigmoid = nn.Sigmoid()

        
    def forward(self,x):
        
        x1 = self.conv_1(x)
        # 48*48*16
        x2 = self.max(x1)
        #24*24*16
        x2 = self.conv_2(x2)
        #24*24*32
        x3 = self.max(x2)
        #12*12*32
        x3 = self.conv_3(x3)
        #12*12*64
        x4 = self.max(x3)
        #6*6*64
        x4 = self.conv_4(x4)
        #6*6*128
        d4 = self.up_samp_4(x4)
        #12*12*64
        d4 = torch.cat((x3,d4),dim=1)
        #12*12*128
        d4 = self.up_conv_4(d4)
        #12*12*64
        d3 = self.up_samp_3(d4)
        #24*24*32
        d3 = torch.cat((x2,d3),dim=1)
        #24*24*64
        d3 = self.up_conv_3(d3)
        #24*24*32
        d2 = self.up_samp_2(d3)
        #48*48*16
        d2 = torch.cat((x1,d2),dim=1)
        #48*48*32
        d2 = self.up_conv_2(d2)
        #48*48*16
        d1 = self.conv_1x1(d2)
        #48*48*3
        d1 = self.sigmoid(d1)
        
        return d1

    
class r2u_net(nn.Module):
    def __init__(self,img_ch=3,out_ch=1):
        super(r2u_net,self).__init__()
        self.max = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.batch_norm = nn.BatchNorm2d(3)
        
        self.rrcnn1 = rrcnn_block(3,16)
        
        self.rrcnn2 = rrcnn_block(16,32)
        
        self.rrcnn3 = rrcnn_block(32,64)
        
        self.rrcnn4 = rrcnn_block(64,128)
        
        self.up_samp_4 = up_conv(128,64)
        
        self.up_conv_4 = rrcnn_block(128,64)
        
        self.up_samp_3 = up_conv(64,32)
        
        self.up_conv_3 = rrcnn_block(64,32)
        
        self.up_samp_2 = up_conv(32,16)
        
        self.up_conv_2 = rrcnn_block(32,16)

        self.conv_1x1 = nn.Conv2d(16,out_ch,kernel_size=1,stride=1,padding=0)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        x = self.batch_norm(x)
        
        x1 = self.rrcnn1(x)
        
        x2 = self.max(x1)
        
        x2 = self.rrcnn2(x2)
        
        x3 = self.max(x2)
        
        x3 = self.rrcnn3(x3)
        
        x4 = self.max(x3)
        
        x4 = self.rrcnn4(x4)
        
        d4 = self.up_samp_4(x4)
        
        d4 = torch.cat((x3,d4),dim=1)
        
        d4 = self.up_conv_4(d4)
        
        d3 = self.up_samp_3(d4)
        
        d3 = torch.cat((x2,d3),dim=1)
        
        d3 = self.up_conv_3(d3)
        
        d2 = self.up_samp_2(d3)
        
        d2 = torch.cat((x1,d2),dim=1)
        
        d2 = self.up_conv_2(d2)
        
        d1 = self.conv_1x1(d2)
        
        d1 = self.sigmoid(d1)
        
        return d1