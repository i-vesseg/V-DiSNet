#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 14:58:34 2022

@author: yannis_pitsiorlas
"""

import torch 
import torch.nn as nn 


def double_conv(in_c, out_c, padding='same'):
    conv = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size = 3, padding=padding),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size = 3, padding=padding),
                nn.ReLU(inplace = True) 
                )
    return conv

#crop images to the correct dimensions for each concatenation during the forward 
def crop_img(tensor,target_tensor):
    
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    
    delta = (tensor_size - target_size) // 2
    
    return tensor[:, :, delta:tensor_size-delta,delta:tensor_size-delta]

        
class UNet(nn.Module):
    
    def __init__(self):
        super(UNet, self).__init__() 
        
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size =2, stride =2) 
        
        self.down_conv_1 = double_conv(1,64)
        self.down_conv_2 = double_conv(64,128)
        self.down_conv_3 = double_conv(128,256)
        self.down_conv_4 = double_conv(256,512)
        self.down_conv_5 = double_conv(512,1024)
        
        self.up_trans_1 = nn.ConvTranspose2d(in_channels=1024, 
                                             out_channels=512, 
                                             kernel_size=2, 
                                             stride = 2)
        
        self.up_conv_1 =  double_conv(1024,512)
        
        self.up_trans_2 = nn.ConvTranspose2d(in_channels=512, 
                                             out_channels=256, 
                                             kernel_size=2, 
                                             stride = 2)
        
        self.up_conv_2 =  double_conv(512,256)

        self.up_trans_3 = nn.ConvTranspose2d(in_channels=256, 
                                             out_channels=128, 
                                             kernel_size=2, 
                                             stride = 2)
        
        self.up_conv_3 =  double_conv(256,128)

        self.up_trans_4 = nn.ConvTranspose2d(in_channels=128, 
                                             out_channels=64, 
                                             kernel_size=2, 
                                             stride = 2)
        
        self.up_conv_4 =  double_conv(128,64)
        
        self.out = nn.Conv2d(
            in_channels=64,
            out_channels=1,
            kernel_size = 1)
        
        
        
    def forward(self,image):
        #encoder 
        x1  = self.down_conv_1(image) #this output goes to the second part of the network
        x2  = self.max_pool_2x2(x1)        
        
        x3  = self.down_conv_2(x2)    #this one also 
        x4  = self.max_pool_2x2(x3)
        
        x5  = self.down_conv_3(x4)    #this one also 
        x6  = self.max_pool_2x2(x5)   
        
        x7  = self.down_conv_4(x6)    #that one also 
        x8  = self.max_pool_2x2(x7)
        
        x9  = self.down_conv_5(x8)
      
        #decoder
        #we only need x1,x3,x5 for the decoding part of the network 
        
        x = self.up_trans_1(x9)
        y7 = crop_img(x7, x)
        x77 = self.up_conv_1(torch.cat([x,y7],1))        
        
        x = self.up_trans_2(x77)
        y5 = crop_img(x5, x)
        x55 = self.up_conv_2(torch.cat([x,y5],1))
        
        x = self.up_trans_3(x55)
        y3 = crop_img(x3, x)
        x33 = self.up_conv_3(torch.cat([x,y3],1))
        
        x = self.up_trans_4(x33)
        y1 = crop_img(x1, x)
        x = self.up_conv_4(torch.cat([x,y1],1))
        
        x = self.out(x)
        return x 
        




































