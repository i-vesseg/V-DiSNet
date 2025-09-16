#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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


class Wnet(nn.Module):
    
    
    def __init__(self):
        super(Wnet, self).__init__() 
        
    # 1st Unet
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size =(2), stride =2) 
        
        self.down_conv_1_1 = double_conv(1,64)
        self.down_conv_1_2 = double_conv(64,128)
        self.down_conv_1_3 = double_conv(128,256)
        self.down_conv_1_4 = double_conv(256,512)
        
        self.up_trans_1_1 = nn.ConvTranspose2d(in_channels=512, 
                                             out_channels=256, 
                                             kernel_size=2, 
                                             stride = 2)
        
        self.up_conv_1_1 =  double_conv(512,256)

        self.up_trans_1_2 = nn.ConvTranspose2d(in_channels=256, 
                                             out_channels=128, 
                                             kernel_size=2, 
                                             stride = 2)
        
        self.up_conv_1_2 =  double_conv(256,128)

        self.up_trans_1_3 = nn.ConvTranspose2d(in_channels=128, 
                                             out_channels=64, 
                                             kernel_size=2, 
                                             stride = 2)
        
        self.up_conv_1_3 =  double_conv(128,64)
        
        self.final_conv_1 = nn.Conv2d(in_channels=64,
                                    out_channels=1,
                                    kernel_size = 1)
        
    # 2nd Unet

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size =2, stride =2) 
        self.max_pool_2x2_padd = nn.MaxPool2d(kernel_size =2, stride =2,padding=1)         
        
        self.down_conv_2_1 = double_conv(1,64)
        self.down_conv_2_2 = double_conv(64,128)
        self.down_conv_2_3 = double_conv(128,256)
        self.down_conv_2_4 = double_conv(256,512)
        
        self.up_trans_2_1 = nn.ConvTranspose2d(in_channels=512, 
                                              out_channels=256, 
                                              kernel_size=2, 
                                              stride = 2)
        
        self.up_conv_2_1 =  double_conv(768,256)
        
        self.up_trans_2_2 = nn.ConvTranspose2d(in_channels=256, 
                                              out_channels=128, 
                                              kernel_size=2, 
                                              stride = 2)
        
        self.up_conv_2_2 =  double_conv(384,128)

        self.up_trans_2_3 = nn.ConvTranspose2d(in_channels=128, 
                                              out_channels=43, 
                                              kernel_size=2, 
                                              stride = 2)
        
        self.up_conv_2_3 =  double_conv(128,64)

        self.final_conv_2 = nn.Conv2d(in_channels=64,
                                    out_channels=1,
                                    kernel_size = 1)
        
        self.final_conv_3 = nn.Sigmoid()
    
        
        
        
    def forward(self,image):
        #encoder 1st unet
        x1  = self.down_conv_1_1(image)   #this output goes to the second part of the network
        x2  = self.max_pool_2x2(x1)
        x3  = self.down_conv_1_2(x2)      #this one also 
        x4  = self.max_pool_2x2(x3)
        x5  = self.down_conv_1_3(x4)      #this one also 
        x6  = self.max_pool_2x2(x5)   
        x7  = self.down_conv_1_4(x6)      #that one also 

        
        #decoder 1st unet
        #we only need x1,x3,x5 for the decoding part of the network 
        x = self.up_trans_1_1(x7)
        y5 = crop_img(x5, x)
        x55 = self.up_conv_1_1(torch.cat([x,y5],1))
        
        x = self.up_trans_1_2(x55)
        y3 = crop_img(x3, x)
        x33 = self.up_conv_1_2(torch.cat([x,y3],1))
        
        x = self.up_trans_1_3(x33)
        y1 = crop_img(x1, x)
        x1_1 = self.up_conv_1_3(torch.cat([x,y1],1))
        
        x1_1 = self.final_conv_1(x1_1)

        #encoder 2nd unet
        x8  = self.down_conv_2_1(x1_1)  
        x9  = self.max_pool_2x2(x8)
        
        x10  = self.down_conv_2_2(x9)
        x11  = self.max_pool_2x2(x10)
        
        x12  = self.down_conv_2_3(x11) 
        x13  = self.max_pool_2x2(x12)
        
        x14  = self.down_conv_2_4(x13) 

   
        #decoder 2nd unet 
        x = self.up_trans_2_1(x14)
        y1 = crop_img(x5,x)
        y2 = crop_img(x12, x)
        
        xcon = (torch.cat([y1,y2],1))
        
        x = self.up_conv_2_1(torch.cat([x,xcon],1))        
        x = self.up_trans_2_2(x)
        y1 = crop_img(x3, x)
        y2 = crop_img(x10,x)
        
        xcon = (torch.cat([y1,y2],1))
        
        x = self.up_conv_2_2(torch.cat([x,xcon],1))
        x = self.up_trans_2_3(x)
        y1 = crop_img(x1, x)
        y2 = crop_img(x8, x)
        
        xcon = (torch.cat([y1,y2],1))
        
        x = self.up_conv_2_3(torch.cat([y1,y2],1))
        x = self.final_conv_2(x)
        x = self.final_conv_3(x)
        
        return x 
       
    

