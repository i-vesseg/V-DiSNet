"""
File name: pnetcls_PT.py
Author: Hugo Rechatin
Date created: March 7, 2022

This file defines the pnetcls architecture. 
"""

import torch
import torch.nn as nn
import numpy as np

def conv_block(layer_in, n_filters, n_conv, dilated_rate, kernel_size):
    for _ in range(n_conv):
        layer_in = nn.functional.relu(nn.Conv2d(in_channels=layer_in.shape[1],
                                                out_channels=n_filters, 
                                                kernel_size = kernel_size,
                                                padding='same',
                                                dilation=dilated_rate)(layer_in))
    return layer_in

class Pnet(nn.Module): 
    def __init__(self, patch_size = 96):
        
        super(Pnet, self).__init__()
        
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.functional.relu
        self.sigmoid = torch.sigmoid
        self.flatten = torch.flatten
        self.conv1 = nn.Conv2d(320, 128, kernel_size = (1,1), dilation = 1)
        self.conv2 = nn.Conv2d(128, 1, kernel_size = (1,1), dilation = 1)
        self.fc1 = nn.Linear(patch_size**2, 128)
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, inputs):
        
        layer1 = conv_block(inputs, 64, 2, 1, 3)
        layer2 = conv_block(layer1, 64, 2, 2, 3)
        layer3 = conv_block(layer2, 64, 3, 4, 3)
        layer4 = conv_block(layer3, 64, 3, 8, 3)
        layer5 = conv_block(layer4, 64, 3, 16, 3)
        
        # concatenate 5 blocks
        x = torch.cat((layer1,layer2,layer3,layer4,layer5), 1)
        
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        
        x = self.relu(self.conv2(x))
        x = self.flatten(x, start_dim = 2, end_dim = 3) ##Make sure to flatten the samples and not the entier batch
        
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        
        return x[:,0,:] #fix the channel (there is only one) to avoid dimension issues with the loss when compared with the 
                        #label tensor (otherwise x.shape = [1,1,1] != label.shape = [1,1] )
    