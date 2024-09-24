"""
File name: pnetcls_PT.py
Author: Hugo Rechatin
Date created: March 7, 2022

This file defines the pnetcls architecture. 
"""

import torch
import torch.nn as nn
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, n_filters, n_conv, dilated_rate, kernel_size, first_layer=False):
        super(ConvBlock, self).__init__()
        
        if first_layer:
            self.conv_layers = nn.ModuleList([nn.Conv2d(1 if i==0 else n_filters, n_filters, kernel_size=kernel_size, padding='same', dilation=dilated_rate) for i in range(n_conv)])
        else:
            self.conv_layers = nn.ModuleList([nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, padding='same', dilation=dilated_rate) for i in range(n_conv)])
        self.relu = nn.ReLU()

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            x = self.relu(x)
        return x
    

class PnetCls(nn.Module): 
    def __init__(self, patch_size = 32, n_classes = 1, dropout = 0.5):
        
        super(PnetCls, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        self.relu = nn.functional.relu
        self.sigmoid = torch.sigmoid
        self.flatten = torch.flatten
        
        self.layer1 = ConvBlock(n_filters=64, n_conv=2, dilated_rate=1,  kernel_size=3, first_layer=True)
        self.layer2 = ConvBlock(n_filters=64, n_conv=2, dilated_rate=2,  kernel_size=3)
        self.layer3 = ConvBlock(n_filters=64, n_conv=3, dilated_rate=4,  kernel_size=3)
        self.layer4 = ConvBlock(n_filters=64, n_conv=3, dilated_rate=8,  kernel_size=3)
        self.layer5 = ConvBlock(n_filters=64, n_conv=3, dilated_rate=16, kernel_size=3)
        
        self.conv1 = nn.Conv2d(320, 128, kernel_size = (1,1), dilation = 1)
        self.conv2 = nn.Conv2d(128, 1, kernel_size = (1,1), dilation = 1)
        
        self.fc1 = nn.Linear(patch_size**2, 128)
        self.fc2 = nn.Linear(128, n_classes)
    
        
    def forward(self, inputs):
        
        #print(f"Forwarding Input: {inputs.shape}")
        layer1 = self.layer1(inputs)
        #print(f"Layer 1: {layer1.shape}")
        layer2 = self.layer2(layer1)
        #print(f"Layer 2: {layer2.shape}")
        layer3 = self.layer3(layer2)
        #print(f"Layer 3: {layer3.shape}")
        layer4 = self.layer4(layer3)
        #print(f"Layer 4: {layer4.shape}")
        layer5 = self.layer5(layer4)
        #print(f"Layer 5: {layer5.shape}")
        
        # concatenate 5 blocks
        x = torch.cat((layer1,layer2,layer3,layer4,layer5), 1)
        x = self.relu(x)
        x = self.dropout(x)
        #print(f"\nConcatenated: {x.shape}")
        
        # First convolutional layer
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        #print(f"Conv1: {x.shape}")
        
        # Second convolutional layer
        x = self.conv2(x)
        x = self.relu(x)
        #print(f"Conv2: {x.shape}")
        
        x = self.flatten(x, start_dim = 2, end_dim = 3) ##Make sure to flatten the samples and not the entier batch
        #print(f"\nFlatten: {x.shape}")
        
        # First fully connected layer
        x = self.fc1(x)
        x = self.relu(x)
        #print(f"\nFC1: {x.shape}")
        
        # Last fully connected layer
        x = self.fc2(x)
        x = self.sigmoid(x)
        #print(f"FC2: {x.shape}")
        
        
        return x[:,0,:] #fix the channel (there is only one) to avoid dimension issues with the loss when compared with the 
                        #label tensor (otherwise x.shape = [1,1,1] != label.shape = [1,1] )
    