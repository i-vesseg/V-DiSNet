import os
from matplotlib import pyplot as plt
import torch.nn as nn
import torch

class UpConcatBlock(nn.Module):
    
    """
    Bulding block with up-sampling and concatenation for one level in the first 2D-Unet.
    """
    
    def __init__(self, pool_size, concat_axis):
        super(UpConcatBlock, self).__init__()

        self.up = nn.Upsample(scale_factor=pool_size, mode='bilinear', align_corners=True)
        self.concat_axis = concat_axis

    def forward(self, x1, x2):
        x1 = self.up(x1)
        #print(f"Concat: {x1.shape, x2.shape} along axis {self.concat_axis}")
        out = torch.cat([x1, x2], dim=self.concat_axis)
        return out



class UpConcatBlock2(nn.Module):
    
    """
    Bulding block with up-sampling and concatenation for one level in the second 2D-Unet.
    """
    
    def __init__(self, pool_size, concat_axis):
        super(UpConcatBlock2, self).__init__()

        self.up = nn.Upsample(scale_factor=pool_size, mode='bilinear', align_corners=True)
        self.concat_axis = concat_axis

    def forward(self, x1, x2, x3):
        x1 = self.up(x1)
        #print(f"Concat {x1.shape, x2.shape, x3.shape} along axis {self.concat_axis}")
        out = torch.cat([x1, x2, x3], dim=self.concat_axis)
        return out


class DoublConvBlock(nn.Module):
    
    """
    Bulding block with convolutional layers for one level.
    """
    
    def __init__(self, in_channels, num_kernels, kernel_size, strides, padding, activation, dropout=False, bn=False):
        
        super(DoublConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=num_kernels,
                               kernel_size=kernel_size,stride=strides, padding=padding)
        
        self.activation = activation
        
        self.bn1 = nn.BatchNorm2d(num_kernels) if bn else None
        
        self.dropout = nn.Dropout(p=dropout) if dropout else None
    
        self.conv2 = nn.Conv2d(in_channels=num_kernels, out_channels=num_kernels,
                               kernel_size=kernel_size,stride=strides, padding=padding)
        
        self.bn2 = nn.BatchNorm2d(num_kernels) if bn else None

    
        
    def forward(self, x):
        
        #print("ConvBlock1: ", x.shape)
        x = self.conv1(x)
        
        #print("Activation: ", self.activation)
        x = self.activation(x)
        
        if self.bn1:
            #print("BatchNorm1: ", x.shape)
            x = self.bn1(x)
        
        if self.dropout:
            #print("Dropout: ", x.shape)
            x = self.dropout(x)
        
        #print("ConvBlock2: ", x.shape)
        x = self.conv2(x)
        #print("Activation: ", self.activation)
        
        x = self.activation(x)
        
        if self.bn2:
            #print("BatchNorm2: ", x.shape)
            x = self.bn2(x)
        
        #print("Out: ", x.shape)
        
        return x

class FinalConvBlock(nn.Module):
    
    """
    Bulding block with convolutional layers for one level.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, strides, padding, activation, bn=False):
        
        super(FinalConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size,stride=strides, padding=padding)
        
        self.activation = activation
        
    
        
    def forward(self, x):
        
        #print("ConvBlock1: ", x.shape)
        x = self.conv1(x)
        
        #print("Activation: ", self.activation)
        x = self.activation(x)
        
        #print("Out: ", x.shape)
        
        return x


class WNetSeg(nn.Module):    
    
    """
    Defines the architecture of the wnetseg. 
    """
        
    def __init__(self, patch_size, in_channels, activation, final_activation, kernel_size=(3, 3), pool_size=(2, 2), strides=(1, 1), num_kernels=None, concat_axis=1, padding='same', final_padding = 'same', bn=False,  dropout=False, n_classes=1):
        
        super(WNetSeg, self).__init__()

        if num_kernels is None:
            num_kernels = [64, 128, 256, 512, 1024] # Output channels of each level of the U-net

        
        # The first U-net
        # DOWN-SAMPLING PART (left side of the first U-net)
        # layers on each level: convolution2d -> dropout -> convolution2d -> max-pooling
        # level 0
        self.conv_0_down_1 = DoublConvBlock(in_channels, num_kernels[0], kernel_size, strides, padding, activation, dropout, bn)
        self.pool_0_1 = nn.MaxPool2d(kernel_size=pool_size)
        
        # level 1
        self.conv_1_down_1 = DoublConvBlock(num_kernels[0], num_kernels[1], kernel_size, strides, padding, activation, dropout, bn)
        self.pool_1_1 = nn.MaxPool2d(kernel_size=pool_size)                               
    
        # level 2
        self.conv_2_down_1 = DoublConvBlock(num_kernels[1], num_kernels[2], kernel_size, strides, padding, activation, dropout, bn)
        self.pool_2_1 = nn.MaxPool2d(kernel_size=pool_size)                               
    
        # level 3
        self.conv_3_1 = DoublConvBlock(num_kernels[2], num_kernels[3], kernel_size, strides, padding, activation, dropout, bn)
        
        # UP-SAMPLING PART (right side of the first U-net)
        # layers on each level: upsampling2d -> concatenation with feature maps of corresponding level from down-sampling
    
        # level 2
        self.concat_2_1 = UpConcatBlock(pool_size, concat_axis)
        self.conv_2_up_1 = DoublConvBlock(num_kernels[3]+num_kernels[2], num_kernels[2], kernel_size, strides, padding, activation, dropout,bn)
        
        # level 1
        self.concat_1_1 = UpConcatBlock(pool_size, concat_axis)
        self.conv_1_up_1 = DoublConvBlock(num_kernels[2]+num_kernels[1], num_kernels[1], kernel_size, strides, padding, activation, dropout,bn)

        # level 0
        self.concat_0_1 = UpConcatBlock(pool_size, concat_axis)
        self.conv_0_up_1 = DoublConvBlock(num_kernels[1]+num_kernels[0], num_kernels[0], kernel_size, strides, padding, activation, dropout,bn)
        
        self.final_conv_1 = FinalConvBlock(num_kernels[0], n_classes, 1, strides, padding=final_padding, activation=final_activation)
        
        # The second U-net
        # DOWN-SAMPLING PART (left side of the second U-net)
        
        # level 0
        self.conv_0_down_2 = DoublConvBlock(n_classes, num_kernels[0], kernel_size, strides, padding, activation, dropout, bn)
        self.pool_0_2 = nn.MaxPool2d(kernel_size=pool_size)

        # level 1
        self.conv_1_down_2 = DoublConvBlock(num_kernels[0], num_kernels[1], kernel_size, strides, padding, activation, dropout, bn)
        self.pool_1_2 = nn.MaxPool2d(kernel_size=pool_size)                               
    
        # level 2
        self.conv_2_down_2 = DoublConvBlock(num_kernels[1], num_kernels[2], kernel_size, strides, padding, activation, dropout, bn)
        self.pool_2_2 = nn.MaxPool2d(kernel_size=pool_size)                               
    
        # level 3
        self.conv_3_2 = DoublConvBlock(num_kernels[2], num_kernels[3], kernel_size, strides, padding, activation, dropout, bn)
        
        # UP-SAMPLING PART (right side of the second U-net)
        
        # level 2
        self.concat_2_2 = UpConcatBlock2(pool_size, concat_axis)
        self.conv_2_up_2 = DoublConvBlock(num_kernels[3]+2*num_kernels[2], num_kernels[2], kernel_size, strides, padding, activation, dropout,bn)
        # level 1
        self.concat_1_2 = UpConcatBlock2(pool_size, concat_axis)
        self.conv_1_up_2 = DoublConvBlock(num_kernels[2]+2*num_kernels[1], num_kernels[1], kernel_size, strides, padding, activation, dropout,bn)

        # level 0
        self.concat_0_2 = UpConcatBlock2(pool_size, concat_axis)
        self.conv_0_up_2 = DoublConvBlock(num_kernels[1]+2*num_kernels[0], num_kernels[0], kernel_size, strides, padding, activation, dropout,bn)
        
        self.final_conv_2 = FinalConvBlock(num_kernels[0], n_classes, 1, strides, padding=final_padding, activation=final_activation)
    
    def forward(self, x):
        
        # The first U-net
        #print("\n\n --- First U-net ---")
        # DOWN-SAMPLING PART (left side of the first U-net)
        
        #save_img(x, 'input.png')            
        
        # level 0
        #print("\n\n --- Level 0\n")
        x1 = self.conv_0_down_1(x)
        x2 = self.pool_0_1(x1)
        
        # level 1
        #print("\n\n --- Level 1\n")
        x3 = self.conv_1_down_1(x2)
        x4 = self.pool_1_1(x3)
        
        # level 2
        #print("\n\n --- Level 2\n")
        x5 = self.conv_2_down_1(x4)
        x6 = self.pool_2_1(x5)
        
        # level 3
        #print("\n\n --- Level 3\n")
        x7 = self.conv_3_1(x6)
        
        # UP-SAMPLING PART (right side of the first U-net)
        
        # level 2
        #print("\n\n --- Level 2\n")
        x8 = self.concat_2_1(x7,x5)
        x9 = self.conv_2_up_1(x8)
        
        # level 1
        #print("\n\n --- Level 1\n")
        x10 = self.concat_1_1(x9,x3)
        x11 = self.conv_1_up_1(x10)
        
        # level 0
        #print("\n\n --- Level 0\n")
        x12 = self.concat_0_1(x11,x1)
        x13 = self.conv_0_up_1(x12)
        
        #print("\n\n --- Final Conv\n")
        output_1 = self.final_conv_1(x13)
        
        # The Second U-net
        #print("\n\n --- Second U-net ---")
        # DOWN-SAMPLING PART (left side of the second U-net)
        #save_img(output_1, 'output_1.png')
        
        # level 0
        #print("\n\n --- Level 0\n")
        y1 = self.conv_0_down_2(output_1)
        y2 = self.pool_0_2(y1)
        
        # level 1
        #print("\n\n --- Level 1\n")
        y3 = self.conv_1_down_2(y2)
        y4 = self.pool_1_2(y3)
        
        # level 2
        #print("\n\n --- Level 2\n")
        y5 = self.conv_2_down_2(y4)
        y6 = self.pool_2_2(y5)
        
        # level 3
        #print("\n\n --- Level 3\n")
        y7 = self.conv_3_2(y6)
        

        # UP-SAMPLING PART (right side of the second U-net)
        
        # level 2
        #print("\n\n --- Level 2\n")
        y8 = self.concat_2_2(y7,x5,y5)
        y9 = self.conv_2_up_2(y8)
        
        # level 1
        #print("\n\n --- Level 1\n")
        y10 = self.concat_1_2(y9,x3,y3)
        y11 = self.conv_1_up_2(y10)
        
        # level 0
        #print("\n\n --- Level 0\n")
        y12 = self.concat_0_2(y11,x1,y1)
        y13 = self.conv_0_up_2(y12)
        
        #print("\n\n --- Final Conv\n")
        output_2 = self.final_conv_2(y13)
        #save_img(output_2, 'output_2.png')
        
        return output_2
        
        
def save_img(img, name):
    fig = plt.figure()
    plt.imshow(img[0,0,:,:].cpu().detach().numpy())
    # if name already exists, change name
    name = get_new_name(name)
    plt.savefig(name)
    plt.close()
    
def get_new_name(name):
    if os.path.exists(name):
        if name[0].isdigit():
            id = f"{int(name[0]) + 1}"
            name = name[1:]
        else:
            id = 0
            id = f"{id + 1}_"
            
        name =  id + name
        name = get_new_name(name)
    return name