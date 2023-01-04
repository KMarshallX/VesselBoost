"""
3D-Unet model

Editor: Marshall Xu
Last Edited: 12/13/2022 optimized the structure

Ref: https://github.com/Thvnvtos/Lung_Segmentation
https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
"""

import torch
from torch import nn
import numpy as np

class ConvBlock(nn.Module):
    """
    3D conv block:

    Conv layer (3*3*3) + Batch Norm + ReLU +
    Conv layer (3*3*3) + Batch Norm + ReLU
    """
    def __init__(self, in_chan, out_chan):
        super().__init__()
        # nn sequential block
        self.convb = nn.Sequential(
            nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_chan),
            nn.ReLU(inplace=True), # changes input directly

            nn.Conv3d(out_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_chan),
            nn.ReLU(inplace=True) # changes input directly
        )
    
    def forward(self, x):
        return self.convb(x)

class EncBlock(nn.Module):
    """
    One encoder block:

    ConvBlock +
    MaxPooling (2*2*2, stride=2)
    
    """
    def __init__(self, in_chan, num_filter):
        super().__init__()

        self.convb = ConvBlock(in_chan, num_filter)
        self.maxp = nn.MaxPool3d(2) # default stride is equal to kernel_size

    
    def forward(self, x):
        xx = self.convb(x)
        p = self.maxp(xx)
        return xx, p
    
class DecBlock(nn.Module):
    """
    One decoder block:

    upconvolution (2*2*2, stride = 2) +
    ConvBlock (concatenated)

    """
    def __init__(self, in_chan, feat_chan, num_filter):
        super().__init__()
        # Mathematically, feat_chan(int) = 1.5*in_chan, 
        # num_filter(int) = 0.5*in_chan
        self.upsample = nn.ConvTranspose3d(in_chan, in_chan, 
        kernel_size=(2,2,2), stride=2)
        self.convb = ConvBlock(feat_chan, num_filter)
    
    def forward(self, x, cat_block):
        
        xx = self.upsample(x)
        cated_block = torch.cat([cat_block, xx], dim=1)
        out = self.convb(cated_block)

        return out


class Unet(nn.Module):
    """
    3D Unet model skeleton
    """
    def __init__(self, in_chan, out_chan, filter_num):
        super().__init__()
        # filter_num = 64 
        # Encoder units
        self.EncB1 = EncBlock(in_chan, filter_num)
        self.EncB2 = EncBlock(filter_num, filter_num*2)
        self.EncB3 = EncBlock(filter_num*2, filter_num*4)
        self.EncB4 = EncBlock(filter_num*4, filter_num*8)

        # bridge
        self.bridge = ConvBlock(filter_num*8, filter_num*16)

        # Decoder units
        
        self.DecB1 = DecBlock(filter_num*16, filter_num*24, filter_num*8)
        self.DecB2 = DecBlock(filter_num*8, filter_num*12, filter_num*4)
        self.DecB3 = DecBlock(filter_num*4, filter_num*6, filter_num*2)
        self.DecB4 = DecBlock(filter_num*2, filter_num*3, filter_num)


        self.out = nn.Conv3d(filter_num, out_chan, kernel_size=1) # out_chan = 1
    
    def forward(self, x):
        xx1, p1 = self.EncB1(x)
        xx2, p2 = self.EncB2(p1)
        xx3, p3 = self.EncB3(p2)
        xx4, p4 = self.EncB4(p3)

        p5 = self.bridge(p4)

        p6 = self.DecB1(p5, xx4)
        p7 = self.DecB2(p6, xx3)
        p8 = self.DecB3(p7, xx2)
        p9 = self.DecB4(p8, xx1)

        return self.out(p9)


if __name__ == "__main__":


    test_input = np.ones([64,64,64])
    test_input = test_input[None, :]

    test_input = torch.from_numpy(test_input).to(torch.float32)
    test_input = test_input.unsqueeze(0)
    print(test_input.shape)

    Model = Unet(1,1,64)

    test_output = Model(test_input)

