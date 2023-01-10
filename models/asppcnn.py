"""
Atrous Spatial Pyramid Pooling 3D-CNN model

Editor: Marshall Xu
Last Edited: 01/10/2023 

Ref: 
https://github.com/SiyuLiu0329/vessel-app/blob/main/backend/implementations/dl_models/aspp_cnn.py
"""

import torch
from torch import nn

class InputBlock(nn.Module):
    """
    Input Block
    """
    def __init__(self, in_chan):
        super().__init__()

        self.inp = nn.Sequential(
            nn.Conv3d(in_chan, in_chan, kernel_size=1),
            nn.LeakyReLU(),
            nn.Dropout3d(0.1)
        )
    
    def forward(self, x):
        return self.inp(x)

class DenseBlock(nn.Module):
    """
    One encoder block
    """
    def __init__(self, in_chan):
        super().__init__()

        self.dense = nn.Sequential(
            nn.Conv3d(in_chan, in_chan, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(in_chan, in_chan, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout3d(0.1)
        )
    def forward(self, x):
        
        return self.dense(x)

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling
    """
    def __init__(self, dilation, in_chan):
        super().__init__()

        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv3d(in_chan, in_chan, kernel_size=1),
                nn.LeakyReLU(),
                nn.Dropout3d(0.1)
            )
        )
        for dil_num in dilation:
            self.branches.append(
                nn.Sequential(
                    nn.Conv3d(in_chan, in_chan, kernel_size=3, padding=dil_num, dilation=dil_num),
                    nn.LeakyReLU(),
                    nn.Dropout3d(0.1)
                )
            )
        
        self.merge = nn.Sequential(
            nn.Conv3d(in_chan * (len(dilation) + 3), in_chan, kernel_size=1),
            nn.LeakyReLU(),
            nn.Dropout3d(0.1)
        )

    def forward(self, x):
        out = []
        for layer in self.branches:
            out.append(layer(x))
        x = torch.cat(out, dim=1)
        x = self.merge(x)

        return x

class OutputBlock(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()

        self.out = nn.Sequential(
            nn.Conv3d(in_chan, in_chan, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(in_chan, out_chan, kernel_size=1)
        )

    def forward(self, x):

        return self.out(x)

class ASPPCNN(nn.Module):
    def __init__(self, in_chan, out_chan, dilation):
        super().__init__()

        self.inp = InputBlock(in_chan)
        self.block1 = DenseBlock(in_chan)
        self.block2 = DenseBlock(in_chan)

        self.asppblock = ASPP(dilation, in_chan)

        self.block3 = DenseBlock(in_chan)
        self.out = OutputBlock(in_chan, out_chan)

    def forward(self, x):
        x = self.inp(x)

        sc = x
        x = self.block1(x)
        x += sc

        sc = x
        x = self.block2(x)
        x += sc

        sc = x
        x = self.asppblock(x)
        x += sc

        sc = x
        x = self.block3(x)
        x += sc
        x = self.out(x)

        return x
