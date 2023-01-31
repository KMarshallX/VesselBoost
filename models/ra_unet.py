"""
Single level UNet3D with multipath residual attention block

Editor: Marshall Xu
Last Edited: 01/28/2023 

Ref: https://www.sciencedirect.com/science/article/pii/S1319157822001069
"""

import torch
from torch import nn
import numpy as np

class AtrousBlock(nn.Module):
    def __init__(self, in_chan, filter_num, dilation):
        super().__init__()

        self.block = nn.Sequential(
            nn.GroupNorm(in_chan,in_chan),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=in_chan, out_channels=filter_num, kernel_size=3, dilation=dilation, padding=dilation)
        )
    def forward(self, x):
        return self.block(x)


class AtrousSeries(nn.Module):
    """
    The sequence of atrous convolutions consists of three groups, Group Normalization, Relu, and Convolution, with different dilata-
    tions.
    """
    def __init__(self, in_chan, filter_num, dilation_series):
        super().__init__()

        self.atrous1 = AtrousBlock(in_chan, filter_num, dilation_series[0])
        self.atrous2 = AtrousBlock(filter_num, filter_num, dilation_series[1])
        self.atrous3 = AtrousBlock(filter_num, filter_num, dilation_series[2])

    def forward(self, x):
        x = self.atrous1(x)
        x = self.atrous2(x)
        x = self.atrous3(x)
        
        return x

class Attention(nn.Module):
    def __init__(self, in_chan, filter_num):
        super().__init__()

        self.atrousBlock = AtrousBlock(in_chan, filter_num, dilation=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, orginal_x, atrous_x):

        residual_connection = torch.add(orginal_x, atrous_x)
        xx = self.atrousBlock(residual_connection)
        xx = self.sigmoid(xx)
        
        return torch.mul(residual_connection, xx)

class OuterAttention(nn.Module):
    def __init__(self, in_chan) -> None:
        super().__init__()

        self.conv = nn.Conv3d(in_chan, 8, kernel_size=1)
        self.attention = Attention(8, 1)
    
    def forward(self, g, x):
        g = self.conv(g)
        x = self.conv(x)
        return self.attention(g,x)

class MRABlock(nn.Module):
    """
    Multipath Residual Attention Block

    """
    def __init__(self, in_chan, filter_num, upper_dilation_series, lower_dilation_series):
        super().__init__()
        
        # input: feature map from previous layer
        # filter: number of filter
        # dilation series: tuple, e.g. (1,2,4)
        
        self.upperAtrousConv = AtrousSeries(in_chan, filter_num, upper_dilation_series)
        self.lowerAtrousConv = AtrousSeries(in_chan, filter_num, lower_dilation_series)
        self.attention = Attention(filter_num, 1)
        self.out = AtrousBlock(3*filter_num, filter_num, dilation=1)

    def forward(self, x):
        #upper road
        upper = self.upperAtrousConv(x)
        upper = self.attention(x, upper)

        #lower road
        lower = self.lowerAtrousConv(x)
        lower = self.attention(x, lower)

        #concatnate
        cat = torch.cat([x, upper, lower], dim=1)

        return self.out(cat)

class MainArchitecture(nn.Module):
    def __init__(self, in_chan=1, filter_num=8, upper_dilations=(1,2,4), lower_dialtions=(2,4,1)) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Dropout3d(0.2),
            nn.Conv3d(in_chan, filter_num, kernel_size=1)
        )
        
        self.mrab_8 = MRABlock(filter_num, filter_num, upper_dilations, lower_dialtions)

        self.halfBridge = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(filter_num, filter_num*2, kernel_size=1)
        )

        self.mrab_16 = MRABlock(filter_num*2, filter_num*2, upper_dilations, lower_dialtions)

        self.transpConv = nn.ConvTranspose3d(filter_num*2, filter_num, kernel_size=2, stride=2)

        self.attention = OuterAttention(filter_num)

        self.out = AtrousBlock(filter_num*2, 1, 3)

    def forward(self, x):
        x = self.encoder(x)
        x = self.mrab_8(x)
        g = x

        #Bridge
        x = self.halfBridge(x)
        for i in range(6):
            x = self.mrab_16(x)
        x = self.transpConv(x)
        sc = x

        #attention
        x = self.attention(g, x)

        #concatnate
        x = torch.cat([x, sc], dim=1)

        return self.out(x)
        

if __name__ == "__main__":

    test_input = np.ones((1,64,64,64))
    test_input = torch.from_numpy(test_input).to(torch.float32).unsqueeze(0)

    test_input2 = np.ones((8,64,64,64))
    test_input2 = torch.from_numpy(test_input2).to(torch.float32).unsqueeze(0)

    test_conv = nn.Conv3d(8,8,1)
    test_tconv = nn.ConvTranspose3d(8,4,2,2)
    test_atrous = AtrousBlock(8, 8, 1)
    test_atrous_series = AtrousSeries(8, 8, (1,2,4))
    test_attention = Attention(8,1)
    test_mrab = MRABlock(8, 8, (1,2,4), (2,4,1))
    test_main = MainArchitecture()

    # test_output = test_attention(test_input, test_input2)
    test_x = test_main(test_input)
    print(test_x.shape)
