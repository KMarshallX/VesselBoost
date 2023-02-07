import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding
# from shared.modules.vector_quantizer import VectorQuantizer1D

class CustomSegmentationNetwork(nn.Module):
    def __init__(self, dilation_rates=[1, 2, 3, 5, 7]):
        super().__init__()
        c = 16
        self.block1 = nn.Sequential(
            nn.Conv3d(1, c, 1),
            nn.LeakyReLU(),
            nn.Dropout3d(0.1)
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(c, c, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(c, c, 3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout3d(0.1)
        )

        self.block3 = nn.Sequential(
            nn.Conv3d(c, c, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(c, c, 3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout3d(0.1)
        )

        self.branches = nn.ModuleList(nn.Sequential(
            nn.Conv3d(c, c, 1),
            nn.LeakyReLU(),
            nn.Dropout3d(0.1)
        ))
        for r in dilation_rates:
            self.branches.append(nn.Sequential(
                nn.Conv3d(c, c, 3, padding=r, dilation=r), 
                nn.LeakyReLU(),
                nn.Dropout3d(0.1)
            ))
        self.merge = nn.Sequential(
            nn.Conv3d(c * (len(dilation_rates) + 3), c, 1), 
            nn.LeakyReLU(),
            nn.Dropout3d(0.1)
        )
        
        self.block4 = nn.Sequential(
            nn.Conv3d(c, c, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(c, c, 3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout3d(0.1)
        )

        self.out = nn.Sequential(
            nn.Conv3d(c, c, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(c, 1, 1)
        )

    def forward(self, x):
        x = self.block1(x)

        sc = x
        x = self.block2(x)
        x += sc

        sc = x
        x = self.block3(x)
        x += sc

        sc = x
        outs = []
        for l in self.branches:
            outs.append(l(x))
        x = torch.cat(outs, 1)
        x = self.merge(x)
        x += sc
        
        sc = x
        x = self.block4(x)
        x += sc
        x = self.out(x)
        return x