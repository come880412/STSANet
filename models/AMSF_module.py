'''
Modified Date: 2022/03/08
Author: Gi-Luen Huang
mail: come880412@gmail.com
'''

import torch.nn as nn
import torch

class Fusion_module(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Fusion_module, self).__init__()
        self.branch_1 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            nn.ReLU(),
            nn.Conv3d(in_channel, out_channel, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))
        )

        self.branch_2 = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=(3,1,1), stride=(1,1,1), padding=(1,0,0)),
            nn.BatchNorm3d(out_channel, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU(),
            nn.Conv3d(out_channel, out_channel, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),
        )

        self.branch_3 = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=(3,1,1), stride=(1,1,1), padding=(1,0,0)),
            nn.BatchNorm3d(out_channel, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU(),
            nn.Conv3d(out_channel, out_channel, kernel_size=(1,5,5), stride=(1,1,1), padding=(0,2,2)),
        )

        self.branch_4 = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0)),
        )
    
    def forward(self, x):
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)
        branch_3 = self.branch_3(x)
        branch_4 = self.branch_4(x)

        out = branch_1 + branch_2 + branch_3 + branch_4
        return out

class AMSF(nn.Module):
    def __init__(self, in_channel, out_channel, C, T, w, h):
        super(AMSF, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.conv1_1 = nn.Conv3d(2*in_channel, 1, (1,1,1), (1,1,1), (0,0,0))
        self.conv1_2 = nn.Conv3d(2*in_channel, in_channel//2, (1,1,1), (1,1,1), (0,0,0))
        self.conv1_3 = nn.Conv3d(in_channel//2, 2*in_channel, (1,1,1), (1,1,1), (0,0,0))
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)

        self.fusion = Fusion_module(in_channel, out_channel)
        self.upsample = nn.Upsample(size=(T, 2*w, 2*h), mode='trilinear')
        self.layernorm = nn.LayerNorm((C//2, T, 1, 1))
    
    def forward(self, x_low, x_high):
        batch, C, T, w, h = x_low.size()
        
        x_concat = torch.cat((x_low, x_high), dim=1)
        Fm = self.sigmoid(self.conv1_1(x_concat))
        x_concat = torch.mul(x_concat, Fm)
        x_concat = self.global_avg_pooling(x_concat)
        x_concat = self.sigmoid(self.conv1_3(self.relu(self.layernorm(self.conv1_2(x_concat)))))
        x_split_low, x_split_high = x_concat[:,:C,:,:,:], x_concat[:,C:,:,:,:]

        x_sum = x_low * x_split_low + x_high * x_split_high

        x_fusion = self.fusion(x_sum)
        x_fusion = self.upsample(x_fusion)
        return x_fusion
