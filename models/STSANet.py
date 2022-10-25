'''
Modified Date: 2022/03/08
Author: Gi-Luen Huang
mail: come880412@gmail.com
'''

import torch.nn as nn
import torch

import os

from models.S3D_backbone import BackBoneS3D
from models.STSA_module import STSA, STSA_layer1, STSA_layer2
from models.AMSF_module import AMSF

class Spatial_bottleneck(nn.Module):
    def __init__(self, in_channel, w, h):
        super(Spatial_bottleneck, self).__init__()
        self.pooling = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=(0,0,0), return_indices=True)
        self.STSA_layer1 = STSA_layer1(in_channel, w//2, h//2)
        self.down_conv = nn.Conv3d(in_channel, in_channel//2, (1,1,1), stride=(1,1,1), padding=(0,0,0))
        self.STSA_layer2 = STSA_layer2(in_channel//2, w//2, h//2)
        self.unpooling = nn.MaxUnpool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=(0,0,0))

    def forward(self, x):
        x, pooling_indices = self.pooling(x)
        x = self.STSA_layer1(x)
        x = self.unpooling(x, pooling_indices)
        x = self.down_conv(x)
        x, pooling_indices = self.pooling(x)
        x = self.STSA_layer2(x)
        x = self.unpooling(x, pooling_indices)

        return x

class STSANet(nn.Module):
    def __init__(self, T=32, w=384, h=224):
        super(STSANet, self).__init__()
        self.backbone = BackBoneS3D() # channels: 192 --> 480 --> 832 --> 1024
        
        self.spatial_bottleneck = Spatial_bottleneck(in_channel=192, w=w//4, h=h//4)
        self.STSA_0 = STSA(in_channel=480, w=w//8, h=h//8)
        self.STSA_1 = STSA(in_channel=832, w=w//16, h=h//16)
        self.STSA_2 = STSA(in_channel=1024, w=w//32, h=h//32)
        
        self.downsample_conv3d_0 = nn.Sequential(
            nn.Conv3d(192, 192, kernel_size=(4,1,1), stride=(4,1,1), padding=(0,0,0)),
            nn.BatchNorm3d(192, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU())
        self.downsample_conv3d_1 = nn.Sequential(
            nn.Conv3d(480, 480, kernel_size=(4,1,1), stride=(4,1,1), padding=(0,0,0)),
            nn.BatchNorm3d(480, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU())
        self.downsample_conv3d_2 = nn.Sequential(
            nn.Conv3d(832, 832, kernel_size=(2,1,1), stride=(2,1,1), padding=(0,0,0)),
            nn.BatchNorm3d(832, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU())
        
        self.deconv3d = nn.Sequential(
            nn.Conv3d(512, 416, kernel_size=(3,1,1), stride=(1,1,1), padding=(1,0,0)),
            nn.BatchNorm3d(416, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU(),
            nn.Upsample(size=(T//8, w//16, h//16), mode='trilinear'))
        
        self.AMSF_1 = AMSF(in_channel=416, out_channel=240, C=416, T=T//8, w=w//16, h=h//16)
        self.AMSF_2 = AMSF(in_channel=240, out_channel=96, C=240, T=T//8, w=w//8, h=h//8)
        self.AMSF_3 = AMSF(in_channel=96, out_channel=48, C=96, T=T//8, w=w//4, h=h//4)

        self.out_module = nn.Sequential(
            nn.Conv3d(48, 16, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(16, eps=1e-3, momentum=0.001, affine=True),
			nn.ReLU(),
			nn.Upsample(size=(4, w, h), mode='trilinear'),
            nn.Conv3d(16, 16, kernel_size=(3, 1, 1), stride=(3,1,1), padding=(0,0,0)),
            nn.BatchNorm3d(16, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU(),
            nn.Conv3d(16, 1, kernel_size=1, stride=1, bias=True),
			nn.Sigmoid()
        )
        
    def forward(self, x):
        [y0, y1, y2, y3] = self.backbone(x)

        y0 = self.downsample_conv3d_0(y0) # (batch, 192, 4, 56, 96)
        y1 = self.downsample_conv3d_1(y1) # (batch, 480, 4, 28, 48)
        y2 = self.downsample_conv3d_2(y2) # (batch, 832, 4, 14, 24)
        # y3 shape: (batch, 1024, 4, 7, 12)

        y0 = self.spatial_bottleneck(y0) # (batch, 96, 4, 56, 96)
        y1 = self.STSA_0(y1) # (batch, 240, 4, 28, 48)
        y2 = self.STSA_1(y2) # (batch, 416, 4, 14, 24)
        y3 = self.STSA_2(y3) # (batch, 512, 4, 7, 12)

        y3 = self.deconv3d(y3) # (batch, 416, 4, 14, 24)

        out = self.ASMF_1(y3, y2) # (batch, 240, 4, 28, 48)
        out = self.ASMF_2(out, y1) # (batch, 96, 4, 56, 96)
        out = self.ASMF_3(out, y0) # (batch, 16, 4, 112, 192)

        out = self.out_module(out) # (batch, 1, 1, 224, 384)
        out = out.view(out.size(0), out.size(3), out.size(4)) # (batch, 224, 384)
    
        return out


if __name__ == '__main__':
    image = torch.randn(1, 3, 32, 224, 384).cuda(1)
    model = STSANet().cuda(1)

    file_weight = './checkpoints/S3D_kinetics400.pt'
    if os.path.isfile(file_weight):
        print ('loading weight file')
        weight_dict = torch.load(file_weight)
        model_dict = model.backbone.state_dict()
        for name, param in weight_dict.items():
            if 'module' in name:
                name = '.'.join(name.split('.')[1:])
            if 'base.' in name:
                bn = int(name.split('.')[1])
                sn_list = [0, 5, 8, 14]
                sn = sn_list[0]
                if bn >= sn_list[1] and bn < sn_list[2]:
                    sn = sn_list[1]
                elif bn >= sn_list[2] and bn < sn_list[3]:
                    sn = sn_list[2]
                elif bn >= sn_list[3]:
                    sn = sn_list[3]
                name = '.'.join(name.split('.')[2:])
                name = 'base%d.%d.'%(sn_list.index(sn)+1, bn-sn)+name
            if name in model_dict:
                if param.size() == model_dict[name].size():
                    model_dict[name].copy_(param)
                else:
                    print (' size? ' + name, param.size(), model_dict[name].size())
            else:
                print (' name? ' + name)

        print (' loaded')
        model.backbone.load_state_dict(model_dict)
    else:
        print ('weight file?')

    pred = model(image)
    print(pred.shape)
