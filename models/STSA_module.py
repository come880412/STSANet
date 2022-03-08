'''
Modified Date: 2022/03/08
Author: Gi-Luen Huang
mail: come880412@gmail.com
'''

import torch.nn as nn
import torch

class STSA_layer1(nn.Module):
    def __init__(self, in_channel, w, h):
        super(STSA_layer1, self).__init__()
        self.value = nn.Sequential(
            nn.Conv3d(in_channel, in_channel, kernel_size=(1,3,1), stride=(1,1,1), padding=(0,1,0)),
            nn.Conv3d(in_channel, in_channel, kernel_size=(1,1,3), stride=(1,1,1), padding=(0,0,1))
        )

        self.key = nn.Sequential(
            nn.Conv3d(in_channel, in_channel//2, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0)),
            nn.Conv3d(in_channel//2, in_channel//2, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1))
        )

        self.query = nn.Sequential(
            nn.Conv3d(in_channel, in_channel//2, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0)),
            nn.Conv3d(in_channel//2, in_channel//2, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1))
        )
        self.softmax = nn.Softmax(dim=-1)
        self.layer_norm = nn.LayerNorm((in_channel,4,w,h))

    def forward(self, x):
        x1, x2, x3, x4 = x[:,:,0,:,:].unsqueeze(2), x[:,:,1,:,:].unsqueeze(2), x[:,:,2,:,:].unsqueeze(2), x[:,:,3,:,:].unsqueeze(2)
        batch, C, T, w, h = x1.size()

        value_1, value_2 = self.value(x1).flatten(start_dim=2), self.value(x2).flatten(start_dim=2)
        key_1, key_2 = self.key(x1).flatten(start_dim=2), self.key(x2).flatten(start_dim=2)
        query_1, query_2 = self.query(x1).flatten(start_dim=2).permute(0,2,1), self.query(x2).flatten(start_dim=2).permute(0,2,1)

        value_3, value_4 = self.value(x3).flatten(start_dim=2), self.value(x4).flatten(start_dim=2)
        key_3, key_4 = self.key(x3).flatten(start_dim=2), self.key(x4).flatten(start_dim=2)
        query_3, query_4 = self.query(x3).flatten(start_dim=2).permute(0,2,1), self.query(x4).flatten(start_dim=2).permute(0,2,1)

        out_1 = self.Self_Att(query_2, key_1, value_1).view(batch, C, T, w, h)
        out_2 = self.Self_Att(query_1, key_2, value_2).view(batch, C, T, w, h)
        out_3 = self.Self_Att(query_4, key_3, value_3).view(batch, C, T, w, h)
        out_4 = self.Self_Att(query_3, key_4, value_4).view(batch, C, T, w, h)

        out = self.layer_norm(torch.cat((out_1, out_2, out_3, out_4), dim=2))

        return out + x
    
    def Self_Att(self, query, key, value):
        energy = torch.bmm(query, key)
        attention = self.softmax(energy)
        out = torch.bmm(value, attention.permute(0,2,1))
        return out

class STSA_layer2(nn.Module):
    def __init__(self, in_channel, w, h):
        super(STSA_layer2, self).__init__()
        self.value = nn.Sequential(
            nn.Conv3d(in_channel, in_channel, kernel_size=(1,3,1), stride=(1,1,1), padding=(0,1,0)),
            nn.Conv3d(in_channel, in_channel, kernel_size=(1,1,3), stride=(1,1,1), padding=(0,0,1))
        )

        self.key = nn.Sequential(
            nn.Conv3d(in_channel, in_channel//2, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0)),
            nn.Conv3d(in_channel//2, in_channel//2, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1))
        )

        self.query = nn.Sequential(
            nn.Conv3d(in_channel, in_channel//2, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0)),
            nn.Conv3d(in_channel//2, in_channel//2, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1))
        )
        self.softmax = nn.Softmax(dim=-1)
        self.layer_norm = nn.LayerNorm((in_channel,4,w,h))
    
    def forward(self, x):
        x1, x2 = x[:,:,0:2,:,:], x[:,:,2:4,:,:]
        batch, C, T, w, h = x1.size()

        value_1, value_2 = self.value(x1).flatten(start_dim=2), self.value(x2).flatten(start_dim=2)
        key_1, key_2 = self.key(x1).flatten(start_dim=2), self.key(x2).flatten(start_dim=2)
        query_1, query_2 = self.query(x1).flatten(start_dim=2).permute(0,2,1), self.query(x2).flatten(start_dim=2).permute(0,2,1)

        out_1 = self.Self_Att(query_2, key_1, value_1).view(batch, C, T, w, h)
        out_2 = self.Self_Att(query_1, key_2, value_2).view(batch, C, T, w, h)

        out = self.layer_norm(torch.cat((out_1, out_2), dim=2))
        return out + x

    
    def Self_Att(self, query, key, value):
        energy =  torch.bmm(query, key)
        attention = self.softmax(energy)
        out = torch.bmm(value, attention.permute(0,2,1) )
        return out
        
class STSA(nn.Module):
    def __init__(self, in_channel, w, h):
        super(STSA, self).__init__()
        self.STSA_layer1 = STSA_layer1(in_channel, w, h)
        self.down_conv = nn.Conv3d(in_channel, in_channel//2, (1,1,1), stride=(1,1,1), padding=(0,0,0))
        self.STSA_layer2 = STSA_layer2(in_channel//2, w, h)

    def forward(self, x):
        x = self.STSA_layer1(x)
        x = self.down_conv(x)
        x = self.STSA_layer2(x)

        return x