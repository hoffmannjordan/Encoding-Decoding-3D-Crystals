from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pickle
import numpy as np
from torch.utils.data.dataset import Dataset
from pymatgen.core.structure import Structure

# Code adpated by Jordan from: https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py
# Changed from 2-D  to 3-D
 
class Attention3D(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention3D,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        return x*psi

class conv_block3D(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block3D,self).__init__()
        self.Conv3D_ = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.Conv3D_(x)
        return x

class conv_block3D2(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block3D2,self).__init__()
        self.Conv3D_ = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=4,stride=1,padding=1,bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=4,stride=1,padding=1,bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.Conv3D_(x)
        return x


class up_conv3D(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv3D,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
            super(Interpolate, self).__init__()
            self.interp = nn.functional.interpolate
            self.scale_factor = scale_factor
            self.mode = mode

    def forward(self, x):
            x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
            return x

class AttU_Net3D(nn.Module):
    def __init__(self,input_ch=1,output_ch=95): #Number of classes
        super(AttU_Net3D,self).__init__()
        
        self.Maxpool3D = nn.MaxPool3d(kernel_size=2,stride=2)

        self.pad = nn.ReplicationPad3d(1)
        self.sig = nn.Sigmoid()
        self.Conv3D_1 = conv_block3D(ch_in=input_ch,ch_out=64)
        self.Conv3D_2 = conv_block3D(ch_in=64,ch_out=128)
        self.Conv3D_3 = conv_block3D(ch_in=128,ch_out=256)
        self.Conv3D_4 = conv_block3D(ch_in=256,ch_out=512)
        self.Conv3D_5 = conv_block3D(ch_in=512,ch_out=1024)

        self.Up5 = up_conv3D(ch_in=1024,ch_out=512)
        self.Att3D_5 = Attention3D(F_g=512,F_l=512,F_int=256)
        self.Up3D_conv5 = conv_block3D(ch_in=1024, ch_out=512)

        self.Up4 = up_conv3D(ch_in=512,ch_out=256)
        self.Att3D_4 = Attention3D(F_g=256,F_l=256,F_int=128)
        self.Up3D_conv4 = conv_block3D(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv3D(ch_in=256,ch_out=128)
        self.Att3D_3 = Attention3D(F_g=128,F_l=128,F_int=64)
        self.Up3D_conv3 = conv_block3D(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv3D(ch_in=128,ch_out=64)
        self.Att3D_2 = Attention3D(F_g=64,F_l=64,F_int=32)
        self.Up3D_conv2 = conv_block3D2(ch_in=128, ch_out=64)

        self.Conv3D_1x1 = nn.Conv3d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        x  = self.pad(x) #Needed for 2^N input
        x1 = self.Conv3D_1(x)
        x2 = self.Maxpool3D(x1)
        x2 = self.Conv3D_2(x2)
        x3 = self.Maxpool3D(x2)
        x3 = self.Conv3D_3(x3)
        x4 = self.Maxpool3D(x3)
        x4 = self.Conv3D_4(x4)
        x5 = self.Maxpool3D(x4)
        x5 = self.Conv3D_5(x5)
        d5 = self.Up5(x5)        
        x4 = self.Att3D_5(g=d5,x=x4)        
        d5 = torch.cat((x4,d5),dim=1)            
        d5 = self.Up3D_conv5(d5)  
        d4 = self.Up4(d5)
        x3 = self.Att3D_4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up3D_conv4(d4)
        d3 = self.Up3(d4)
        x2 = self.Att3D_3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up3D_conv3(d3)
        d2 = self.Up2(d3)
        x1 = self.Att3D_2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up3D_conv2(d2)
        d1 = self.Conv3D_1x1(d2)
        d1 = self.sig(d1)
        return d1