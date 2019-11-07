# Network used for the discriminator shown in the SI.
# Code written by Jordan
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pickle
import numpy as np


class DNet(nn.Module):
    def __init__(self):
        super(DNet, self).__init__()
        self.Network = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=7, stride=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(16, 32, kernel_size=5, stride=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.Conv3d(32, 64, kernel_size=5, stride=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),
            nn.Conv3d(128, 256, kernel_size=3, stride=2),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(),
            nn.Conv3d(256, 512, kernel_size=3, stride=1),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(32768, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 2),
            nn.Softmax()
            )

    def forward(self, x):
        x = self.Network(x)
        return x