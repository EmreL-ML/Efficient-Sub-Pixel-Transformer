import torch
import torch.nn as nn
from utils import *


class ConvBlock(nn.Module):
    def __init__(self, num_input_channels, max_kernel_size, min_kernel_size, kernel_size_step, device='cuda'):
        super().__init__()
        self.in_channels = num_input_channels
        self.out_channels = self.calculate_num_of_output_channels()
        self.kernel_sizes = [ks for ks in reversed(range(min_kernel_size, max_kernel_size+1, kernel_size_step))]
        self.kernel_sizes_len = len(self.kernel_sizes)
        self.layers = []
        self.multipliers = []
        for i, ks in enumerate(self.kernel_sizes):
            if i == 0:
                same_padding = calculate_conv2d_same_padding(ks)
                conv_layer = nn.Conv2d(in_channels=self.in_channels, kernel_size=(ks, ks), stride=2,
                                       padding=same_padding, out_channels=self.out_channels, device=device)
                self.layers.append(conv_layer)
                multiplier = nn.Parameter(torch.tensor(1., device=device), True)
                self.multipliers.append(multiplier)
            else:
                if self.kernel_sizes_len > 2:
                    new_in_channels = self.layers[i-1].out_channels
                    new_out_channels = int(self.out_channels * (1. + .5*i))
                    conv_layer = nn.Conv2d(in_channels=new_in_channels, kernel_size=(ks, ks), stride=1,
                                           padding='same', out_channels=new_out_channels, device=device)
                    self.layers.append(conv_layer)
                    multiplier = nn.Parameter(torch.tensor(1., device=device), True)
                    self.multipliers.append(multiplier)
                else:
                    new_in_channels = self.out_channels
                    new_out_channels = self.out_channels * 2
                    conv_layer = nn.Conv2d(in_channels=new_in_channels, kernel_size=(ks, ks), stride=1,
                                           padding='same', out_channels=new_out_channels, device=device)
                    self.layers.append(conv_layer)
                    multiplier = nn.Parameter(torch.tensor(1., device=device), True)
                    self.multipliers.append(multiplier)
            if (i+1) == self.kernel_sizes_len:
                self.out_channels = conv_layer.out_channels

        #self.model = nn.Sequential(*self.layers)
        #self.out_channels = [module for module in self.model.modules()][-1].out_channels
        del self.kernel_sizes
        del self.kernel_sizes_len

    def calculate_num_of_output_channels(self, num_in_channels=None):
        if num_in_channels is None:
            num_channels = self.in_channels
        else:
            num_channels = num_in_channels
        counter = 0
        while True:
            counter += 1
            if 2**counter < num_channels*2:
                continue
            return 2**counter

    def forward(self, x):
        for layer, mul in zip(self.layers, self.multipliers):
            x = layer(x)
            x = nn.functional.tanh(x*mul)*mul
        return x