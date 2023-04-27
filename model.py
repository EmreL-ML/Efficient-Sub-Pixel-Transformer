from conv_block import ConvBlock
from torchinfo import summary
import torch.nn as nn
import torch


class Interpolation(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.downscale_first_img = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1,
                                   padding='same', device=self.device)
        self.multiplier_first_img = nn.Parameter(torch.tensor(1., device=self.device), True)
        self.downscale_second_img = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1,
                                   padding='same', device=self.device)
        self.multiplier_second_img = nn.Parameter(torch.tensor(1., device=self.device), True)
        self.conv_block0_first_img = ConvBlock(1, 7, 3, 2, self.device)
        self.conv_block0_second_img = ConvBlock(1, 7, 3, 2, self.device)
        self.conv_block1_first_img = ConvBlock(self.conv_block0_first_img.out_channels, 7, 3, 2, self.device)
        self.conv_block1_second_img = ConvBlock(self.conv_block0_second_img.out_channels, 7, 3, 2, self.device)
        self.conv_block2_first_img = ConvBlock(self.conv_block1_first_img.out_channels, 7, 3, 2, self.device)
        self.conv_block2_second_img = ConvBlock(self.conv_block1_second_img.out_channels, 7, 3, 2, self.device)

    def forward(self, x):
        x1 = x[:, :, 0, ...]
        x2 = x[:, :, 1, ...]
        x1 = torch.div(x1, 127.5)
        x2 = torch.div(x2, 127.5)
        x1 = torch.sub(x1, -1.)
        x2 = torch.sub(x2, -1.)
        x1 = self.downscale_first_img(x1)
        x1 = nn.functional.tanh(x1*self.multiplier_first_img)*self.multiplier_first_img
        x2 = self.downscale_second_img(x2)
        x2 = nn.functional.tanh(x2*self.multiplier_second_img)*self.multiplier_second_img
        x1 = self.conv_block0_first_img(x1)
        x2 = self.conv_block0_second_img(x2)
        x1 = self.conv_block1_first_img(x1)
        x2 = self.conv_block1_second_img(x2)
        x1 = self.conv_block2_first_img(x1)
        x2 = self.conv_block2_second_img(x2)
        x = torch.cat([x1, x2], dim=1)
        b, c, h, w = x.size()
        dif_h = 0
        dif_w = 0
        if h % 4 != 0:
            dif_h = 4-(h % 4)
        if w % 4 != 0:
            dif_w = 4 - (w % 4)
        if dif_w != 0 or dif_h != 0:
            x = nn.functional.pad(x, [dif_w, 0, dif_h, 0])
        x = nn.PixelUnshuffle(4)(x)
        b, c, h, w = x.size()
        x = torch.reshape(x, (b, c, h*w))
        x = torch.permute(x, (0, 2, 1))
        x, _ = nn.MultiheadAttention(2048, 16, batch_first=True, device=self.device)(x, x, x)
        x = torch.permute(x, (0, 2, 1))
        x = torch.reshape(x, (b, c, h, w))
        return x


#print(Interpolation()(torch.ones((1, 3, 2, 360, 640), device='cuda')))
summary(Interpolation(), (1, 3, 2, 360, 640), depth=16, device='cuda')