""" Network architectures.
"""
# pylint: disable=W0221,W0622,C0103,R0913

##
import math
from typing import Tuple
import torchvision.utils as vutils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import cv2 as cv
import torch.nn.functional as F


def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)

###

class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, act, se):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                               groups=in_size, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = act(inplace=True)

        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                               groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.act2 = act(inplace=True)
        self.se = eca_layer(expand_size) if se else nn.Identity()

        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.act3 = act(inplace=True)

        self.skip = None
        if in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

    def forward(self, x):
        skip = x

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))

        if self.skip is not None:
            skip = self.skip(skip)

        out = self.act3(out + skip)
        return out

##

#U-net网络
def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=False
    )

def upconv2x2(in_channels, out_channels, mode="transpose"):
    if mode == "transpose":
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    else:
        return nn.Sequential(
            nn.Upsample(mode="bilinear", scale_factor=2),
            conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
        )

class UNetDownBlockstart(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(UNetDownBlockstart, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv1 = conv(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )
        self.bn1 = nn.BatchNorm2d(self.out_channels, eps=1e-05)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        return x

class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(UNetDownBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.block1 = Block(self.kernel_size, self.in_channels, 2 * self.in_channels, self.out_channels, nn.ReLU, False)

        self.maxpool1 = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.maxpool1(x)
        x = self.block1(x)

        return x


class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, merge_mode="concat", up_mode="transpose"):
        super(UNetUpBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels, mode=self.up_mode)

        self.block1 = Block(3, 2*self.out_channels, 2 * self.out_channels, self.out_channels, nn.ReLU, False)

    def forward(self, from_up, from_down):
        from_up = self.upconv(from_up)

        if self.merge_mode == "concat":
            x = torch.cat((from_up, from_down), 1)
        x = self.block1(x)
        return x

class UNet(nn.Module):
    def __init__(self, n_channels=3, merge_mode="concat", up_mode="transpose"):
        super(UNet, self).__init__()
        self.n_chnnels = n_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.down1 = UNetDownBlockstart(self.n_chnnels, 64, 3, 1, 1)
        self.down2 = UNetDownBlock(64, 128, 3, 1, 1)
        self.down3 = UNetDownBlock(128, 256, 3, 1, 1)
        self.down4 = UNetDownBlock(256, 512, 3, 1, 1)
        self.down5 = UNetDownBlock(512, 512, 3, 1, 1)

        self.up1 = UNetUpBlock(512, 512, merge_mode=self.merge_mode, up_mode=self.up_mode)
        self.up2 = UNetUpBlock(512, 256, merge_mode=self.merge_mode, up_mode=self.up_mode)
        self.up3 = UNetUpBlock(256, 128, merge_mode=self.merge_mode, up_mode=self.up_mode)
        self.up4 = UNetUpBlock(128, 64, merge_mode=self.merge_mode, up_mode=self.up_mode)
        self.conv_final = nn.Sequential(conv(64, 3, 3, 1, 1), nn.Tanh())

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.conv_final(x)

        return x

class U_featureNet(nn.Module):
    def __init__(self, n_channels=3, merge_mode="concat", up_mode="transpose"):
        super(U_featureNet, self).__init__()
        self.n_chnnels = n_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.down1 = UNetDownBlockstart(self.n_chnnels, 64, 3, 1, 1)
        self.down2 = UNetDownBlock(64, 128, 3, 1, 1)
        self.down3 = UNetDownBlock(128, 256, 3, 1, 1)
        self.down4 = UNetDownBlock(256, 512, 3, 1, 1)
        self.down5 = UNetDownBlock(512, 512, 3, 1, 1)

        self.up1 = UNetUpBlock(512, 512, merge_mode=self.merge_mode, up_mode=self.up_mode)
        self.up2 = UNetUpBlock(512, 256, merge_mode=self.merge_mode, up_mode=self.up_mode)
        self.up3 = UNetUpBlock(256, 128, merge_mode=self.merge_mode, up_mode=self.up_mode)
        self.up4 = UNetUpBlock(128, 64, merge_mode=self.merge_mode, up_mode=self.up_mode)
        self.conv_final = nn.Sequential(conv(64, 3, 3, 1, 1), nn.Tanh())

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.conv_final(x)

        return x

##
class NetG(nn.Module):
    """
    GENERATOR NETWORK
    """
    def __init__(self, opt):
        super(NetG, self).__init__()
        self.opt = opt
        self.generation = UNet()
        self.pregeneration = U_featureNet()

    def _reconstruct(self, mb_img, cutout_size, selecttest):

        _, _, h, w = mb_img.shape
        num_disjoint_masks = 3
        disjoint_masks = self._create_disjoint_masks((h, w), cutout_size, num_disjoint_masks)

        mb_reconst = 0
        pre_reconst = 0
        for mask in disjoint_masks:
            mb_cutout = mb_img * mask

            pre = self.pregeneration(mb_cutout)
            if selecttest == 0:
                mb_cutout = mb_cutout + pre * (1 - mask)
            else:
                mb_cutout = mb_cutout + 0.5 * pre * (1 - mask) + 0.5 * mb_img * (1 - mask)

            mb_inpaint = self.generation(mb_cutout)
            mb_reconst += mb_inpaint * (1 - mask)

            pre_reconst += pre * (1 - mask)
        return mb_reconst, pre_reconst

    def _create_disjoint_masks(
        self,
        img_size: Tuple[int, int],
        cutout_size: int = 8,
        num_disjoint_masks: int = 3,
    ):

        img_h, img_w = img_size
        grid_h = math.ceil(img_h / cutout_size)
        grid_w = math.ceil(img_w / cutout_size)
        num_grids = grid_h * grid_w
        disjoint_masks = []
        for grid_ids in np.array_split(np.random.permutation(num_grids), num_disjoint_masks):
            flatten_mask = np.ones(num_grids)
            flatten_mask[grid_ids] = 0
            mask = flatten_mask.reshape((grid_h, grid_w))
            mask = mask.repeat(cutout_size, axis=0).repeat(cutout_size, axis=1)
            mask = torch.tensor(mask, requires_grad=False, dtype=torch.float)
            mask = mask.to(torch.device("cuda:0" if self.opt.device != 'cpu' else "cpu"))
            disjoint_masks.append(mask)

        return disjoint_masks

    def forward(self, x, cutout_size, selecttest = 0):
        return self._reconstruct(x, cutout_size, selecttest)