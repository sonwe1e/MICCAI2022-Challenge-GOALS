#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""

import torch
import torch.nn as nn


def conv(in_channels, out_channels, kernel_size=5, stride=1, padding=2):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)


def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
    else:
        return nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                             conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1))


class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(UNetDownBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv1 = conv(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride,
                             padding=self.padding)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.relu1 = nn.PReLU()

        self.att = scam3d(self.out_channels, 2)

        self.conv2 = conv(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.relu2 = nn.PReLU()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.att(x)
        x = self.relu2(self.bn2(self.conv2(x))) + x
        return x


class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, merge_mode='concat', up_mode='transpose'):
        super(UNetUpBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels, mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv(2 * self.out_channels, self.out_channels)
        else:
            self.conv1 = conv(self.out_channels, self.out_channels)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.relu1 = nn.PReLU()

        self.att = scam3d(self.out_channels, 2)

        self.conv2 = conv(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.relu2 = nn.PReLU()

    def forward(self, from_up, from_down):
        from_up = self.upconv(from_up)

        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.att(x)
        x = self.relu2(self.bn2(self.conv2(x))) + x

        return x


class UNet(nn.Module):
    def __init__(self, n_channels=3, merge_mode='concat', up_mode='transpose'):
        super(UNet, self).__init__()
        self.n_chnnels = n_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.down1 = UNetDownBlock(self.n_chnnels, 32, 3, 1, 1)
        self.down2 = UNetDownBlock(32, 64, 4, 2, 1)
        self.down3 = UNetDownBlock(64, 128, 4, 2, 1)
        self.down4 = UNetDownBlock(128, 256, 4, 2, 1)
        self.down5 = UNetDownBlock(256, 512, 4, 2, 1)
        self.up1 = UNetUpBlock(512, 256, merge_mode=self.merge_mode, up_mode=self.up_mode)
        self.up2 = UNetUpBlock(256, 128, merge_mode=self.merge_mode, up_mode=self.up_mode)
        self.up3 = UNetUpBlock(128, 64, merge_mode=self.merge_mode, up_mode=self.up_mode)
        self.up4 = UNetUpBlock(64, 32, merge_mode=self.merge_mode, up_mode=self.up_mode)

        self.conv_final = nn.Sequential(conv(32, 64, 1, 1, 0), conv(64, 4, 5, 1, 2))

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

class scam3d(nn.Module):
    def __init__(self, c=128, r=4, pool_mode='avg'):
        super(scam3d, self).__init__()
        # 通过全局平均池化得到空间注意力
        self.pool_mode = pool_mode
        # pool_mode 是判断全局池化是最大池化还是平均池化
        if self.pool_mode == 'avg':
            self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        elif self.pool_mode == 'max':
            self.spatial_pool = nn.AdaptiveMaxPool2d(1)
        else:
            raise NotImplementedError
        # 通过求平均得到通道注意力机制
        self.conv1 = nn.Conv2d(c, c // r, kernel_size=1, stride=1)
        # 对应ConvBlock的第一个1x1 c和r与word文档对应 分别为通道数和降采样率
        self.conv1x1 = nn.Conv2d(c // r, c // r, kernel_size=1, stride=1)
        self.conv3x3 = nn.Conv2d(c // r, c // r, kernel_size=3, stride=1, padding=1)
        self.conv7x7 = nn.Conv2d(c // r, c // r, kernel_size=7, stride=1, padding=3)
        # 上述三个卷积分别为不同尺度的特征提取 分别为1x1,3x3,7x7 得到的特征将在通道维度拼接
        self.conv2 = nn.Conv2d(c * 3 // r, c, kernel_size=1, stride=1)
        # 对应ConvBlock的第二个1x1 再将通道复原
        self.sig = nn.Sigmoid()
        # 对特征进行sigmoid生成 attention

    def forward(self, x):
        spatial_pool = self.spatial_pool(x)
        if self.pool_mode == 'avg':
            channel_pool = torch.mean(x, dim=1, keepdim=True)
        elif self.pool_mode == 'max':
            channel_pool = torch.max(x, dim=1, keepdim=True)[0]
        # 上述两个操作是得到池化的特征
        s_c_3d = spatial_pool * channel_pool
        # 通过两个特征相乘得到 3D S&C Descriptor 即 s_c_3d
        s_c_3d = self.conv1(s_c_3d)
        # 通过第一个卷积对通道下采样
        s_c_3d1 = self.conv1x1(s_c_3d)
        s_c_3d2 = self.conv3x3(s_c_3d)
        s_c_3d3 = self.conv7x7(s_c_3d)
        s_c_3d = torch.cat([s_c_3d1, s_c_3d2, s_c_3d3], dim=1)
        # 将上述三个卷积的结果拼接
        s_c_3d = self.conv2(s_c_3d)
        # 通过第二个卷积对通道复原
        s_c_3d_attention = self.sig(s_c_3d)
        # 得到 3D S&C Attention
        x = x * s_c_3d_attention
        # 将其与原始特征相乘得到输出
        return x

class featrue(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(featrue, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class mynet(nn.Module):
    def __init__(self):
        super(mynet, self).__init__()
        self.conv1 = featrue(3, 64, 3, 1, 1)
        self.conv2 = featrue(64, 64, 3, 1, 1)
        self.down1 = featrue(64, 128, 4, 2, 1)
        self.conv3 = featrue(128, 128, 5, 1, 2)
        self.conv4 = featrue(128, 128, 5, 1, 2)
        self.down2 = featrue(128, 256, 4, 2, 1)
        self.conv5 = featrue(256, 256, 7, 1, 3)
        self.conv6 = featrue(256, 256, 7, 1, 3)
        self.down3 = featrue(256, 512, 4, 2, 1)
        self.conv7 = featrue(512, 512, 9, 1, 4)
        self.conv8 = featrue(512, 512, 9, 1, 4)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 5, 2, 1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
        )
        self.conv9 = featrue(512, 256, 7, 1, 3)
        self.conv10 = featrue(256, 256, 7, 1, 3)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
        )
        self.conv11 = featrue(256, 128, 5, 1, 2)
        self.conv12 = featrue(128, 128, 5, 1, 2)
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )
        self.conv13 = featrue(128, 128, 3, 1, 1)
        self.finalconv = featrue(128, 4, 1, 1, 0)

    def forward(self, x):
        x0 = self.conv1(x)
        x = self.conv2(x0) + x0
        x1 = self.down1(x)  # (128, 550, 400)
        x = self.conv3(x1)
        x = self.conv4(x) + x
        x2 = self.down2(x)  # (256, 275, 200)
        x = self.conv5(x2)
        x = self.conv6(x) + x
        x3 = self.down3(x)  # (512, 137, 100)
        x = self.conv7(x3)
        x = self.conv8(x) + x
        x = self.up1(x)[..., :-1]  # (256, 275, 200)
        x = self.conv9(torch.cat((x, x2), dim=1))
        x = self.conv10(x) + x
        x = self.up2(x)  # (128, 550, 400)
        x = self.conv11(torch.cat((x, x1), dim=1))
        x = self.conv12(x) + x
        x = self.up3(x)
        x = self.conv13(torch.cat((x, x0), dim=1))
        x = self.finalconv(x)
        return x


print('=========Testing Model========')
model = UNet()
a = torch.ones((2, 3, 1104, 800))
print(torch.argmax(model(a), dim=1).shape)
