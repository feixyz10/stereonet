import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class StereoNet(nn.Module):
    def __init__(self, K, dmax, R):
        super().__init__()
        self.K = K
        self.dmax = dmax
        self.R = R

        self.feature_extractor = FeatureExtractor(K)
        self.cost_volume = CostVolume(K, dmax)
        self.cost_volume_filter = nn.Sequential(
            Conv3dBlock(32, 32), Conv3dBlock(32, 32),
            Conv3dBlock(32, 32), Conv3dBlock(32, 32),
            nn.Conv3d(32, 1, 3, 1, padding=1),
        )
        self.refiner = RefinementBlock()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, left_img, right_img):
        left_feature = self.feature_extractor(left_img)
        right_feature = self.feature_extractor(right_img)
        cost_vol = self.cost_volume(left_feature, right_feature)
        cost_vol_filtered = self.cost_volume_filter(cost_vol)
        disp_coarse = soft_argmin(cost_vol_filtered)

        disp_upsampled = F.interpolate(disp_coarse, size=left_img.shape[2:], mode='bilinear', align_corners=True) 
        disp_all = [F.relu(disp_upsampled).squeeze(1) * 2**self.K]
        for i in range(1, self.R+1):
            h, w = disp_coarse.size(2) * 2, disp_coarse.size(3) * 2
            disp_coarse = F.interpolate(disp_coarse, size=[h, w], mode='bilinear', align_corners=True) * 2
            img_downsampled = F.interpolate(left_img, size=[h, w], mode='bilinear', align_corners=True)
            disp_coarse = self.refiner(disp_coarse, img_downsampled)
            disp_upsampled = F.interpolate(disp_coarse, size=left_img.shape[2:], mode='bilinear', align_corners=True) 
            disp_all.append(F.relu(disp_upsampled).squeeze(1) * 2**(self.K - i))

        return disp_all


class FeatureExtractor(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.downsample = nn.Sequential()
        self.downsample.add_module('0', nn.Conv2d(3, 32, 5, 2, 2))  # no activation ???
        for i in range(1, K):
            self.downsample.add_module(str(i), nn.Conv2d(32, 32, 5, 2, 2))

        self.resblocks = nn.Sequential(
            ResBlock(32), ResBlock(32),
            ResBlock(32), ResBlock(32),
            ResBlock(32), ResBlock(32),
            nn.Conv2d(32, 32, 3, 1, padding=1)
        )

    def forward(self, x):
        out = self.downsample(x)
        return self.resblocks(out)


class CostVolume(nn.Module):
    def __init__(self, K, dmax):
        super().__init__()
        self.K = K
        self.dmax = dmax

    def forward(self, left_img, right_img):
        D = self.dmax // 2**self.K
        costvolume = [left_img - right_img]
        for d in range(1, D):
            diff = left_img[:,:,:,d:] - right_img[:,:,:,:-d]
            costvolume.append(F.pad(diff, [d, 0]))
        return torch.stack(costvolume, 4)  # N, C, H, W, D


def soft_argmin(inp):
    out = F.softmax(-inp, 4)  # N, C=1, H, W, D
    d = torch.arange(inp.size(4), dtype=inp.dtype, device=inp.device)
    d = d.reshape(1, 1, 1, 1, -1)
    out = d * out
    return torch.sum(out, dim=4)  # N, C=1, H, W


class RefinementBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.refineblock = nn.Sequential(
            nn.Conv2d(4, 32, 3, 1, padding=1),
            ResBlock(32, dilation=1), ResBlock(32, dilation=2),
            ResBlock(32, dilation=4), ResBlock(32, dilation=8),
            ResBlock(32, dilation=1), ResBlock(32, dilation=1),
            nn.Conv2d(32, 1, 3, 1, padding=1),
        )

    def forward(self, disp, img):
        concat = torch.cat([img, disp], 1)
        residual = self.refineblock(concat)

        return F.relu(residual + disp)


class ResBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.LeakyReLU(0.2)

    def forward(self, x):
        res = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + res
        return self.relu2(out)


class Conv3dBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.block(x)


if __name__ == '__main__':
    dev = torch.device("cuda")

    left_img = torch.randn(1, 3, 320, 320)
    right_img = torch.randn(1, 3, 320, 320)
    disparity = torch.randn(1, 1, 320, 320)

    model = StereoNet(3, 192, 3)
    model.to(dev)
    disp_all = model(left_img.to(dev), right_img.to(dev))
    print(disp_all.shape)
