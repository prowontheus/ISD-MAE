import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel, shape='2D'):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    if shape == '2D':
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    elif shape == '3D':
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(-1)
        _3D_window = _2D_window.expand(channel, 1, window_size, window_size, window_size)
        window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    else:
        raise ValueError("Invalid shape. Shape should be '2D' or '3D'.")

    return window



def _ssim(img1, img2, window, window_size, channel, size_average = True, shape='2D'):
    if shape == '2D':
        mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
        mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)
    elif shape == '3D':
        mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel)
        mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    if shape == '2D':
        sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2
    elif shape == '3D':
        sigma1_sq = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
        sigma2_sq = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
        sigma12 = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        if shape == '2D':
            return ssim_map.mean(1).mean(1).mean(1)
        elif shape == '3D':
            return ssim_map.mean(1).mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True, shape='2D'):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.shape = shape
        self.window = create_window(window_size, self.channel, self.shape)

    def forward(self, img1, img2):
        channel = img1.shape[1]

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel, self.shape)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average, shape=self.shape)

def ssim(img1, img2, window_size = 11, size_average = True, shape='2D'):
    channel = img1.shape[1]
    window = create_window(window_size, channel, shape=shape)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average, shape=shape)
