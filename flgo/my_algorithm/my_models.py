import torch,copy
import torch.nn as nn
import torch.nn.functional as F

#https://github.com/hermittt/VQGAN-pytorch/blob/main/helper.py
class UpSampleBlock(nn.Module):
    def __init__(self, channels):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0)
        return self.conv(x)
class DownSampleBlock(nn.Module):
    def __init__(self, channels):
        super(DownSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 2, 0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        return self.conv(x)

class ResBlock(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.block = nn.Sequential(
        nn.Conv2d(dim, dim, 3, 1, 1),
        nn.BatchNorm2d(dim),
        nn.ReLU(True),
        nn.Conv2d(dim, dim, 1),
        nn.BatchNorm2d(dim),
        nn.ReLU(True),
    )
  def forward(self, x):
    return x + self.block(x)
    
class Encoder(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.input = nn.Conv2d(args.image_channels, args.latent_dim, args.in_kernel[0], args.in_kernel[1], args.in_kernel[2])
    self.model = nn.Sequential(
        ResBlock(args.latent_dim),
        DownSampleBlock(args.latent_dim),
        ResBlock(args.latent_dim),
        DownSampleBlock(args.latent_dim),
        ResBlock(args.latent_dim),
        nn.Conv2d(args.latent_dim, args.latent_dim, 1, 1, 0),
        nn.Tanh(),
        ).to(device=args.device)
  def forward(self, x):
    x = self.input(x)
    return self.model(x)

class Decoder(nn.Module):
  def __init__(self, args): #args.latent_dim
    super().__init__()
    self.model = nn.Sequential(
        ResBlock(args.latent_dim),
        UpSampleBlock(args.latent_dim),
        ResBlock(args.latent_dim),
        UpSampleBlock(args.latent_dim),
        ResBlock(args.latent_dim),
        nn.Conv2d(args.latent_dim, args.image_channels, args.out_kernel[0], args.out_kernel[1], args.out_kernel[2]),
        ).to(device=args.device)
  def forward(self, x):
    return F.tanh(self.model(x))
