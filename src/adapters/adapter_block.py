import torch.nn as nn
from torch import Tensor

class ResidualAdapterBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int, out_ch: int, v: int, R: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)#conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        red_ch = out_ch // R #reduced channels number   #this was changed from red = in_ch // R to this !!!
        pad = v // 2 #padding
        self.conv3 = nn.Conv2d(in_ch, red_ch, 1)
        self.conv4 = nn.Conv2d(red_ch, red_ch, v, padding=pad)
        self.conv5 = nn.Conv2d(red_ch, out_ch, 1)
        

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        ad_out = self.conv3(x)
        ad_out = self.conv4(ad_out)
        ad_out = self.conv5(ad_out)

        out = out + identity + ad_out
        return out