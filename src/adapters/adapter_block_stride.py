import torch.nn as nn
from torch import Tensor
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)
from compressai.layers.gdn import GDN

class ResidualAdapterBlockStride(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int, out_ch: int, v: int, R: int, stride=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gdn = GDN(out_ch)

        self.skip = nn.Conv2d(in_ch, out_ch, 1, stride=stride)

        red_ch = out_ch // R #reduced channels number   #this was changed from red = in_ch // R to this !!!
        pad = v // 2 #padding
        self.conv3 = nn.Conv2d(out_ch, red_ch, 1)
        self.conv4 = nn.Conv2d(red_ch, red_ch, v, padding=pad)
        self.conv5 = nn.Conv2d(red_ch, out_ch, 1)
        

    def forward(self, x: Tensor) -> Tensor:

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.gdn(out)

        identity = self.skip(x)

        ad_out = self.conv3(identity)
        ad_out = self.conv4(ad_out)
        ad_out = self.conv5(ad_out)

        out = out + identity + ad_out
        return out