import torch.nn as nn

from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN

from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)

from compressai.models.google import JointAutoregressiveHierarchicalPriors
from adapters.adapter_block import ResidualAdapterBlock

class AdapterNet(JointAutoregressiveHierarchicalPriors):
    def __init__(self, N=128, v=3, R=2, **kwargs):
        super().__init__(N=N, M=N, **kwargs)

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualAdapterBlock(N, N, v, R),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualAdapterBlock(N, N, v, R),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualAdapterBlock(N, N, v, R),
            conv3x3(N, N, stride=2),
        )

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )

        self.g_s = nn.Sequential(
            ResidualAdapterBlock(N, N, v, R),
            ResidualBlockUpsample(N, N, 2),
            ResidualAdapterBlock(N, N, v, R),
            ResidualBlockUpsample(N, N, 2),
            ResidualAdapterBlock(N, N, v, R),
            ResidualBlockUpsample(N, N, 2),
            ResidualAdapterBlock(N, N, v, R),
            subpel_conv3x3(N, 3, 2),
        )

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.conv1.weight"].size(0)
        v = state_dict["g_a.1.conv4.weight"].size(2)
        R = state_dict["g_a.1.conv3.weight"].size(1) // state_dict["g_a.1.conv3.weight"].size(0)

        net = cls(N=N, v=v, R=R)
        
        net.load_state_dict(state_dict)
        return net