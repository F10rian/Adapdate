import torch.nn as nn

class Adapter_for_adapter_net(nn.Module):
    def __init__(self, N=128, v=3, R=2, **kwargs):
        super(Adapter_for_adapter_net, self).__init__()

        red_ch = N // R #reduced channels number
        pad = v // 2 #padding
        
        self.g_a1conv3 = nn.Conv2d(N, red_ch, 1)
        self.g_a1conv4 = nn.Conv2d(red_ch, red_ch, v, padding=pad)
        self.g_a1conv5 = nn.Conv2d(red_ch, N, 1)

        self.g_a3conv3 = nn.Conv2d(N, red_ch, 1)
        self.g_a3conv4 = nn.Conv2d(red_ch, red_ch, v, padding=pad)
        self.g_a3conv5 = nn.Conv2d(red_ch, N, 1)

        self.g_a5conv3 = nn.Conv2d(N, red_ch, 1)
        self.g_a5conv4 = nn.Conv2d(red_ch, red_ch, v, padding=pad)
        self.g_a5conv5 = nn.Conv2d(red_ch, N, 1)

        self.g_s0conv3 = nn.Conv2d(N, red_ch, 1)
        self.g_s0conv4 = nn.Conv2d(red_ch, red_ch, v, padding=pad)
        self.g_s0conv5 = nn.Conv2d(red_ch, N, 1)

        self.g_s2conv3 = nn.Conv2d(N, red_ch, 1)
        self.g_s2conv4 = nn.Conv2d(red_ch, red_ch, v, padding=pad)
        self.g_s2conv5 = nn.Conv2d(red_ch, N, 1)

        self.g_s4conv3 = nn.Conv2d(N, red_ch, 1)
        self.g_s4conv4 = nn.Conv2d(red_ch, red_ch, v, padding=pad)
        self.g_s4conv5 = nn.Conv2d(red_ch, N, 1)

        self.g_s6conv3 = nn.Conv2d(N, red_ch, 1)
        self.g_s6conv4 = nn.Conv2d(red_ch, red_ch, v, padding=pad)
        self.g_s6conv5 = nn.Conv2d(red_ch, N, 1)