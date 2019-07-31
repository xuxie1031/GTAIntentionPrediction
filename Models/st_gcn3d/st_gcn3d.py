import torch
import torch.nn as nn

from .graph import *

class GraphConvNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, s_kernel_size=1, t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1, bias=True):
        super(GraphConvNet3D, self).__init__()

        self.s_kernel_size = s_kernel_size
        self.conv = nn.Conv3d(in_channels, out_channels*s_kernel_size,
                                kernel_size=(t_kernel_size, 1, 1),
                                padding=padding(t_padding, 0),
                                stride=stride(t_stride, 1),
                                dilation=(t_dilation, 1),
                                bias=bias)

    
    def forward(self, x, A):
        assert A.size(0) == self.s_kernel_size

        x = self.conv(x)

        n, kc, t, v, v = x.size()
        x = x.view(n, self.s_kernel_size, kc // self.s_kernel_size, t, v, v)

        # only consider one s kernel:
        # x = x.sum(dim=1, keepdim=True)

        