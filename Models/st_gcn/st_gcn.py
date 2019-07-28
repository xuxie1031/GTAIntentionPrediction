import torch
import torch.nn as nn

class GraphConvNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, s_kernel_size=1, t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1, bias=True):
        super(GraphConvNet, self).__init__()

        self.s_kernel_size = s_kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels*s_kernel_size, 
                              kernel_size=(t_kernel_size, 1),
                              padding=(t_padding, 1),
                              stride=(t_stride, 1),
                              dilation=(t_dilation, 1),
                              bias=bias)
        
    
    def forward(self, x, A):
        assert A.size(0) == self.s_kernel_size

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc // self.s_kernel_size, t, v)

        # only consider one s kernel:
        x = x.sum(dim=1)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A


class ST_GCN2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0, residual=True):
        super(ST_GCN2D, self).__init__()

        