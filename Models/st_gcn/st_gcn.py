import torch
import torch.nn as nn

from .graph import *

class GraphConvNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, s_kernel_size=1, t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1, bias=True):
        super(GraphConvNet2D, self).__init__()

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
        # x = x.sum(dim=1, keepdim=True)

        x = torch.einsum('nkctv,nkvw->nctw', (x, A))

        return x.contiguous(), A


class ST_GCN2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0, residual=True):
        super(ST_GCN2D, self).__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0]-1) // 2, 0)

        self.gcn = GraphConvNet2D(in_channels, out_channels, kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True)
        )

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)
                ),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU()

    
    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.relu(self.tcn(x)+res)

        return x, A


class STGCN2DModel(nn.Module):
    def __init__(self, in_channels, spatial_kernel_size, temporal_kernel_size, dec_hidden_size, **kwargs):
        super(STGCN2DModel, self).__init__()

        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}

        self.st_gcn2d_modules = nn.ModuleList((
            ST_GCN2D(in_channels, 64, kernel_size, stride=1, residual=False, **kwargs0),
            ST_GCN2D(64, 64, kernel_size, stride=1, **kwargs),
            ST_GCN2D(64, 64, kernel_size, stride=1, **kwargs),
            ST_GCN2D(64, 64, kernel_size, stride=1, **kwargs),
            ST_GCN2D(64, 128, kernel_size, stride=2, **kwargs),
            ST_GCN2D(128, 128, kernel_size, stride=1, **kwargs),
            ST_GCN2D(128, 128, kernel_size, stride=1, **kwargs),
            ST_GCN2D(128, 256, kernel_size, stride=2, **kwargs),
            ST_GCN2D(256, 256, kernel_size, stride=1, **kwargs),
            ST_GCN2D(256, 256, kernel_size, stride=1, **kwargs)
        ))

        self.dec_lstm = nn.LSTM(256, dec_hidden_size)

    
    def forward(self, x):
        # init graphs on batch

        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(N, V*C, T)
        data_bn = nn.BatchNorm1d(V*C, affine=False)
        x = data_bn(x)
        x = x.view(N, V, C, T)
        x = x.permute(0, 2, 3, 1).contiguous()
