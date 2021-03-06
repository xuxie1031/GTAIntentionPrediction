import torch
import torch.nn as nn

from graph_full import Graph
from utils import *

class GraphConvNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, s_kernel_size=1, t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1, bias=True):
        super(GraphConvNet2D, self).__init__()

        self.s_kernel_size = s_kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels*s_kernel_size, 
                              kernel_size=(t_kernel_size, 1),
                              padding=(t_padding, 0),
                              stride=(t_stride, 1),
                              dilation=(t_dilation, 1),
                              bias=bias)
        
    
    def forward(self, x, A):
        assert A.size(1) == self.s_kernel_size

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc // self.s_kernel_size, t, v)

        # only consider one s kernel:
        # x = x.sum(dim=1, keepdim=True)

        x = torch.einsum('nkctv,nkvw->nctw', (x, A))

        return x.contiguous(), A


class GraphConvNetBrief2D(nn.Module):
    def __init__(self, in_channels, out_channels, t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1, bias=True):
        super(GraphConvNetBrief2D, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)


    def forward(self, x, A):
        x = self.conv(x)

        n, c, t, v = x.size()
        x = torch.einsum('nctv,nvw->nctw', (x, A))

        return x, A        


class NaiveGCN(nn.Module):
    def __init__(self, in_channels, s_kernel_size, bias=True):
        super(NaiveGCN, self).__init__()

        self.s_kernel_size = s_kernel_size
        # self.conv1 = nn.Conv1d(in_channels, out_channels*s_kernel_size, 
        #                       kernel_size=1)
        
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=1)

        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x, A):
        assert A.size(1) == self.s_kernel_size
        
        x = self.leaky_relu(self.conv1(x))

        return x, A


class ST_GCN2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0, residual=True, apply_gcn=False):
        super(ST_GCN2D, self).__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        # padding = ((kernel_size[0]-1) // 2, 0)

        self.apply_gcn = apply_gcn
        self.gcn = GraphConvNet2D(in_channels, out_channels, kernel_size[1])
        # self.gcn = GraphConvNetBrief2D(in_channels, out_channels)

        tcn_in = out_channels if apply_gcn else in_channels

        self.tcn = nn.Sequential(
            # nn.BatchNorm2d(tcn_in),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.ReLU(inplace=True),
            nn.Conv2d(
                tcn_in,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1)
                # padding
            ),
            # nn.BatchNorm2d(out_channels),
            # nn.Dropout(dropout, inplace=True)
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
        
        self.leaky_relu = nn.LeakyReLU(0.1)

    
    def forward(self, x, A):
        res = self.residual(x)
        if self.apply_gcn:
            x, A = self.gcn(x, A)
        x = self.leaky_relu(self.tcn(x)+res)
        # x = self.relu(x+res)

        return x, A


class STGCN2DModel(nn.Module):
    def __init__(self, pred_len, in_channels, spatial_kernel_size, temporal_kernel_size, dyn_hidden_size, self_hidden_size, enc_hidden_size, dec_hidden_size, out_dim, gru=False, use_cuda=True, device=None, **kwargs):
        super(STGCN2DModel, self).__init__()

        self.enc_dim = enc_hidden_size
        self.pred_len = pred_len
        self.out_dim = out_dim
        self.device = device

        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout' and k != 'residual'}

        self.st_gcn2d_modules = nn.ModuleList((
            ST_GCN2D(enc_hidden_size, 64, kernel_size, stride=1, residual=False, apply_gcn=True, **kwargs0),
            ST_GCN2D(64, 16, kernel_size, stride=1, **kwargs),
            ST_GCN2D(16, 8, kernel_size, stride=1, **kwargs)
            # ST_GCN2D(64, 64, kernel_size, stride=1, **kwargs),
            # ST_GCN2D(64, 64, kernel_size, stride=1, **kwargs),
            # ST_GCN2D(64, 64, kernel_size, stride=1, **kwargs),
            # ST_GCN2D(64, 128, kernel_size, stride=2, apply_gcn=True, **kwargs),
            # ST_GCN2D(128, 128, kernel_size, stride=1, **kwargs),
            # ST_GCN2D(128, 128, kernel_size, stride=1, **kwargs),
            # ST_GCN2D(128, 256, kernel_size, stride=2, **kwargs),
            # ST_GCN2D(256, 256, kernel_size, stride=1, **kwargs),
            # ST_GCN2D(256, 256, kernel_size, stride=1, **kwargs)
        ))

        self.naive_gcn = NaiveGCN(enc_hidden_size, kernel_size[1])

        self.dyn = nn.Linear(in_channels, dyn_hidden_size)

        self.enc = nn.LSTM(dyn_hidden_size, enc_hidden_size)
        if gru:
            self.enc = nn.GRU(dyn_didden_size, enc_hidden_size)

        self.hidden = nn.Linear(enc_hidden_size, self_hidden_size)

        self.dec = nn.LSTM(40+self_hidden_size, dec_hidden_size)
        if gru:
            self.dec = nn.GRU(40+self_hidden_size, dec_hidden_size)

        self.output = nn.Linear(dec_hidden_size, out_dim)
        
        self.leaky_relu = nn.LeakyReLU(0.1)

        if use_cuda:
            self.to(device)


    def forward(self, x, A):
        N, T, V, _ = x.size()
        o_enc = torch.zeros(N, T, V, self.enc_dim).to(self.device)
        o_enc_h = torch.zeros(N, V, self.enc_dim).to(self.device)
        o_pred = torch.zeros(N, self.pred_len, V, self.out_dim).to(self.device)

        x = self.leaky_relu(self.dyn(x))

        for i, data in enumerate(x):
            h_enc, tup_enc = self.enc(data)
            o_enc[i, :] = h_enc
            o_enc_h[i, :] = tup_enc[0].view(V, self.enc_dim)

        x = o_enc.permute(0, 3, 1, 2).contiguous()

        for gcn in self.st_gcn2d_modules:
            x, _ = gcn(x, A)
        
        _, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        data_pool = nn.MaxPool2d((1, 2), padding=(0, 1))
        x = data_pool(x)
        x = x.view(-1, V, C*5)

        # for i, data in enumerate(x):
            # _, tup_enc = self.enc(data)
            # o_enc_h[i, :] = tup_enc[0].view(V, self.enc_dim)

        # x = o_enc_h.clone()
        # x = x.permute(0, 2, 1)
        # x, _ = self.naive_gcn(x, A)
        # x = x.permute(0, 2, 1)

        o_enc_h = self.leaky_relu(self.hidden(o_enc_h))
        
        x = torch.cat((x, o_enc_h), 2)

        # prediction
        for i, data in enumerate(x):
            data = data.repeat(self.pred_len, 1, 1)
            h_dec, _ = self.dec(data)
            o = self.output(h_dec)
            o_pred[i, :] = output_activation(o)
        
        return o_pred
