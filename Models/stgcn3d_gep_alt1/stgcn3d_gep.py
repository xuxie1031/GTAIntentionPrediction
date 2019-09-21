import torch
import torch.nn as nn
import torch.nn.functional as F

from st_gcn3d import *
from utils import *

class EmbedLayer(nn.Moudle):
    def __init__(self, emb_dim, in_channels=4, nc=3, kernel_size=1, stride=1, batch_norm=False):
        super(EmbedLayer, self).__init__()

        assert kernel_size % 2 == 1
        padding = ((kernel_size-1) // 2, 0, 0)

        self.dyn_emb = nn.Conv3d(
            in_channels,
            emb_dim,
            (kernel_size, 1, 1),
            (stride, 1, 1),
            padding
        )

        self.onehots_emb = nn.Conv3d(
            nc,
            emb_dim,
            (kernel_size, 1, 1),
            (stride, 1, 1),
            padding
        )

        self.bn = None
        if batch_norm: self.bn = nn.BatchNorm3d(emb_dim)

    
    def forward(self, x, onehots):
        x = self.dyn_emb(x)
        x = x if self.bn is None else self.bn(x)

        onehots = self.onehots_emb(onehots)
        onehots = onehots if self.bn is None else self.bn(onehots)

        x = torch.cat((x, onehots), dim=1)

        return x
    

class PredictionLayer(nn.Module):
    def __init__(self, h_dim, e_h_dim, out_dim=5, activation='relu', batch_norm=True, dropout=0.0):
        self.enc_h = make_mlp([h_dim, e_h_dim], activation=activation, batch_norm=batch_norm, dropout=dropout)

        self.out = nn.Linear(e_h_dim, out_dim)

    # x:(L, V, H)
    def forward(self, x):
        assert len(x.size()) == 3

        x = self.enc_h(x)
        x = self.out(x)

        return output_activation(x)


class STGCN3DGEPModel(nn.Module):
    def __init__(self, args, device=None, activation='relu', batch_norm=True):
        super(STGCN3DGEPModel, self).__init__()

        self.pred_len = args.pred_len
        self.emb_dim = args.emb_dim
        self.out_dim = args.out_dim
        self.nc = args.nc
        self.gru = args.gru
        self.device = device

        self.emb = EmbedLayer(args.emb_dim, args.in_channels, args.nc, args.emb_kernel_size)

        self.stgcn = STGCN3DModule(
            args.emb_dim*2,
            args.cell_input_dim,
            args.spatial_kernel_size,
            args.temporal_kernel_size,
            dropout=args.dropout,
            residual=args.residual
        )

        self.dec = nn.LSTM(args.cell_input_dim, args.cell_h_dim)
        if args.gru:
            self.dec = nn.GRU(args.cell_input_dim, args.cell_h_dim)
        
        self.predictor = PredictionLayer(
            h_dim=args.cell_h_dim,
            e_h_dim=args.e_h_dim,
            out_dim=args.out_dim,
            activation=activation,
            batch_norm=batch_norm,
            dropout=args.dropout
        )

        if args.use_cuda:
            self.to(device)
        

    def forward(self, x, A, one_hots_c_obs_seq):
        N, _, _, _, V = x.size()
        pred_outs = torch.zeros(N, self.pred_len, V, self.out_dim).to(self.device)

        x = self.emb(x, one_hots_c_obs_seq)

        x = self.stgcn(x, A)

        N, V, _ = x.size()
        x = x.unsqueeze(1)
        x = x.repeat(1, self.pred_len, 1, 1)

        for num in range(N):
            h, _ = self.dec(x)
            out = self.predictor(h)
            pred_outs[num] = out

        return pred_outs