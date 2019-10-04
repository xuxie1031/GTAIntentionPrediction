import torch
import torch.nn as nn
import torch.nn.functional as F

from st_gcn3d import *
from utils import *


class ObsEmbedLayer(nn.Module):
    def __init__(self, emb_dim, in_channels=4, nc=3, kernel_size=1, stride=1, batch_norm=False):
        super(ObsEmbedLayer, self).__init__()

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


class PredEmbedLayer(nn.Module):
    def __init__(self, emb_dim, nc=3):
        super(PredEmbedLayer, self).__init__()

        self.cell_in_emb = nn.Identity()

        self.onehots_emb = nn.Linear(nc, emb_dim)

    
    def forward(self, x, onehots):
        pass


class PredictionLayer(nn.Module):
	def __init__(self, h_dim, e_h_dim, out_dim=5, activation='relu', batch_norm=True, dropout=0.0):
		super(PredictionLayer, self).__init__()

		# self.enc_h = make_mlp([h_dim, e_h_dim], activation=activation, batch_norm=batch_norm, dropout=dropout)
		self.enc_h = nn.Linear(h_dim, e_h_dim)

		self.out = nn.Linear(e_h_dim, out_dim)

	
	def forward(self, x):
		assert len(x.size()) == 3

		x = self.enc_h(x)
		x = self.out(x)

		return output_activation(x)


