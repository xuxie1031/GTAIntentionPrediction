import torch
import torch.nn as nn
import torch.nn.functional as F

from st_gcn3d import *
from utils import *

class PredEmbedLayer(nn.Module):
	def __init__(self, emb_dim, nc=3, kernel_size=1, stride=1, batch_norm=False):
		super(PredEmbedLayer, self).__init__()

		assert kernel_size % 2 == 1
		padding = ((kernel_size-1) // 2, 0)

		self.cell_in_emb = nn.Identity()

		self.onehots_emb = nn.Conv2d(
			nc,
			emb_dim,
			(kernel_size, 1),
			(stride, 1),
			padding
		)

		self.bn = None
		if batch_norm: self.bn = nn.BatchNorm2d(emb_dim)

	
	def forward(self, x, onehots):
		x = self.cell_in_emb(x)

		onehots = self.onehots_emb(onehots)
		onehots = onehots if self.bn is None else self.bn(onehots)

		onehots = onehots.permute(0, 2, 3, 1).contiguous()
		x = torch.cat([x, onehots], dim=3)

		return x


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


class STGCN3DGEPModel(nn.Module):
	def __init__(self, args, device=None, activation='relu', batch_norm=True):
		super(STGCN3DGEPModel, self).__init__()
	
		self.pred_len = args.pred_len
		self.onehots_emb_dim = args.onehots_emb_dim
		self.out_dim = args.out_dim
		self.nc = args.nc
		self.gru = args.gru
		self.device = device

		self.stgcn = STGCN3DModule(
			args.in_channels,
			args.cell_input_dim,
			args.spatial_kernel_size,
			args.temporal_kernel_size,
			dropout=args.dropout,
			residual=args.residual
		)

		# self.onehots_emb = nn.Linear(args.nc, args.onehots_emb_dim)

		self.dec = nn.LSTM(args.cell_input_dim+args.onehots_emb_dim, args.cell_h_dim)
		if args.gru:
			self.dec = nn.GRU(args.cell_input_dim+args.onehots_emb_dim, args.cell_h_dim)
		
		self.pred_emb = PredEmbedLayer(
			emb_dim=args.onehots_emb_dim,
			nc=args.nc,
			kernel_size=1
		)

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

	
	def forward(self, x, A, one_hots_c_pred_seq):
		N, _, _, _, V = x.size()
		pred_outs = torch.zeros(N, self.pred_len, V, self.out_dim).to(self.device)

		x = self.stgcn(x, A)

		N, V, _ = x.size()
		x = x.unsqueeze(1)
		x = x.repeat(1, self.pred_len, 1, 1)

		# for num in range(N):
		# 	onehots_emb_seq = self.onehots_emb(one_hots_c_pred_seq[num])
		# 	onehots_emb_seq = onehots_emb_seq.unsqueeze(1)
		# 	onehots_emb_seq = onehots_emb_seq.repeat(1, V, 1)

		# 	cell_input = torch.cat((x[num], onehots_emb_seq), dim=2)
		# 	h, _ = self.dec(cell_input)
		# 	out = self.predictor(h)
		# 	pred_outs[num] = out

		x = self.pred_emb(x, one_hots_c_pred_seq)

		for num in range(N):
			cell_input = x[num]
			h, _ = self.dec(cell_input)
			out = self.predictor(h)
			pred_outs[num] = out

		return pred_outs