import torch
import torch.nn as nn
import torch.nn.functional as F

from st_gcn3d import *
from utils import *

class ClassifierLayer(nn.Module):
	def __init__(self, h_dim=256, n_c=5):
		self.fcn = nn.Conv2d(h_dim, n_c, kernel_size=1)


    def forward(self, x):
		assert len(x.size()) == 3
		N, V, H = x.size()

		x = x.permute(0, 2, 1).contiguous()
		x = x.view(-1, H, V, 1)

		x = F.avg_pool2d(x, x.size()[2:])

		x = self.fcn(x)
		x = x.view(-1, n_c)

		return x


class PredictionLayer(nn.Module):
	def __init__(self, h_dim=256, e_h_dim=256, e_c_dim=256, n_c=5, out_dim=5, activation='relu', batch_norm=True, dropout=0.0):
		self.enc_h = make_mlp(
			[h_dim, e_h_dim],
			activation=activation,
			batch_norm=batch_norm,
			dropout=dropout
		)

		self.enc_c = make_mlp(
			[n_c, e_c_dim],
			activation=activation,
			batch_norm=batch_norm,
			dropout=dropout
		)

		self.out = nn.Linear(e_h_dim+e_c_dim, out_dim)


	def forward(self, x, one_hots_c):
		assert len(x.size()) == 3
		assert len(one_hots_c.size()) == 2
		
		N, V, H = x.size()
		x = self.enc_h(x)

		one_hots_c = one_hots_c.unsqueeze(1).repeat(1, V, 1)
		one_hots_c = self.enc_c(one_hots_c)

		x = torch.cat((x, one_hots_c), 2)
		out = self.out(x)

		return out


class STGCN3DGEPModel(nn.Module):
	def __init__(self, args, activation='relu', batch_norm=True):
		self.pred_len = args.pred_len
		self.out_dim = args.out_dim
		self.n_c = args.n_c

		self.stgcn = STGCN3DModule(
			args.in_channels,
			args.spatial_kernel_size,
			args.temporal_kernel_size,
			dropout=args.dropout,
			residual=args.residual
		)

		self.cell = nn.LSTMCell(args.input_dim, args.cell_h_dim)
		if args.gru:
			self.cell = nn.GRUCell(args.input_size, args.cell_h_dim)
		
		self.classifier = ClassifierLayer(h_dim=args.cell_h_dim, n_c=args.n_c)
		
		self.predictor = PredictionLayer(
			h_dim=args.cell_h_dim, 
			e_h_dim=args.e_h_dim,
			e_c_dim=args.e_c_dim,
			n_c=args.n_c,
			out_dim=args.out_dim,
			activation=activation,
			batch_norm=batch_norm,
			dropout=dropout
		)

		if args.use_cuda:
			self.to(args.device)

	
	def forward(self, x, A):
		N, _, _, _, V = x.size()
		pred_outs = torch.zeros(self.pred_len, N, V, self.out_dim)
		c_outs = torch.zeros(self.pred_len, N, self.n_c)

		x = self.stgcn(x, A)

		N, V, C = x.size()
		x = x.view(-1, C)
		x = x.repeat(self.pred_len, 1, 1)

		for i in range(self.pred_len):
			h = self.cell(x[i])
			_, H = h.size()
			h = h.view(N, V, H)

			o_c = self.classifier(h)
			c_outs[i] = o_c

			o_p = self.predictor(h)
			pred_outs[i] = o_p
		
		return pred_outs, c_outs