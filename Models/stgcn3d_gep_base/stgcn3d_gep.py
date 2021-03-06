import torch
import torch.nn as nn
import torch.nn.functional as F

from st_gcn3d import *
from utils import *

class ClassifierLayer(nn.Module):
	def __init__(self, h_dim=256, nc=5):
		super(ClassifierLayer, self).__init__()

		self.nc = nc
		self.fcn = nn.Conv2d(h_dim, nc, kernel_size=1)


	def forward(self, x):
		assert len(x.size()) == 3
		N, V, H = x.size()

		x = x.permute(0, 2, 1).contiguous()
		x = x.view(-1, H, V, 1)

		x = F.avg_pool2d(x, x.size()[2:])

		x = self.fcn(x)
		x = x.view(-1, self.nc)

		return x


class PredictionLayer(nn.Module):
	def __init__(self, h_dim=256, e_h_dim=256, e_c_dim=256, nc=5, out_dim=5, activation='relu', batch_norm=True, dropout=0.0):
		super(PredictionLayer, self).__init__()

		self.enc_h = make_mlp([h_dim, e_h_dim], activation=activation, batch_norm=batch_norm, dropout=dropout)
		self.enc_c = make_mlp([nc, e_c_dim], activation=activation, batch_norm=batch_norm, dropout=dropout)
		#self.enc_h = nn.Sequential(
		#	nn.Conv2d(h_dim, e_h_dim, kernel_size=1),
		#	nn.BatchNorm2d(e_h_dim),
		#	nn.ReLU(inplace=True)
		#)

		#self.enc_c = nn.Sequential(
		#	nn.Conv2d(nc, e_c_dim, kernel_size=1),
		#	nn.BatchNorm2d(e_c_dim),
		#	nn.ReLU(inplace=True)
		#)

		#self.hidden = make_mlp([e_h_dim+e_c_dim, e_h_dim+e_c_dim], activation=activation, batch_norm=batch_norm, dropout=dropout)
		self.out = nn.Linear(e_h_dim+e_c_dim, out_dim)

		if activation == 'relu':
			self.relu = nn.ReLU()
		elif activation == 'leakyrelu':
			self.relu = nn.LeakyReLU()

		self.dropout = nn.Dropout(p=dropout)


	def forward(self, x, one_hots_c):
		assert len(x.size()) == 3
		assert len(one_hots_c.size()) == 2

		N, V, H = x.size()
		x = x.view(N*V, -1)
		x = self.enc_h(x)

		one_hots_c = one_hots_c.unsqueeze(1).repeat(1, V, 1)
		one_hots_c = one_hots_c.view(N*V, -1)
		one_hots_c = self.enc_c(one_hots_c)

		x = torch.cat((x, one_hots_c), 1)

		#N, V, H = x.size()

		#x = x.permute(0, 2, 1).contiguous()
		#x = x.view(N, H, V, 1)
		#x = self.enc_h(x)
		#x = x.view(N, V, -1)

		#one_hots_c = one_hots_c.unsqueeze(1).repeat(1, V, 1)
		#one_hots_c = one_hots_c.permute(0, 2, 1).contiguous()
		#one_hots_c = one_hots_c.view(N, -1, V, 1)
		#one_hots_c = self.enc_c(one_hots_c)
		#one_hots_c = one_hots_c.view(N, V, -1)

		#x = torch.cat((x, one_hots_c), 2)

		#x = self.hidden(x)
		out = self.out(x)
		out = out.view(N, V, -1)

		return out


class STGCN3DGEPModel(nn.Module):
	def __init__(self, args, device=None, activation='relu', batch_norm=True):
		super(STGCN3DGEPModel, self).__init__()

		self.pred_len = args.pred_len
		self.out_dim = args.out_dim
		self.nc = args.nc
		self.gru = args.gru
		self.use_grammar = args.use_grammar
		self.device = device

		self.stgcn = STGCN3DModule(
			args.in_channels,
			args.cell_input_dim,
			args.spatial_kernel_size,
			args.temporal_kernel_size,
			dropout=args.dropout,
			residual=args.residual
		)

		self.cell = nn.LSTMCell(args.cell_input_dim, args.cell_h_dim)
		if args.gru:
			self.cell = nn.GRUCell(args.cell_input_dim, args.cell_h_dim)
		
		self.classifier = ClassifierLayer(h_dim=args.cell_h_dim, nc=args.nc)
		
		self.predictor = PredictionLayer(
			h_dim=args.cell_h_dim, 
			e_h_dim=args.e_h_dim,
			e_c_dim=args.e_c_dim,
			nc=args.nc,
			out_dim=args.out_dim,
			activation=activation,
			batch_norm=batch_norm,
			dropout=args.dropout
		)

		if args.use_cuda:
			self.to(device)

	# one_hots_c_seq forms differently in train / test
	# x: (N, C, T, V, V); A: (N, V, V); hidden_states: (N, V, H); cell_states: (N, V, C)
	# one_hots_c_pred_seq: (L, N, NC)
	# grammar_gep: _; gep_parsed_sentence: (N, _)
	# will add gae only part
	def forward(self, x, A, hidden_states, cell_states, one_hots_c_pred_seq, grammar_gep, history, curr_l):
		N, _, _, _, V = x.size()
		pred_outs = torch.zeros(self.pred_len, N, V, self.out_dim).to(self.device)
		c_outs = torch.zeros(self.pred_len, N, self.nc).to(self.device)

		x = self.stgcn(x, A)

		N, V, C = x.size()
		x = x.repeat(self.pred_len, 1, 1, 1)

		for i in range(self.pred_len):
			for num in range(N):
				if self.gru:
					hidden_states[num] = self.cell(x[i, num], hidden_states[num].clone())
				else:
					hidden_states[num], cell_states[num] = self.cell(x[i, num], (hidden_states[num].clone(), cell_states[num].clone()))

			o_c = self.classifier(hidden_states.clone())
			c_outs[i] = o_c

			if self.training:
				one_hots_c = one_hots_c_pred_seq[i]
			else:
				if self.use_grammar:
					one_hots_c = gep_update(o_c, grammar_gep)
				else:
					one_hots_c, history, curr_l = general_update(o_c, history, curr_l)

			o_p = self.predictor(hidden_states.clone(), one_hots_c)
			pred_outs[i] = o_p 
		
		pred_outs = pred_outs.permute(1, 0, 2, 3).contiguous()
		for i in range(len(pred_outs)):
			pred_outs[i] = output_activation(pred_outs[i])

		return pred_outs, c_outs
