import torch
import torch.nn as nn

from graph_full import Graph
from utils import *

class GraphConvNet2D(nn.Module):
	def __init__(self, in_channels, out_channels, s_kernel_size=1, t_kernel_size=1, t_padding=0, t_dilation=1, bias=True):
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

		x = torch.einsum('nkctv,nkvw->nctw', (x, A))

		return x, A


class ST_GCN2D(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0, residual=True, apply_gcn=False):
		super(ST_GCN2D, self).__init__()

		assert len(kernel_size) == 2
		assert kernel_size[0] % 2 == 1
		# padding = ((kernel_size[0]-1) // 2, 0)

		self.apply_gcn = apply_gcn
		self.gcn = GraphConvNet2D(in_channels, out_channels, kernel_size[1])

		tcn_in = out_channels if apply_gcn else in_channels

		self.tcn = nn.Sequential(
			# nn.BatchNorm2d(tcn_in),
			# nn.ReLU(inplace=True),
			# nn.LeakyReLU(0.1, inplace=True),
			nn.Conv2d(
				tcn_in,
				out_channels,
				(kernel_size[0], 1),
				(stride, 1),
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
					tcn_in,
					out_channels,
					kernel_size=1,
					stride=(stride, 1)
				),
				# nn.BatchNorm2d(out_channels)
			)

		self.leaky_relu = nn.LeakyReLU(0.1)


	def forward(self, x, A):
		res = self.residual(x)
		if self.apply_gcn:
			x, A = self.gcn(x, A)
		x = self.leaky_relu(self.tcn(x)+res)

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
			ST_GCN2D(enc_hidden_size, 64, kernel_size, stride=1, residual=False, **kwargs0),
			ST_GCN2D(64, 16, kernel_size, stride=1, **kwargs),
		))

		self.dyn = nn.Linear(in_channels, dyn_hidden_size)

		self.enc = nn.LSTM(dyn_hidden_size, enc_hidden_size)
		if gru:
			self.enc = nn.GRU(dyn_hidden_size, enc_hidden_size)

		self.hidden = nn.Linear(enc_hidden_size, self_hidden_size)

		self.dec = nn.LSTM(96+self_hidden_size, dec_hidden_size)
		if gru:
			self.dec = nn.GRU(96+self_hidden_size, dec_hidden_size)

		self.output = nn.Linear(dec_hidden_size, out_dim)

		self.leaky_relu = nn.LeakyReLU(0.1)

		if use_cuda:
			self.to(device)


	def forward(self, x, ngbrs, A):
		N = x.size(1)

		x = self.leaky_relu(self.dyn(x))
		ngbrs = self.leaky_relu(self.dyn(ngbrs))

		_, tup_enc = self.enc(x)
		x = tup_enc[0].view(N, self.enc_dim)

		ngbrs, _ = self.enc(ngbrs)
		T, NV, C = ngbrs.size()
		ngbrs = ngbrs.view(T, V, NV // V, C)
		ngbrs = ngbrs.permute(1, 3, 0, 2)

		for gcn in self.st_gcn2d_modules:
			ngbrs, _ = gcn(ngbrs, A)

		_, C, T, V = ngbrs.size()
		data_pool = nn.MaxPool2d((2, V), padding=(1, 0))
		ngbrs = data_pool(ngbrs)
		ngbrs = ngbrs.view(N, -1)

		x = self.leaky_relu(self.hidden(x))

		x = torch.cat((x, ngbrs), 1)
		
		x = x.repeat(self.pred_len, 1, 1)
		h_dec = self.dec(x)
		o = self.output(h_dec)

		return o