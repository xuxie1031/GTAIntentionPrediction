import torch
import torch.nn as nn

# from graph_full import Graph
from utils import *


class STSocialModel(nn.Module):
	def __init__(self, pred_len, in_channels, dyn_hidden_size, self_hidden_size, enc_hidden_size, dec_hidden_size, out_dim, gru=False, use_cuda=True, device=None, **kwargs):
		super(STSocialModel, self).__init__()

		self.enc_dim = enc_hidden_size
		self.soc_size = 36
		self.pred_len = pred_len
		self.out_dim = out_dim
		self.device = device

		self.dyn = nn.Linear(in_channels, dyn_hidden_size)

		self.enc = nn.LSTM(dyn_hidden_size, enc_hidden_size)
		if gru:
			self.enc = nn.GRU(dyn_hidden_size, enc_hidden_size)

		self.convs = nn.Sequential(
			nn.Conv2d(enc_hidden_size, 64, kernel_size=3, stride=1),
			nn.LeakyReLU(0.1, inplace=True),
			nn.Conv2d(64, 16, kernel_size=3, stride=1),
			nn.LeakyReLU(0.1, inplace=True),
                        nn.Conv2d(16, 4, kernel_size=3, stride=1),
                        nn.LeakyReLU(0.1, inplace=True),
		)

		self.pool = nn.MaxPool2d((2, 2))

		self.hidden = nn.Linear(enc_hidden_size, self_hidden_size)

		self.dec = nn.LSTM(self.soc_size+self_hidden_size, dec_hidden_size)
		if gru:
			self.dec = nn.GRU(self.soc_size+self_hidden_size, dec_hidden_size)

		self.output = nn.Linear(dec_hidden_size, out_dim)

		self.leaky_relu = nn.LeakyReLU(0.1)

		if use_cuda:
			self.to(device)


	def forward(self, x, ngbrs, masks):
		x = self.leaky_relu(self.dyn(x))
		ngbrs = self.leaky_relu(self.dyn(ngbrs))

		_, tup_enc = self.enc(x)
		x = tup_enc[0].view(-1, self.enc_dim)

		_, tup_enc = self.enc(ngbrs)
		ngbrs = tup_enc[0].view(-1, self.enc_dim)

		soc_enc = torch.zeros_like(masks).float()
		soc_enc = soc_enc.masked_scatter_(masks, ngbrs)
                soc_enc = soc_enc.permute(0, 3, 2, 1)

		soc_enc = self.pool(self.convs(soc_enc))
		soc_enc = soc_enc.view(-1, self.soc_size)

		x = self.leaky_relu(self.hidden(x))

		x = torch.cat((soc_enc, x), 1)
		
		x = x.repeat(self.pred_len, 1, 1)
		h_dec, _ = self.dec(x)
		o = self.output(h_dec)

		return output_activation(o)
