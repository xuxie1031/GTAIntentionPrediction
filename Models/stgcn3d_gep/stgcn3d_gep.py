import torch
import torch.nn as nn
import torch.nn.functional as F

from st_gcn3d import *
from utils import *

class ClassifierLayer(nn.Module):
	def __init__(self, h_dim=64, c_h_dim=128, n_c=5, activation='relu', batch_norm=True, dropout=0.0):
		dims = [h_dim, c_h_dim]
		self.body = make_mlp(
			dims,
			activation=activation,
			batch_norm=batch_norm,
			dropout=dropout
		)

		self.out = nn.Linear(c_h_dim, n_c)


    def forward(self, x):
    	h = self.body(x)
    	out = self.out(h)

    	return F.softmax(out, dim=1)


class STGCN3DGEPModel(nn.Module):
	def __init__(self):
		pass