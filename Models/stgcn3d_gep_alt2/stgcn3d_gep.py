import torch
import torch.nn as nn
import torch.nn.functional as F

from st_gcn3d import *
from utils import *

class PredictionLayer(nn.Module):
	def __init__(self, emb_dim, e_h_dim, out_dim=5, activation='relu', batch_norm=True):
		pass