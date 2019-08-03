import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import time
from .st_gcn3d import STGCN3DModel
from .graph import Graph
from .utils import *

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..', '..'))
from Dataset import *


def exec_model(dataloader_train, dataloader_test, args):
	net = STGCN3DModel(args.pred_len, args.in_channels, args.spatial_kernel, args.temporal_kernel, args.dec_hidden_size, args.out_dim, args.use_cuda, dropout=args.dropout)
	optimizer = optim.Adam(net.parameters(), lr=args.lr)

	err_epochs = []
	for epoch in range(args.num_epochs):
		print('****** Training beginning ******')
		loss_epoch = 0

		num_batch = 0
		for batch in dataloader_train:
			pass