import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import time
from .gcn_vae import GCNVAE
from .graph import Graph
from .utils import *

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..', '..'))
from Dataset import *


def exec_model(dataloader, args):
    net = GCNVAE(args.in_channels, args.h_dim1, args.h_dim2, dropout=args.dropout)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    for epoch in range(args.num_epochs):
        print('****** Training beginning ******')
        loss_epoch = 0

        num_batch = 0
        for batch in dataloader:
            t_start = time.time()
            input_data_list, pred_data_list, _, num_nodes_list = batch

            