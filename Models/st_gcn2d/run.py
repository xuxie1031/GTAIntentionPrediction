import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import time
from .st_gcn2d import STGCN2DModel
from utils import *

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..', '..'))
from DataSet import *
from graph import *


def exec_model(dataloader_train, dataloader_test, args):
    net = STGCN2DModel(args.pred_len, args.in_channels, args.spatial_kernel_size, args.temporal_kernel_size, args.dec_hidden_size, args.out_dim, args.use_cuda, args.dropout)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    err_epochs = []
    for epoch in range(args.num_epochs):
        print('****** Training beginning ******')
        loss_epoch = 0

        num_batch = 0
        for batch in dataloader_train:
            t_start = time.time()
            input_data_list, pred_data_list, _, num_nodes_list = batch

            loss_batch = 0.0
            num2input_dict, num2pred_dict = data_batch(input_data_list, pred_data_list, num_list)
            for num in num2input_dict.keys():
                batch_input_data, batch_pred_data = torch.stack(num2input_dict[num]), torch.stack(num2pred_dict[num])

                batch_data = torch.cat((batch_input_data, batch_pred_data), dim=1)
                batch_data, _ = data_vectorize(batch_data)
                batch_input_data, batch_pred_data = batch_data[:, :-args.pred_len, :, :], batch_data[:, -args.pred_len:, :, :]

                inputs = data_feeder(batch_input_data)

                g = Graph(batch_input_data[:, 0, :, :])
                As = g.normalize_undigraph()

                if args.use_cuda:
                    inputs = inputs.cuda()
                    As = As.cuda()
                
                preds = net(inputs, As)
                for i in range(len(preds)):
                    if epoch < args.pretrain_epochs:
                        pass