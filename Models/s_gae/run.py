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

            loss_batch = 0.0
            for idx in range(dataloader.batch_size):
                input_data = input_data_list[idx]
                pred_data = pred_data_list[idx]
                data = torch.cat((input_data, pred_data), dim=0)

                num_nodes = num_nodes_list[idx]

                data, _ = data_vectorize(data)
                inputs = data_feeder(data)

                g = Graph(data)
                As = g.normalize_undigraph()
                pos_weights = g.graph_pos_weights()
                norms = g.graph_norms()
                targets = g.graph_As()

                if args.use_cuda:
                    inputs = inputs.cuda()
                    As = As.cuda()
                    pos_weights = pos_weights.cuda()
                    norms = norms.cuda()
                    targets = targets.cuda()

                recovered, mu, logvar = net(inputs, As)
                loss = vae_loss(recovered, targets, mu, logvar, num_nodes, norms, pos_weights)
                loss_batch += loss.item() / dataloader.batch_size

                optimizer.zero_grad()
                loss.backward()

                # torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
                optimizer.step()
            t_end = time.time()
            loss_epoch += loss_batch
            num_batch += 1

            print('epoch {}, batch {}, train_loss = {:.6f}, time/batch = {:.3f}'.format(epoch, num_batch, loss_batch, t_end-t_start))
        
        loss_epoch /= num_batch
        print('epoch {}, train_loss = {:.6f\n}'.format(epoch, loss_epoch))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--obs_len', type=int, default=10)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--in_channels', type=int, default=4)
    parser.add_argument('--h_dim1', type=int, default=128)
    parser.add_argument('--h_dim2', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--grad_clip', type=float, default=10.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--dset_name', type=str, default='GTADataset')
    parser.add_argument('--dset_tag', type=str, default='GTAS')
    parser.add_argument('--dset_feature', type=int, default=4)
    parser.add_argument('--frame_skip', type=int, default=1)

    args = parser.parse_args()

    _, d_loader = data_loader(args, os.path.join(os.getcwd(), '..', '..', 'Dataset', 'dataset', args.dset_name, args.dset_tag, 'train'))

    exec_model(d_loader, args)

if __name__ == '__main__':
    main()