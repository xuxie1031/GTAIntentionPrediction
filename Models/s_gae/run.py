import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import time
from gcn_vae import GCNVAE
from graph import Graph
from utils import *

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..', '..'))
from DataSet import *


def exec_model(dataloader, args):
    dev = torch.device('cuda:0')
    net = GCNVAE(args.in_channels, args.h_dim1, args.h_dim2, dropout=args.dropout, use_cuda=args.use_cuda, device=dev)
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
                    inputs = inputs.to(dev)
                    As = As.to(dev)
                    pos_weights = pos_weights.to(dev)
                    norms = norms.to(dev)
                    targets = targets.to(dev)

                recovered, mu, logvar = net(inputs, As)
                loss = vae_loss(recovered, targets, mu, logvar, num_nodes, norms, pos_weights, dev)
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
        print('epoch {}, train_loss = {:.6f}'.format(epoch, loss_epoch))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=20)
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
    parser.add_argument('--num_epochs', type=int, default=30)

    args = parser.parse_args()

    _, d_loader = data_loader(args, os.path.join(os.getcwd(), '..', '..', 'DataSet', 'dataset', args.dset_name, args.dset_tag, 'train'))
    print(len(d_loader))

    exec_model(d_loader, args)

if __name__ == '__main__':
    main()
