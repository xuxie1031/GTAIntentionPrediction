import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import time
from social_conv_lstm import SocialConvLSTM
from utils import *

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..', '..'))
from DataSet import *


def exec_model(dataloader_train, dataloader_test, args):
    net = SocialConvLSTM(args.obs_len, args.pred_len, args.input_dim, args.output_dim, args.encoder_dim, args.decoder_dim, 
                        args.dyn_embedding_dim, args.input_embedding_dim)
    
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    for epoch in range(args.num_epochs):
        print('****** Training beginning ******')
        loss_epoch = 0

        num_batch = 0
        for batch in dataloader_train:
            t_start = time.time()
            input_data_list, pred_data_list, _, num_nodes_list = batch

            loss_batch = 0
            for idx in range(args.batch_size):
                input_data = input_data_list[idx]
                pred_data = pred_data_list[idx]
                num_nodes = num_nodes_list[idx]

                if args.use_cuda:
                    input_data = input_data.cuda()
                    pred_data = pred_data.cuda()

                data = torch.cat((input_data, pred_data), dim=0)
                data, _ = data_vectorize(data)
                input_data, pred_data = data[:args.obs_len, :, :], data[args.obs_len:, :, :]

                input_data_nbrs, last_frame_mask = get_conv_mask(input_data[-1], input_data, num_nodes, args.encoder_dim, args.neighbor_size, args.grid_size, use_cuda=args.use_cuda)
                if args.use_cuda:
                    # input_data_nbrs = input_data_nbrs.cuda()
                    last_frame_mask = last_frame_mask.cuda()

                pred_out = net(input_data, input_data_nbrs, last_frame_mask)
                if epoch < args.pretrain_epochs:
                    loss = mse_loss(pred_out, pred_data)
                else:
                    loss = nll_loss(pred_out, pred_data)
                loss_batch += loss

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
                optimizer.step()
            
            t_end = time.time()
            loss_batch /= args.batch_size
            loss_epoch += loss_batch
            num_batch += 1

            print('epoch {}, batch {}, train_loss = {:.6f}, time/batch ={:.3f}'.format(epoch, num_batch, loss_batch, t_end-t_start))
        
        loss_epoch /= num_batch
        print('epoch {}, train_loss = {:.6f}\n'.format(epoch, loss_epoch))
                
        print('****** Testing beginning ******')
        err_epoch = 0.0

        with torch.no_grad():
            num_batch = 0
            for batch in dataloader_test:
                t_start = time.time()
                input_data_list, pred_data_list, ids_list, num_nodes_list = batch

                err_batch = 0.0
                for idx in range(args.batch_size):
                    input_data = input_data_list[idx]
                    pred_data = pred_data_list[idx]
                    ids = ids_list[idx]
                    num_nodes = num_nodes_list[idx]

                    if args.use_cuda:
                        input_data = input_data.cuda()
                        pred_data = pred_data.cuda()
                        ids = ids.cuda()
                    
                    input_data, first_values_dict = data_vectorize(input_data)
                    input_data_nbrs, last_frame_mask = get_conv_mask(input_data[-1], input_data, num_nodes, args.encoder_dim, args.neighbor_size, args.grid_size, use_cuda=args.use_cuda)

                    if args.use_cuda:
                        last_frame_mask = last_frame_mask.cuda()

                    output_data = net(input_data, input_data_nbrs, last_frame_mask)
                    ret_data = data_revert(output_data[:, :, :2], first_values_dict)

                    veh_ret_data, _ = veh_ped_seperate(ret_data, ids)
                    veh_pred_data, _ = veh_ped_seperate(pred_data, ids)

                    # error = displacement_error(veh_ret_data, veh_pred_data)
                    error = final_displacement_error(veh_ret_data[-1], veh_pred_data[-1])

                    err_batch += error.item()
                t_end = time.time()
                err_batch /= args.batch_size
                err_epoch += err_batch
                num_batch += 1

                print('epoch {}, batch {}, test_error = {:.6f}, time/batch = {:.3f}'.format(epoch, num_batch, err_batch, t_end-t_start))
            
            err_epoch /= num_batch
            print('epoch {}, test_err = {:.6f}\n'.format(epoch, err_epoch))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--output_dim', type=int, default=5)
    parser.add_argument('--input_embedding_dim', type=int, default=32)
    parser.add_argument('--dyn_embedding_dim', type=int, default=32)
    parser.add_argument('--encoder_dim', type=int, default=64)
    parser.add_argument('--decoder_dim', type=int, default=128)
    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--grad_clip', type=float, default=10.0)
    parser.add_argument('--lr', type=float, default=.003)
    parser.add_argument('--neighbor_size', type=int, default=32)
    parser.add_argument('--grid_size', type=int, default=8)    # fix for conv
    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--pretrain_epochs', type=int, default=5)

    args = parser.parse_args()

    _, train_loader = data_loader(args, os.path.join(os.getcwd(), '..', '..', 'DataSet', 'dataset', 'train'))
    _, test_loader = data_loader(args, os.path.join(os.getcwd(), '..', '..', 'DataSet', 'dataset', 'test'))

    exec_model(train_loader, test_loader, args)


if __name__ == '__main__':
    main()