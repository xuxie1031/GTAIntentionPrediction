import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import time
from social_conv_lstm import SocialConvLSTM
from utils import *


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
            for idx in range(dataloader_train.batch_size):
                input_data = input_data_list[idx]
                pred_data = pred_data_list[idx]
                num_nodes = num_nodes_list[idx]

                if args.use_cuda:
                    input_data = input_data.cuda()
                    pred_data = pred_data.cuda()

                data = torch.cat((input_data, pred_data), dim=0)
                data, _ = data_vectorize(data)
                input_data, pred_data = data[:args.obs_len, :, :], data[args.obs_len:, :, :]

                input_data_nbrs, last_frame_mask = get_conv_mask(input_data[-1], input_data, args.units, num_nodes, args.encoder_dim, args.neighbor_size, args.grid_size)
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
            loss_batch /= dataloader_train.batch_size
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
                for idx in range(dataloader_test.batch_size):
                    input_data = input_data_list[idx]
                    pred_data = pred_data_list[idx]
                    ids = ids_list[idx]
                    num_nodes = num_nodes_list[idx]

                    if args.use_cuda:
                        input_data = input_data.cuda()
                        pred_data = pred_data.cuda()
                    
                    input_data, first_values_dict = data_vectorize(input_data)
                    input_data_nbrs, last_frame_mask = get_conv_mask(input_data[-1], input_data, args.units, num_nodes, args.encoder_dim, args.neighbor_size, args.grid_size)

                    output_data = net(input_data, input_data_nbrs, last_frame_mask)
                    ret_data = data_revert(output_data[:, :, :2], first_values_dict)
                    veh_ret_data = veh_ped_seperate(ret_data, ids)

                    # get err_batch (pred_data, ret_data)
                t_end = time.time()
                loss_batch /= dataloader_test.batch_size
                loss_epoch += loss_batch
                num_batch += 1

                print('epoch {}, batch {}, test_error = {:.6f}, time/batch = {:.3f}'.format(epoch, num_batch, err_batch, t_end-t_start))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_worker', type=int, default=4)
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
    parser.add_argument('--grid_size', type=int, default=8)
    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--pretrain_epochs', type=int, default=5)


if __name__ == '__main__':
    main()