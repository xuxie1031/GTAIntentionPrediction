import torch

import argparse
import time
from social_lstm import SocialLSTM
from grid import *
from utils import *

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..', '..'))
from DataSet import *


def sample(net, input_data, input_grids, units, num_nodes, args):
    with torch.no_grad():
        hidden_states = torch.zeros(num_nodes, net.hidden_size)
        if args.use_cuda:
            hidden_states = hidden_states.cuda()
        
        if not args.gru:
            cell_states = torch.zeros(num_nodes, net.hidden_size)
            if args.use_cuda:
                cell_states = cell_states.cuda()
        else:
            cell_states = None

        output_data = torch.zeros(args.pred_len, num_nodes, args.input_size)
        if args.use_cuda:
            output_data = output_data.cuda()

        for tstep in range(args.obs_len):
            output_obs, hidden_states, cell_states = net([input_data[tstep]], [input_grids[tstep]], hidden_states, cell_states, 1, num_nodes)

            mux, muy, sx, sy, corr = get_coef(output_obs)
            next_x, next_y = sample_gaussian_2d(mux, muy, sx, sy, corr)

        for tstep in range(args.pred_len):
            output_data[tstep, :, 0] = next_x
            output_data[tstep, :, 1] = next_y

            curr_data = output_data[tstep]
            curr_grid = get_grid_mask(curr_data, units, num_nodes, args.neighbor_size, args.grid_size)
            curr_grid = torch.from_numpy(curr_grid).float()
            if args.use_cuda:
                curr_grid = curr_grid.cuda()

            outputs, hidden_states, cell_states = net([curr_data], [curr_grid], hidden_states, cell_states, 1, num_nodes)
            mux, muy, sx, sy, corr = get_coef(outputs)
            next_x, next_y = sample_gaussian_2d(mux, muy, sx, sy, corr)

        return output_data


def exec_model(dataloader_train, dataloader_test, args):
    net = SocialLSTM(args)

    optimizer = torch.optim.Adagrad(net.parameters(), weight_decay=args.lambda_param)
    lr = args.lr

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
                input_data = torch.cat((input_data, pred_data), dim=0)

                num_nodes = num_nodes_list[idx]

                # raw data process
                if args.use_cuda:
                    input_data = input_data.cuda()
                grids = get_grid_mask_seq(input_data, dataloader_train.units, args.neighbor_size, args.grid_size, args.use_cuda)
                input_data, _ = data_vectorize(input_data)

                hidden_states = torch.zeros(num_nodes, args.hidden_size)
                if args.use_cuda:
                    hidden_states = hidden_states.cuda()

                cell_states = torch.zeros(num_nodes, args.hidden_size)
                if args.use_cuda:
                    cell_states = cell_states.cuda()
                
                net.zero_grad()
                optimizer.zero_grad()

                outputs, _, _ = net(input_data, grids, hidden_states, cell_states, args.obs_len, num_nodes)
                
                loss = gaussian_likelihood_2d(outputs, input_data)
                loss_batch += loss.item()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
                optimizer.step()
            
            t_end = time.time()
            loss_batch /= dataloader_train.batch_size
            loss_epoch += loss_batch
            num_batch += 1

            print('epoch {}, batch {}, train_loss = {:.6f}, time/batch = {:.3f}'.format(epoch, num_batch, loss_batch, t_end-t_start))
        
        loss_epoch /= num_batch
        print('epoch {}, train_loss = {:.6f}\n'.format(epoch, loss_epoch))

        print('****** Testing beginning ******')
        err_epoch = 0.0

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
                    ids = ids.cuda()
                grids = get_grid_mask_seq(input_data, dataloader_test.units, args.neighbor_size, args.grid_size, args.use_cuda)
                input_data, first_values_dict = data_vectorize(input_data)

                ret_seq = sample(net, input_data, grids, dataloader_test.units, num_nodes, args)
                ret_seq = data_revert(ret_seq, first_values_dict)

                veh_ret_seq, _ = veh_ped_seperate(ret_seq, ids)
                veh_pred_seq, _ = veh_ped_seperate(pred_data, ids)
                
                error = displacement_error(veh_ret_seq, veh_pred_seq)
                # error = final_displacement_error(veh_ret_seq, veh_pred_seq)
                err_batch += error.item()
            
            t_end = time.time()
            err_batch /= dataloader_test.batch_size
            err_epoch += err_batch
            num_batch += 1

            print('epoch {}, batch {}, test_error = {:.6f}, time/batch = {:.3f}'.format(epoch, num_batch, err_batch, t_end-t_start))

        err_epoch /= num_batch
        print('epoch {}, test_err = {:.6f}\n'.format(epoch, err_epoch))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=5)
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--grad_clip', type=float, default=10.0)
    parser.add_argument('--lr', type=float, default=.003)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--neighbor_size', type=int, default=32)
    parser.add_argument('--grid_size', type=int, default=4)
    parser.add_argument('--lambda_param', type=float, default=.0005)
    parser.add_argument('--freq_optimizer', type=int, default=8)
    parser.add_argument('--gru', action='store_true', default=False)
    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--num_epochs', type=int, default=30)

    args = parser.parse_args()

    _, train_loader = data_loader(args, os.path.join(os.getcwd(), '..', '..', 'DataSet', 'dataset', 'train'))
    _, test_loader = data_loader(args, os.path.join(os.getcwd(), '..', '..', 'DataSet', 'dataset', 'test'))

    exec_model(train_loader, test_loader, args)


if __name__ == '__main__':
    main()