import torch

import argparse
import time
from graph_lstm import GraphLSTM
from utils import *

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..', '..'))
from DataSet import *


def sample(net, veh_input_data, ped_input_data, ped_pred_data, veh_num_nodes, ped_num_nodes, args):
    with torch.no_grad():
        cell_hidden_state_tuple = (torch.zeros(veh_num_nodes, args.cell_hidden_size), torch.zeros(veh_num_nodes, args.cell_hidden_size))
        graph_veh_hidden_state_tuple = (torch.zeros(veh_num_nodes, args.graph_veh_hidden_size), torch.zeros(veh_num_nodes, args.graph_veh_hidden_size))
        graph_ped_hidden_state_tuple = (torch.zeros(ped_num_nodes, args.graph_ped_hidden_size), torch.zeros(ped_num_nodes, args.graph_ped_hidden_size))
        if args.use_cuda:
            cell_hidden_state_tuple = (cell_hidden_state_tuple[0].cuda(), cell_hidden_state_tuple[1].cuda())
            graph_veh_hidden_state_tuple = (graph_veh_hidden_state_tuple[0].cuda(), graph_veh_hidden_state_tuple[1].cuda())
            graph_ped_hidden_state_tuple = (graph_ped_hidden_state_tuple[0].cuda(), graph_ped_hidden_state_tuple[1].cuda())

        output_data = torch.zeros(args.pred_len, veh_num_nodes, args.input_size)
        if args.use_cuda:
            output_data = output_data.cuda()
        
        for tstep in range(args.obs_len):
            output_obs, cell_hidden_state_tuple, graph_veh_hidden_state_tuple, graph_ped_hidden_state_tuple = \
                        net([veh_input_data[tstep]], [ped_input_data[tstep]], cell_hidden_state_tuple, graph_veh_hidden_state_tuple, graph_ped_hidden_state_tuple, 1, veh_num_nodes)

            mux, muy, sx, sy, corr = get_coef(output_obs)
            next_x, next_y = sample_gaussian_2d(mux, muy, sx, sy, corr)
        
        for tstep in range(args.pred_len):
            output_data[tstep, :, 0] = next_x
            output_data[tstep, :, 1] = next_y

            curr_data = output_data[tstep]
            outputs, cell_hidden_state_tuple, graph_veh_hidden_state_tuple, graph_ped_hidden_state_tuple = \
                        net([curr_data], [ped_pred_data[tstep]], cell_hidden_state_tuple, graph_veh_hidden_state_tuple, graph_ped_hidden_state_tuple, 1, veh_num_nodes)
            
            mux, muy, sx, sy, corr = get_coef(outputs)
            next_x, next_y = sample_gaussian_2d(mux, muy, sx, sy, corr)
        
        return output_data


def exec_model(dataloader_train, dataloader_test, args):
    net = GraphLSTM(args)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1.0)

    err_epochs = []
    for epoch in range(args.num_epochs):
        print('****** Training beginning ******')
        scheduler.step()
        loss_epoch = 0

        num_batch = 0
        for batch in dataloader_train:
            t_start = time.time()
            input_data_list, pred_data_list, ids_list, num_nodes_list = batch

            loss_batch = 0
            for idx in range(args.batch_size):
                input_data = input_data_list[idx]
                pred_data = pred_data_list[idx]
                ids = ids_list[idx]
                num_nodes = num_nodes_list[idx]

                input_data = torch.cat((input_data, pred_data), dim=0)
                if args.use_cuda:
                    input_data = input_data.cuda()
                    ids = ids.cuda()
                input_data, _ = data_vectorize(input_data)

                veh_data, ped_data = veh_ped_seperate(input_data, ids)
                veh_num_nodes, ped_num_nodes = veh_data.size(1), ped_data.size(1)

                cell_hidden_state_tuple = (torch.zeros(veh_num_nodes, args.cell_hidden_size), torch.zeros(veh_num_nodes, args.cell_hidden_size))
                graph_veh_hidden_state_tuple = (torch.zeros(veh_num_nodes, args.graph_veh_hidden_size), torch.zeros(veh_num_nodes, args.graph_veh_hidden_size))
                graph_ped_hidden_state_tuple = (torch.zeros(ped_num_nodes, args.graph_ped_hidden_size), torch.zeros(ped_num_nodes, args.graph_ped_hidden_size))
                if args.use_cuda:
                    cell_hidden_state_tuple = (cell_hidden_state_tuple[0].cuda(), cell_hidden_state_tuple[1].cuda())
                    graph_veh_hidden_state_tuple = (graph_veh_hidden_state_tuple[0].cuda(), graph_veh_hidden_state_tuple[1].cuda())
                    graph_ped_hidden_state_tuple = (graph_ped_hidden_state_tuple[0].cuda(), graph_ped_hidden_state_tuple[1].cuda())

                net.zero_grad()
                optimizer.zero_grad()

                outputs, _, _, _ = net(veh_data, ped_data, cell_hidden_state_tuple, graph_veh_hidden_state_tuple, graph_ped_hidden_state_tuple, args.obs_len+args.pred_len, veh_num_nodes)

                loss = gaussian_likelihood_2d(outputs, veh_data)
                loss_batch += loss

                loss.backward()

                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
                optimizer.step()

            t_end = time.time()
            loss_batch /= args.batch_size
            loss_epoch += loss_batch
            num_batch += 1

            print('epoch {}, batch {}, train_loss = {:.6f}, time/batch = {:.3f}'.format(epoch, num_batch, loss_batch, t_end-t_start))

        loss_epoch /= num_batch
        print('epoch {}, train_loss = {:.6f}\n'.format(epoch, loss_epoch))

        print('****** Testing beginning *******')
        err_epoch = 0.0

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

                data = torch.cat((input_data, pred_data), dim=0)
                if args.use_cuda:
                    data = data.cuda()
                    ids = ids.cuda()
                
                veh_data, ped_data = veh_ped_seperate(data, ids)
                veh_data, first_values_dict = data_vectorize(veh_data)
                veh_input_data = veh_data[:args.obs_len, :, :]
                veh_pred_data = veh_data[args.obs_len:, :, :]

                ped_data, _ = data_vectorize(ped_data)
                ped_input_data = ped_data[:args.obs_len, :, :]
                ped_pred_data = ped_data[args.obs_len:, :, :]

                veh_num_nodes, ped_num_nodes = veh_input_data.size(1), ped_input_data.size(1)
                ret_seq = sample(net, veh_input_data, ped_input_data, ped_pred_data, veh_num_nodes, ped_num_nodes, args)
                
                veh_pred_data = data_revert(veh_pred_data, first_values_dict)
                ret_seq = data_revert(ret_seq, first_values_dict)

                # error = displacement_error(ret_seq, veh_pred_data)
                error = final_displacement_error(ret_seq[-1], veh_pred_data[-1])

                err_batch += error.item()
            
            t_end = time.time()
            err_batch /= args.batch_size
            err_epoch += err_batch
            num_batch += 1

            print('epoch {}, batch {}, test_error = {:.6f}, time/batch = {:.3f}'.format(epoch, num_batch, err_batch, t_end-t_start))

        err_epoch /= num_batch
        err_epochs.append(err_epoch)
        print('epoch {}, test_err = {:.6f}\n'.format(epoch, err_epoch))
        print(err_epochs)

# lr:.003, grad_clip: 1.0, dropout: 0.5
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=5)
    parser.add_argument('--dyn_veh_embedding_size', type=int, default=8)
    parser.add_argument('--dyn_ped_embedding_size', type=int, default=8)
    parser.add_argument('--graph_veh_embedding_size', type=int, default=8)
    parser.add_argument('--graph_ped_embedding_size', type=int, default=8)
    parser.add_argument('--graph_veh_hidden_size', type=int, default=16)
    parser.add_argument('--graph_ped_hidden_size', type=int, default=16)
    parser.add_argument('--mat_veh_hidden_size', type=int, default=16)
    parser.add_argument('--mat_ped_hidden_size', type=int, default=16)
    parser.add_argument('--cell_hidden_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=.003)
    parser.add_argument('--grad_clip', type=float, default=10.0)
    parser.add_argument('--gru', action='store_true', default=True)
    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--num_epochs', type=int, default=1000)

    args = parser.parse_args()

    _, train_loader = data_loader(args, os.path.join(os.getcwd(), '..', '..', 'DataSet', 'dataset', 'train'))
    _, test_loader = data_loader(args, os.path.join(os.getcwd(), '..', '..', 'DataSet', 'dataset', 'test'))

    exec_model(train_loader, test_loader, args)


if __name__ == '__main__':
    main()