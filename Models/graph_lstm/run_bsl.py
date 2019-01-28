import torch

import argparse
import time
from graph_lstm_bsl import GraphLSTMBSL
from utils import *


def sample(net, input_data, pred_data, num_nodes, args):
    with torch.no_grad():
        cell_hidden_state_tuple = (torch.zeros(num_nodes, args.cell_hidden_size), torch.zeros(num_nodes, args.cell_hidden_size))
        graph_hidden_state_tuple = (torch.zeros(num_nodes, args.graph_hidden_size), torch.zeros(num_nodes, args.graph_hidden_size))
        if args.use_cuda:
            cell_hidden_state_tuple = (cell_hidden_state_tuple[0].cuda(), cell_hidden_state_tuple[1].cuda())
            graph_hidden_state_tuple = (graph_hidden_state_tuple[0].cuda(), graph_hidden_state_tuple[1].cuda())
        
        output_data = torch.zeros(args.pred_len, num_nodes, args.input_size)
        if args.use_cuda:
            output_data = output_data.cuda()
        
        for tstep in range(args.obs_len):
            output_obs, cell_hidden_state_tuple, graph_hidden_state_tuple = \
                net([input_data[tstep]], cell_hidden_state_tuple, graph_hidden_state_tuple)
            
            mux, muy, sx, sy, corr = get_coef(output_obs)
            next_x, next_y = sample_gaussian_2d(mux, muy, sx, sy, corr)
        
        for tstep in range(args.pred_len):
            output_data[tstep, :, 0] = next_x
            output_data[tstep, :, 1] = next_y

            curr_data = output_data[tstep]

            outputs, cell_hidden_state_tuple, graph_hidden_state_tuple = \
                net([curr_data], cell_hidden_state_tuple, graph_hidden_state_tuple)
            
            mux, muy, sx, sy, corr = get_coef(outputs)
            next_x, next_y = sample_gaussian_2d(mux, muy, sx, sy, corr)

        return output_data


def exec_model(dataloader_train, dataloader_test, args):
    net = GraphLSTMBSL(args)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

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

                input_data = torch.cat((input_data, pred_data), dim=0)
                if args.use_cuda:
                    input_data = input_data.cuda()
                input_data, _ = data_vectorize(input_data)

                cell_hidden_state_tuple = (torch.zeros(num_nodes, args.cell_hidden_size), torch.zeros(num_nodes, args.cell_hidden_size))
                graph_hidden_state_tuple = (torch.zeros(num_nodes, args.graph_hidden_size), torch.zeros(num_nodes, args.graph_hidden_size))
                if args.use_cuda:
                    cell_hidden_state_tuple = (cell_hidden_state_tuple[0].cuda(), cell_hidden_state_tuple[1].cuda())
                    graph_hidden_state_tuple = (graph_hidden_state_tuple[0].cuda(), graph_hidden_state_tuple[1].cuda())

                net.zero_grad()
                optimizer.zero_grad()

                outputs, _, _ = net(input_data, cell_hidden_state_tuple, graph_hidden_state_tuple, args.obs_len+args.pred_len, num_nodes)

                loss = gaussian_likelihood_2d(outputs, input_data)
                loss_batch += loss

                loss.backward()

                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
                optimizer.step()

            t_end = time.time()
            loss_batch /= dataloader_train.batch_size
            loss_epoch += loss_batch
            num_batch += 1

            print('epoch {}, batch {}, train_loss = {:.6f}, time/batch = {:.3f}'.format(epoch, num_batch, loss_batch, t_end-t_start))

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
                    
                    input_data, first_value_dict = data_vectorize(input_data)
                    ret_seq = sample(net, input_data, pred_data, num_nodes, args)
                    ret_seq = data_revert(ret_seq, first_value_dict)

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
    parser.add_argument('--dyn_embedding_size', type=int, default=32)
    parser.add_argument('--graph_embedding_size', type=int, default=32)
    parser.add_argument('--graph_hidden_size', type=int, default=64)
    parser.add_argument('--mat_hidden_size', type=int, default=64)
    parser.add_argument('--cell_hidden_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=.003)
    parser.add_argument('--grad_clip', type=float, default=10.0)
    parser.add_argument('--gru', action='store_true', default=False)
    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--num_epochs', type=int, default=30)


if __name__ == '__main__':
    main()