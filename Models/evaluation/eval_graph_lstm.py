import torch

import argparse
import time

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..', 'graph_lstm'))
from glob import glob

from utils import *
from eval_utils import *
from graph_lstm import GraphLSTM


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


def eval_model(data_dir, net, args):
    all_files = os.listdir(data_dir)
    all_files = [os.path.join(data_dir, path) for path in all_files]
    all_eval_files = [os.path.join(data_dir, path+'_eval') for path in all_files]

    for eid, file in enumerate(all_files):
        seq_list, frame_seq_list, ids_seq_list, num_nodes_seq_list = parse_file_trajs(file, args)

        with open(all_eval_files[eid], 'w') as eval_f:
            num_seq = len(frame_seq_list)

            for idx in range(num_seq):
                data = seq_list[idx]
                frames = frame_seq_list[idx]
                ids = ids_seq_list[idx]
                num_nodes = num_nodes_seq_list[idx]

                data = torch.from_numpy(data).type(torch.float).permute(2, 0, 1)
                ids = torch.from_numpy(ids)

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
                k_ret_seq = torch.zeros(args.best_k, args.pred_len, veh_num_nodes, args.input_size)
                if args.use_cuda:
                    k_ret_seq = k_ret_seq.cuda()

                veh_pred_data = data_revert(veh_pred_data, first_values_dict)

                error = 0.0
                for k in range(args.best_k):
                    ret_seq = sample(net, veh_input_data, ped_input_data, ped_pred_data, veh_num_nodes, ped_num_nodes, args)
                    ret_seq = data_revert(ret_seq, first_values_dict)

                    k_ret_seq[k] = ret_seq
                    error += final_displacement_error(ret_seq[-1], veh_pred_data[-1])
                error /= args.best_k
                print('file {}, seq_id {}, error = {}'.format(all_eval_files[eid], idx, error.item()))

                k_ret_seq = k_ret_seq.cpu().numpy()

                veh_ids = ids.cpu().numpy()
                veh_ids = veh_ids[veh_ids < 100]

                pred_frame = frames[-args.pred_len]
                for i in range(len(veh_ids)):
                    for k in range(args.best_k):
                        # eval_f.write(str(pred_frame)+','+str(veh_ids[i])+','+str(k_ret_seq[k, -1, i, 0])+','+str(k_ret_seq[k, -1, i, 1])+'\n')
                        for t in range(args.pred_len):
                            eval_f.write(str(pred_frame)+','+str(veh_ids[i])+','+str(k_ret_seq[k, t, i, 0])+','+str(k_ret_seq[k, t, i, 1])+'\n')
        eval_f.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=5)
    parser.add_argument('--dyn_veh_embedding_size', type=int, default=32)
    parser.add_argument('--dyn_ped_embedding_size', type=int, default=32)
    parser.add_argument('--graph_veh_embedding_size', type=int, default=32)
    parser.add_argument('--graph_ped_embedding_size', type=int, default=32)
    parser.add_argument('--graph_veh_hidden_size', type=int, default=64)
    parser.add_argument('--graph_ped_hidden_size', type=int, default=64)
    parser.add_argument('--mat_veh_hidden_size', type=int, default=64)
    parser.add_argument('--mat_ped_hidden_size', type=int, default=64)
    parser.add_argument('--cell_hidden_size', type=int, default=256)
    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--best_k', type=int, default=3)
    parser.add_argument('--lr', type=float, default=.003)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--gru', action='store_true', default=True)
    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--path_checkpt', type=str, default='../graph_lstm/')

    args = parser.parse_args()

    net = GraphLSTM(args)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    model_name = 'graph_lstm.pth.tar'
    state = torch.load(os.path.join(args.path_checkpt, model_name))
    net.load_state_dict(state['network_dict'])
    optimizer.load_state_dict(state['opt_dict'])

    data_dir = os.path.join(os.getcwd(), '..', '..', 'DataSet', 'dataset', 'eval')
    for file in glob(os.path.join(data_dir, '*_eval')):
        os.remove(file)

    eval_model(data_dir, net, args)


if __name__ == '__main__':
    main()