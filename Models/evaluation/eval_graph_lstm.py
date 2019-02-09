import torch

import argparse
import time

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..', 'graph_lstm'))

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
                error = final_displacement_error(ret_seq[-1], veh_pred_data[-1])
                print('file {}, seq_id {}, error = {}'.format(all_eval_files[eid], idx, error.item()))

                veh_pred_data = veh_pred_data.cpu().numpy()

                veh_ids = ids.cpu().numpy()
                veh_ids = veh_ids[veh_ids < 100]

                final_frame = frames[-1]
                for i in range(len(veh_ids)):
                    eval_f.write(str(final_frame)+','+str(veh_ids[i])+','+str(veh_pred_data[-1, i, 0]+','+str(veh_pred_data[-1, i, 1]+'\n')))
        eval_f.close()


def main():
    parser = argparse.ArgumentParser()


if __name__ == '__main__':
    main()