import torch
import numpy as np

import argparse
import time

from utils import *

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..', '..'))
from DataSet import *


def sample(pred_data, batch_prev_pos, batch_curr_pos, betas, sigmaws, sigmads, args):
    
    min_err = np.inf
    best_beta, best_sigmaw, best_sigmad = 0.0, 0.0, 0.0
    for beta_ in betas:
        for sigmaw in sigmaws:
            for sigmad in sigmads:

                ret_seq = np.zeros((args.pred_len, args.batch_size, 2))
                for i in range(args.pred_len):
                    w = w_mat(batch_prev_pos, batch_curr_pos, beta_, sigmaw)

                    for node in range(args.batch_size):
                        self_pos = batch_curr_pos[node, :]
                        delta_poses = np.random.uniform(0.0, args.axis_len, (args.best_k, 2))

                        min_e = np.inf
                        arg_delta_pos = np.array([0.0, 0.0])
                        for k in range(args.best_k):
                            delta_pos = delta_poses[k, :]
                            e = energy(w, node, delta_pos, self_pos, batch_prev_pos, batch_curr_pos, args.sigmad)
                            if e < min_e:
                                min_e = e
                                arg_delta_pos = delta_pos
                        
                        ret_seq[i, node, :] = batch_curr_pos[node, :]+arg_delta_pos
                    
                    batch_prev_pos = batch_curr_pos
                    batch_curr_pos = ret_seq[i]

                batch_err = final_displacement_error(ret_seq, pred_data)
                if batch_err < min_err:
                    min_err = batch_err
                    best_beta, best_sigmaw, best_sigmad = beta_, sigmaw, sigmad
                print('current best params: beta_={}, sigmaw={}, sigmad={}, err={}'.format(best_beta, best_sigmaw, best_sigmad, min_err))


def exec_model(dataloader_train, dataloader_test, args):
    betas = np.linspace(1e-6, 5.0, 1000)
    sigmaws = np.linspace(1e-6, 50.0, 1000)
    sigmads = np.linspace(1e-6, 50.0, 1000)

    # for epoch in range(args.num_epochs):
    for batch in dataloader_test:
        t_start = time.time()
        input_data_list, pred_data_list, _, num_nodes_list = batch

        for idx in range(1):
            input_data = input_data_list[idx]
            pred_data = pred_data_list[idx]
            num_nodes = num_nodes_list[idx]
            if num_nodes <= 1: continue
            if args.obs_len <= 1: continue

            input_data = input_data.numpy()
            pred_data = pred_data.numpy()

            sample(pred_data, input_data[-2], input_data[-1], betas, sigmaws, sigmads, args)
        break


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--beta_', type=float, default=1.0)
    parser.add_argument('--sigmaw', type=float, default=1.0)
    parser.add_argument('--sigmad', type=float, default=1.0)
    parser.add_argument('--axis_len', type=float, default=3.0)
    parser.add_argument('--best_k', type=int, default=100)

    args = parser.parse_args()

    _, train_loader = data_loader(args, os.path.join(os.getcwd(), '..', '..', 'DataSet', 'dataset', 'train'))
    _, test_loader = data_loader(args, os.path.join(os.getcwd(), '..', '..', 'DataSet', 'dataset', 'test'))

    exec_model(train_loader, test_loader, args)


if __name__ == '__main__':
    main()