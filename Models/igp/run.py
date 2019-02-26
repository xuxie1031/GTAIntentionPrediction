import torch
import numpy as np

import argparse
import time

from utils import *

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..', '..'))
from DataSet import *


def exec_model(dataloader_train, dataloader_test, args):
    # kernel = kernel_mat(args.obs_len+args.pred_len, args.time_scale, args.l, args.sigmaf)
    alphas = np.linspace(1e-6, 1.0, 10)
    hs = np.linspace(1.0, 50.0, 10)
    ls = np.linspace(1e-6, 1e-3, 10)
    sigmafs = np.linspace(1.0, 50.0, 10)
    sigmans = np.linspace(1.0, 50.0, 10)

    # for epoch in range(args.num_epochs):
    
    min_err = np.inf
    best_l, best_sigmaf, best_sigman = 0.0, 0.0, 0.0

    for l in ls:
        for sigmaf in sigmafs:
            kernel = kernel_mat(args.obs_len+args.pred_len, args.time_scale, l, sigmaf)
            for alpha_ in alphas:
                for h in hs:
                    for sigman in sigmans:
                        

                        # print('*** IGP beginning ***')
                        err_epoch = 0.0

                        num_batch = 0
                        for batch in dataloader_test:
                            t_start = time.time()
                            input_data_list, pred_data_list, _, num_nodes_list = batch

                            err_batch = 0.0
                            # for idx in range(args.batch_size):
                            for idx in range(1):
                                input_data = input_data_list[idx]
                                pred_data = pred_data_list[idx]
                                num_nodes = num_nodes_list[idx]
                                if num_nodes <= 1:  continue

                                input_data = input_data.numpy().transpose(1, 0, 2)
                                pred_data = pred_data.numpy()

                                inputx, inputy = input_data[:, :, 0], input_data[:, :, 1]

                                # batch_meanx, batch_covx = kernel_cov(inputx, kernel, args.obs_len, args.pred_len, args.sigman)
                                # batch_meany, batch_covy = kernel_cov(inputy, kernel, args.obs_len, args.pred_len, args.sigman)

                                batch_meanx, batch_covx = kernel_cov(inputx, kernel, args.obs_len, args.pred_len, sigman)
                                batch_meany, batch_covy = kernel_cov(inputy, kernel, args.obs_len, args.pred_len, sigman)

                                pred_samples = np.zeros((args.best_k, num_nodes, args.pred_len, 2))
                                pred_samples[:, :, :, 0] = gaussian_samples(batch_meanx, batch_covx, args.best_k)
                                pred_samples[:, :, :, 1] = gaussian_samples(batch_meany, batch_covy, args.best_k)

                                k_posteriors = np.zeros((args.best_k, ))
                                for i in range(args.best_k):
                                    batch_seq = pred_samples[i]
                                    # k_posteriors[i] = posterior(batch_seq, batch_meanx, batch_covx, batch_meany, batch_covy, args.alpha_, args.h)
                                    k_posteriors[i] = posterior(batch_seq, batch_meanx, batch_covx, batch_meany, batch_covy, alpha_, h)
                                argk = np.argmax(k_posteriors)

                                pred_sample = pred_samples[argk, :, :, :]
                                pred_sample = pred_sample.transpose(1, 0, 2)

                                # error = displacement_error(pred_sample, pred_data)
                                error = final_displacement_error(pred_sample[-1], pred_data[-1])
                                err_batch += error

                            # t_end = time.time()
                            # err_batch /= args.batch_size
                            # err_epoch += err_batch
                            # num_batch += 1

                            # print('epoch {}, batch {}, test_error = {:.6f}, time/batch = {:.3f}'.format(epoch, num_batch, err_batch, t_end-t_start))

                            if err_batch < min_err:
                                min_err = err_batch
                                best_l, best_sigmaf, best_alpha, best_h, best_sigman = l, sigmaf, alpha_, h, sigman
                            print('current best params: l={}, sigmaf={}, alpha={}, h={}, sigman={}, err={}'.format(best_l, best_sigmaf, best_alpha, best_h, best_sigman, min_err))
                            break


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--time_scale', type=float, default=0.01)
    parser.add_argument('--l', type=float, default=0.001)
    parser.add_argument('--sigmaf', type=float, default=1.0)
    parser.add_argument('--sigman', type=float, default=5.0)
    parser.add_argument('--alpha_', type=float, default=0.5)
    parser.add_argument('--h', type=float, default=10.0)
    parser.add_argument('--best_k', type=int, default=100)

    args = parser.parse_args()

    _, train_loader = data_loader(args, os.path.join(os.getcwd(), '..', '..', 'DataSet', 'dataset', 'train'))
    _, test_loader = data_loader(args, os.path.join(os.getcwd(), '..', '..', 'DataSet', 'dataset', 'test'))

    exec_model(train_loader, test_loader, args)


if __name__ == '__main__':
    main()