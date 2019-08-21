import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import time
from .st_gcn3d_gep import STGCN3DGEPModel
from .graph import Graph
from .utils import *

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..', '..', 'DataSet'))
from trajectories import *
from loader import *


def exec_model(dataloader_train, dataloader_test, args):
    if args.use_cuda:
        dev = torch.device('cuda:'+str(args.gpu))

    stgcn_gep = STGCN3DGEPModel(args)

    state = torch.load(args.saved_name)
    s_gae = state['gae']
    cluster_obj = state['cluster']
    grammar_gep = state['grammar']

    phi_params = list(stgcn_gep.stgcn.paramerters())+list(stgcn_gep.cell.paramerters())
    predictor_params = list(stgcn_gep.predictor.paramerters())
    classifier_params = list(stgcn_gep.classifier.paramerters())
    optim_predictor = optim.Adam(phi_params+predictor_params, lr=args.lr)
    optim_classifier = optim.Adam(phi_params+classifier_params, lr=args.lr)

    err_epochs = []
    for epoch in range(args.num_epochs):
        stgcn_gep.train()

        print('****** Training beginning ******')
        loss_epoch_p, loss_epoch_c = 0.0, 0.0

        num_batch = 0
        for batch in dataloader_train:
            t_start = time.time()
            input_data_list, pred_data_list, _, num_node_list = batch

            loss_batch_p, loss_batch_c = 0.0, 0.0
            num2input_dict, num2pred_dict = data_batch(input_data_list, pred_data_list, num_node_list)
            for num in num2input_dict.keys():
                batch_size = len(num2input_dict[num])
                batch_input_data, batch_pred_data = torch.stack(num2input_dict[num]), torch.stack(num2pred_dict[num])

                batch_data = torch.cat((batch_input_data, batch_pred_data), dim=1)
                batch_data, _ = data_vectorize(batch_data)
                batch_input_data, batch_pred_data = batch_data[:, :-args.pred_len, :, :], batch_data[:, -args.pred_len:, :, :]

                inputs = data_feeder(batch_input_data)

                g = Graph(batch_input_data[:, 0, :, :])
                As = g.normalize_undigraph()

                As_seq = []
                for i in range(args.obs_len+args.pred_len-1):
                    g = Graph(batch_data[:, i, :, :])
                    As = g.normalize_undigraph()
                    As_seq.append(As)
                As_seq = torch.stack(As_seq)
                As = As_seq[0]

                if args.use_grammar:
                    obs_sentence_prob = gep_obs_parse(batch_data, args.obs_len+args.pred_len-1, s_gae, As_seq, cluster_obj, grammar_gep, args.nc, device=dev)
                    obs_sentence = convert_sentence(obs_sentence_prob, grammar_gep)
                one_hots_c_pred_seq = convert_one_hots(obs_sentence[-args.pred_len:, :], args.nc)

                if args.use_cuda:
                    inputs = inputs.to(dev)
                    As = As.to(dev)
                    one_hots_c_pred_seq = one_hots_c_pred_seq.to(dev)
                
                pred_outs, c_outs = stgcn_gep(inputs, As, one_hots_c_pred_seq, None, None)

                loss_c = 0.0
                gd_label_seq = obs_sentence[-args.pred_len:, :]
                for i in range(len(c_outs)):
                    loss_c += cross_entropy_loss(c_outs[i], gd_label_seq[i, :])
                loss_batch_c += loss_c.item()

                optim_classifier.zero_grad()
                loss_c.backward()

                torch.nn.utils.clip_grad_norm_(stgcn_gep, args.grad_clip)
                optim_classifier.step()

                loss_p = 0.0
                for i in range(len(pred_outs)):
                    if epoch < args.pretrain_epochs:
                        loss_p += mse_loss(pred_outs[i], batch_pred_data[i])
                    else:
                        loss_p += nll_loss(pred_outs[i], batch_pred_data[i])
                loss_batch_p += loss_p.item() / batch_size
                loss_p /= batch_size

                optim_predictor.zero_grad()
                loss_p.backward()

                torch.nn.utils.clip_grad_norm_(stgcn_gep, args.grad_clip)
                optim_predictor.step()

            t_end = time.time()
            loss_epoch_p += loss_batch_p
            loss_epoch_c += loss_batch_c
            num_batch += 1

            print('epoch {}, batch {}, train_loss_p = {:.6f}, train_loss_c = {:.6f}, time/batch = {:.3f}'.format(epoch, num_batch, loss_batch_p, loss_batch_c, t_end-t_start))
        
        loss_epoch_p /= num_batch
        loss_epoch_c /= num_batch
        print('epoch {}, train_loss_p = {:.6f}, train_loss_c = {:.6f}\n'.format(epoch, loss_epoch_p, loss_epoch_c))

        stgcn_gep.eval()
        print('****** Testing beginning ******')
        err_epoch = 0.0

        num_batch = 0
        for batch in dataloader_test:
            t_start = time.time()
            input_data_list, pred_data_list, _, num_node_list = batch

            err_batch = 0.0
            num2input_dict, num2pred_dict = data_batch(input_data_list, pred_data_list, num_node_list)
            for num in num2input_dict.keys():
                batch_size = len(num2input_dict[num])
                batch_input_data, batch_pred_data = torch.stack(num2input_dict[num]), torch.stack(num2pred_dict[num])

                batch_input_data, first_value_dicts = data_vectorize(batch_input_data)
                inputs = data_feeder(batch_input_data)

                As_seq = []
                for i in range(args.obs_len-1):
                    g = Graph(batch_input_data[:, i, :, :])
                    As = g.normalize_undigraph()
                    As_seq.append(As)
                As_seq = torch.stack(As_seq)
                As = As_seq[0]

                if args.use_grammar:
                    obs_sentence_prob = gep_obs_parse(batch_input_data, args.obs_len-1, s_gae, As_seq, cluster_obj, grammar_gep, args.nc, device=dev)

                if args.use_cuda:
                    inputs = inputs.to(dev)
                    As = As.to(dev)
                
                # need to modify inside
                pred_outs, _ = stgcn_gep(inputs, As, None, grammar_gep, obs_sentence_prob)

                pred_rets = data_revert(pred_outs, first_value_dicts)
                pred_rets = pred_rets[:, :, :, :2]

                error = 0.0
                for i in range(len(pred_rets)):
                    error += displacement_error(pred_rets[i], batch_pred_data[i])
                err_batch += error.item() / batch_size
            
            t_end = time.time()
            err_epoch += err_batch
            num_batch += 1

            print('epoch {}, batch {}, test_error = {:.6f}, time/batch = {:.3f}'.format(epoch, num_batch, err_batch, t_end-t_start))

        err_epoch /= num_batch
        err_epochs.append(err_epoch)
        print('epoch {}, test_err = {:.6f}\n'.format(epoch, err_epoch))
        print(err_epochs)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--obs_len', type=int, default=9)
    parser.add_argument('--pred_len', type=int, default=20)
    parser.add_argument('--in_channels', type=int, default=4)
    parser.add_argument('--out_dim', type=int, default=5)
    parser.add_argument('--nc', type=int, default=5)
    parser.add_argument('--spatial_kernel_size', type=int, default=2)
    parser.add_argument('--temporal_kernel_size', type=int, default=3)
    parser.add_argument('--cell_input_dim', type=int, default=256)
    parser.add_argument('--cell_h_dim', type=int, default=256)
    parser.add_argument('--e_h_dim', type=int, default=256)
    parser.add_argument('--e_c_dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=10.0)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--residual', action='store_true', default=True)
    parser.add_argument('--gru', action='store_true', default=False)
    parser.add_argument('--use_grammar', action='store_true', default=False)
    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dset_name', type=str, default='GTADataset')
    parser.add_argument('--dset_tag', type=str, default='GTAS')
    parser.add_argument('--dset_feature', type=int, default=4)
    parser.add_argument('--frame_skip', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--pretrain_epochs', type=int, default=5)
    parser.add_argument('--saved_name', type=str, default='GAEC5_GEP.pth.tar')

    args = parser.parse_args()

    _, train_loader = data_loader(args, os.path.join(os.getcwd(), '..', '..', 'DataSet', 'dataset', args.dset_name, args.dset_tag, 'train'))
    _, test_loader = data_loader(args, os.path.join(os.getcwd(), '..', '..', 'DataSet', 'dataset', args.dset_name, args.dset_tag, 'test'))

    exec_model(train_loader, test_loader, args)


if __name__ == '__main__':
    main()