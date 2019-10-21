import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.cluster import KMeans

import argparse
import time
from stgcn3d_gep import STGCN3DGEPModel
from graph import Graph
from utils import *

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..', '..', 'DataSet'))
sys.path.append(os.path.join(os.getcwd(), '..', 's_gae'))
sys.path.append(os.path.join(os.getcwd(), '..', '..', 'Saved'))
from trajectories import *
from loader import *
from gcn_vae import *


def exec_model(dataloader_train, dataloader_test, args):
    dev = torch.device('cpu')
    if args.use_cuda:
        dev = torch.device('cuda:'+str(args.gpu))

    stgcn_gep = STGCN3DGEPModel(args, device=dev, activation='relu')

    if not os.path.exists('models'): 
        os.makedirs('models')
        saved_state = {}
        
        state = torch.load(os.path.join('..', 's_gae', 'saved_models', 'SGAE.pth.tar'))
        saved_state['sgae'] = state['model']

        state = torch.load(os.path.join('..', 'cluster', 'saved_models', 'Cluster.pth.tar'))
        saved_state['cluster'] = state['model']

        torch.save(saved_state, os.path.join('models', args.saved_name))

    state = torch.load(os.path.join('models', args.saved_name))
    s_gae = state['sgae']
    cluster_obj = state['cluster']
    
    parser, duration_prior = gep_init(args)

    optimizer = optim.Adam(stgcn_gep.parameters(), lr=args.lr)

    print(len(dataloader_train))
    print(len(dataloader_test))

    err_epochs = []
    for epoch in range(args.num_epochs):
        stgcn_gep.train()

        print('****** Training beginning ******')
        loss_epoch = 0.0
        num_batch = 0

        for batch in dataloader_train:
            input_data_list, pred_data_list, _, num_node_list = batch

            t_start = time.time()
            batch_size = len(input_data_list)
            batch_data = data_batch(input_data_list, pred_data_list, num_node_list, args.obs_len, args.pred_len, args.num_feature, args.vmax)
            batch_data, _ = data_vectorize(batch_data)
            batch_input_data, batch_pred_data = batch_data[:, :-args.pred_len, :, :], batch_data[:, -args.pred_len:, :, :]

            inputs = data_feeder(batch_input_data)

            As_seq = []
            for i in range(args.obs_len-1):
                As = []
                for num in range(batch_size):
                    g = Graph(batch_data[num, i, :, :], args.vmax)
                    A = g.normalize_undigraph()
                    As.append(A)
                As = torch.stack(As)
                As_seq.append(As)
            As_seq = torch.stack(As_seq)
            As = As_seq[0]

            obs_sentence_prob = obs_parse(batch_data, args.obs_len-1, s_gae, As_seq, cluster_obj, args.nc, device=dev)
            pred_sentence = gep_pred_parse(obs_sentence_prob, args.pred_len, duration_prior, args)

            one_hots_pred_seq = data_feeder_onehots(pred_sentence, num_node_list, args.vmax)

            if args.use_cuda:
                inputs = inputs.to(dev)
                batch_pred_data = batch_pred_data.to(dev)
                As = As.to(dev)
                one_hots_pred_seq = one_hots_pred_seq.to(dev)

            pred_outs = stgcn_gep(inputs, As, one_hots_pred_seq)

            loss = 0.0
            for i in range(len(pred_outs)):
                if epoch < args.pretrain_epochs:
                    loss += mse_loss(pred_outs[i, :, :num_node_list[i], :], batch_pred_data[i, :, :num_node_list[i], :])
                else:
                    loss += nll_loss(pred_outs[i, :, :num_node_list[i], :], batch_pred_data[i, :, :num_node_list[i], :])
            loss_batch = loss.item() / batch_size
            loss /= batch_size

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(stgcn_gep.parameters(), args.grad_clip)
            optimizer.step()

            t_end = time.time()
            loss_epoch += loss_batch
            num_batch += 1

            print('epoch {}, batch {}, train_loss = {:.6f}, time/batch = {:.3f}'.format(epoch, num_batch, loss_batch, t_end-t_start))

        loss_epoch /= num_batch
        print('epoch {}, train_loss = {:.6f}\n'.format(epoch, loss_epoch))

        stgcn_gep.eval()
        print('****** Testing beginning ******')
        err_epoch = 0.0
        num_batch = 0

        for batch in dataloader_test:
            input_data_list, pred_data_list, _, num_node_list = batch

            t_start = time.time()
            batch_size = len(input_data_list)
            batch_data = data_batch(input_data_list, pred_data_list, num_node_list, args.obs_len, args.pred_len, args.num_feature, args.vmax)
            batch_input_data, batch_pred_data = batch_data[:, :-args.pred_len, :, :], batch_data[:, -args.pred_len:, :, :]

            batch_input_data, _ = data_vectorize(batch_input_data)
            inputs = data_feeder(batch_input_data)

            As_seq = []
            for i in range(args.obs_len-1):
                As = []
                for num in range(batch_size):
                    g = Graph(batch_input_data[num, i, :, :], args.vmax)
                    A = g.normalize_undigraph()
                    As.append(A)
                As = torch.stack(As)
                As_seq.append(As)
            As_seq = torch.stack(As_seq)
            As = As_seq[0]

            obs_sentence_prob = obs_parse(batch_input_data, args.obs_len-1, s_gae, As_seq, cluster_obj, args.nc, device=dev)
            pred_sentence = gep_pred_parse(obs_sentence_prob, args.pred_len, duration_prior, args)

            one_hots_pred_seq = data_feeder_onehots(pred_sentence, num_node_list, args.vmax)

            if args.use_cuda:
                inputs = inputs.to(dev)
                batch_pred_data = batch_pred_data.to(dev)
                As = As.to(dev)
                one_hots_pred_seq = one_hots_pred_seq.to(dev)

            pred_outs = stgcn_gep(inputs, As, one_hots_pred_seq)

            pred_rets = data_revert(pred_outs, first_value_dicts)
            pred_rets = pred_rets[:, :, :, :2]

            error = 0.0
            for i in range(len(pred_rets)):
                error += displacement_error(pred_rets[i, :, :num_node_list[i], :2], batch_pred_data[i, :, :, :2])
            err_batch = error.item() / batch_size

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
    parser.add_argument('--obs_len', type=int, default=15)
    parser.add_argument('--pred_len', type=int, default=25)
    parser.add_argument('--in_channels', type=int, default=4)
    parser.add_argument('--out_dim', type=int, default=5)
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--spatial_kernel_size', type=int, default=2)
    parser.add_argument('--temporal_kernel_size', type=int, default=3)
    parser.add_argument('--onehots_emb_dim', type=int, default=128)
    parser.add_argument('--cell_input_dim', type=int, default=128)
    parser.add_argument('--cell_h_dim', type=int, default=256)
    parser.add_argument('--e_h_dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--residual', action='store_true', default=True)
    parser.add_argument('--gru', action='store_true', default=True)
    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--grammar_root', type=str, default='../grammar')
    parser.add_argument('--grammar_file', type=str, default='tmp/NGSIM/grammar/NGSIM_xu.pcfg')
    parser.add_argument('--grammar_prior', type=str, default='tmp/NGSIM/NGSIM_xu_duration_prior.json')
    parser.add_argument('--grammar_epsilon', type=float, default=1e-10)
    parser.add_argument('--dset_name', type=str, default='NGSIMDataset')
    parser.add_argument('--dset_tag', type=str, default='')
    parser.add_argument('--dset_feature', type=int, default=4)
    parser.add_argument('--frame_skip', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--pretrain_epochs', type=int, default=0)
    parser.add_argument('--saved_name', type=str, default='NGSIM_GAEC3_GEP.pth.tar')

    args = parser.parse_args()

    _, train_loader = data_loader(args, os.path.join(os.getcwd(), '..', '..', 'DataSet', 'dataset', args.dset_name, args.dset_tag, 'train'))
    _, test_loader = data_loader(args, os.path.join(os.getcwd(), '..', '..', 'DataSet', 'dataset', args.dset_name, args.dset_tag, 'test'))

    exec_model(train_loader, test_loader, args)


if __name__ == '__main__':
    main()