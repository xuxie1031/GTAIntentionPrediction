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
                    loss += mse_loss(pred_outs[i], batch_pred_data[i])
                else:
                    loss += nll_loss(pred_outs[i], batch_pred_data[i])
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
        