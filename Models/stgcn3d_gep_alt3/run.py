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

            num2input_dict, num2pred_dict = data_batch(input_data_list, pred_data_list, num_node_list)
            for num in num2input_dict.keys():
                t_start = time.time()
                batch_size = len(num2input_dict[num])
                batch_input_data, batch_pred_data = torch.stack(num2input_dict[num]), torch.stack(num2pred_dict[num])

                batch_data = torch.cat((batch_input_data, batch_pred_data), dim=1)
                batch_data, _ = data_vectorize(batch_data)
                batch_input_data, batch_pred_data = batch_data[:, :-args.pred_len, :, :], batch_data[:, -args.pred_len:, :, :]

                inputs = data_feeder(batch_input_data)

                As_seq = []
                for i in range(args.obs_len+args.pred_len-1):
                    g = Graph(batch_data[:, i, :, :])
                    As = g.normalize_undigraph()
                    As_seq.append(As)
                As_seq = torch.stack(As_seq)
                As = As_seq[0]

                obs_sentence_prob = obs_parse(batch_data, args.obs_len-1, s_gae, As_seq, cluster_obj, args.nc, device=dev)

                obs_sentence, _, _ = convert_sentence(obs_sentence_prob)
                one_hots_obs_seq = convert_one_hots(obs_sentence, args.nc)
                one_hots_obs = data_feeder_onehots_obs(one_hots_obs_seq, num)

                pred_sentence = gep_pred_parse(obs_sentence_prob, args.pred_len, duration_prior, parser, args)