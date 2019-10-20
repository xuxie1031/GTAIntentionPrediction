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

            num2input_dict, num2pred_dict = data_batch(input_data_list)
            for num in num2input_dict.keys():
                pass