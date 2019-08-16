import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import time
from st_gcn3d_gep import STGCN3DGEPModel
from graph import Graph
from utils import *

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..', '..'))


def exec_model(dataloader_train, dataloader_test, args):
    if args.use_cuda:
        dev = torch.device('cuda:'+str(args.gpu))

    stgcn_gep = STGCN3DGEPModel(args)

    state = torch.load(args.saved_model)
    s_gae = state['gae']
    cluster_obj = state['cluster']
    grammar_gep = state['grammar']

    phi_params = list(net.stgcn.paramerters())+list(net.cell.paramerters())
    predictor_params = list(net.predictor.paramerters())
    classifier_params = list(net.classifier.paramerters())
    optim_predictor = optim.Adam(phi_params+predictor_params, lr=1e-4)
    optim_classifier = optim.Adam(phi_params+classifier_params, lr=1e-4)

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
            num2input_dict, num2pred_dict = data_batch(input_data_list, pred_data_list, num_list)
            for num in num2input_dict.keys():
                batch_size = len(num2input_dict[num])
                batch_input_data, batch_pred_data = torch.stack(num2input_dict[num]), torch.stack(num2pred_dict[num])

                batch_data = torch.cat((batch_input_data, batch_pred_data), dim=1)
                batch_data, _ = data_vectorize(batch_data)
                batch_input_data, batch_pred_data = batch_data[:, :-args.pred_len, :, :], batch_data[:, -args.pred_len:, :, :]

                inputs = data_feeder(batch_input_data)

                g = Graph(batch_input_data[:, 0, :, :])
                As = g.normalize_undigraph()

                gep_parsed_label_seq = gep_obs_parse(batch_data, args.obs_len+args.pred_len, s_gae, cluster_obj, grammar_gep)
                one_hots_c_pred_seq = convert_one_hots(gep_parsed_label_seq[:, -args.pred_len:], args.nc)

                if args.use_cuda:
                    inputs = inputs.to(dev)
                    As = As.to(dev)
                    one_hots_c_pred_seq = one_hots_c_pred_seq.to(dev)
                
                pred_outs, c_outs = stgcn_gep(inputs, As, one_hots_c_pred_seq, None, None)

                loss_c = 0.0
                gd_label_seq = gep_parsed_label_seq[:, -args.pred_len:]
                for i in range(len(c_outs)):
                    loss_c += cross_entropy_loss(c_outs[i], gd_label_seq[:, i])
                loss_batch_c = loss_c.item()

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
                loss_batch_p = loss_p.item() / batch_size
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

                g = Graph(batch_input_data)
                As = g.normalize_undigraph()

                gep_parsed_label_seq = gep_obs_parse(batch_input_data, args.obs_len, s_gae, cluster_obj, grammar_gep)

                if args.use_cuda:
                    inputs = inputs.to(dev)
                    As = As.to(dev)
                
                pred_outs, _ = stgcn_gep(inputs, As, None, grammar_gep, gep_parsed_label_seq)

                pred_rets = data_revert(pred_outs)
                pred_rets = pred_rets[:, :, :, :2]

                error = 0.0
                for i in range(len(pred_rets)):
                    error += displacement_error(pred_rets[i], batch_pred_data[i])
                error /= batch_size
            
            t_end = time.time()