import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import time
from st_gcn2d import STGCN2DModel
from graph import Graph
from utils import *

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..', '..', 'DataSet'))
from trajectories import *
from loader import *


def exec_model(dataloader_train, dataloader_test, args):
    dev = torch.device('cpu')
    if args.use_cuda:
        dev = torch.device('cuda:'+str(args.gpu))

    net = STGCN2DModel(args.pred_len, args.in_channels, args.spatial_kernel_size, args.temporal_kernel_size, args.enc_hidden_size, args.dec_hidden_size, args.out_dim, args.gru, args.use_cuda, dev, dropout=args.dropout) # , residual=args.residual TONY Change
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    err_epochs = []
    for epoch in range(args.num_epochs):
        net.train()
        print('****** Training beginning ******')
        loss_epoch = 0

        num_batch = 0
        for batch in dataloader_train:
            input_data_list, pred_data_list, _, num_nodes_list = batch
            num2input_dict, num2pred_dict = data_batch(input_data_list, pred_data_list, num_nodes_list)

            for num in num2input_dict.keys():
                t_start = time.time()
                batch_size = len(num2input_dict[num])
                batch_input_data, batch_pred_data = torch.stack(num2input_dict[num]), torch.stack(num2pred_dict[num])

                batch_data = torch.cat((batch_input_data, batch_pred_data), dim=1)
                batch_data, _ = data_vectorize(batch_data)
                batch_input_data, batch_pred_data = batch_data[:, :-args.pred_len, :, :], batch_data[:, -args.pred_len:, :, :]

                # inputs = data_feeder(batch_input_data)
                inputs = batch_input_data[:, :, :, :2]

                g = Graph(batch_input_data[:, 0, :, :])
                As = g.normalize_undigraph()

                if args.use_cuda:
                    inputs = inputs.to(dev)
                    As = As.to(dev)
                    batch_pred_data = batch_pred_data.to(dev)
                
                preds = net(inputs, As)

                loss = 0.0
                for i in range(len(preds)):
                    if epoch < args.pretrain_epochs:
                        loss += mse_loss(preds[i], batch_pred_data[i])
                    else:
                        loss += nll_loss(preds[i], batch_pred_data[i])
                loss_batch = loss.item() / batch_size
                loss /= batch_size

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
                optimizer.step()

                t_end = time.time()
                loss_epoch += loss_batch
                num_batch += 1

                print('epoch {}, batch {}, train_loss = {:.6f}, time/batch = {:.3f}'.format(epoch, num_batch, loss_batch, t_end-t_start))

        loss_epoch /= num_batch
        print('epoch {}, train_loss = {:.6f}\n'.format(epoch, loss_epoch))

        net.eval()
        print('****** Testing beginning ******')
        err_epoch = 0.0

        num_batch = 0
        for batch in dataloader_test:
            input_data_list, pred_data_list, _, num_nodes_list = batch
            num2input_dict, num2pred_dict = data_batch(input_data_list, pred_data_list, num_nodes_list)
            for num in num2input_dict.keys():
                t_start = time.time()
                err_batch = 0.0
                batch_size = len(num2input_dict[num])
                batch_input_data, batch_pred_data = torch.stack(num2input_dict[num]), torch.stack(num2pred_dict[num])

                batch_input_data, first_values_dicts = data_vectorize(batch_input_data)
                # inputs = data_feeder(batch_input_data)
                inputs = batch_input_data[:, :, :, :2]

                g = Graph(batch_input_data[:, 0, :, :])
                As = g.normalize_undigraph()

                if args.use_cuda:
                    inputs = inputs.to(dev)
                    As = As.to(dev)
                    batch_pred_data = batch_pred_data.to(dev)
                
                preds = net(inputs, As)
                batch_ret_data = data_revert(preds[:, :, :, :2], first_values_dicts, dev)
                batch_ret_data = batch_ret_data[:, :, :, :2]

                error = 0.0
                for i in range(len(preds)):
                    error += displacement_error(batch_ret_data[i], batch_pred_data[i][:, :, :2])
                    # error += final_displacement_error(batch_ret_data[i][-1], batch_pred_data[i][-1][:, :2])
                err_batch = error.item() / batch_size

                t_end = time.time()
                err_epoch += err_batch
                num_batch += 1

                print('epoch {}, batch {}, test_error = {:.6f}, time/batch = {:.3f}'.format(epoch, num_batch, err_batch, t_end-t_start))

        err_epoch /= num_batch
        err_epochs.append(err_epoch)
        print('epoch {}, test_err = {:.6f}\n'.format(epoch, err_epoch))
        print(err_epochs)
    curr_path = os.getcwd()
    file_count = len([name for name in os.listdir(curr_path) if os.path.isfile(os.path.join(curr_path, name))])
    curr_name = os.path.join(curr_path, '2dADE' + str(file_count) + '.csv')
    np_err_epoch = np.array(err_epochs)
    np.savetxt(curr_name, np_err_epoch, delimiter=',')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--obs_len', type=int, default=15)
    parser.add_argument('--pred_len', type=int, default=25)
    parser.add_argument('--in_channels', type=int, default=2)
    parser.add_argument('--spatial_kernel_size', type=int, default=2)
    parser.add_argument('--temporal_kernel_size', type=int, default=3)
    parser.add_argument('--enc_hidden_size', type=int, default=16)
    parser.add_argument('--dec_hidden_size', type=int, default=256)
    parser.add_argument('--out_dim', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--grad_clip', type=float, default=10.0)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--residual', action='store_true', default=False)
    parser.add_argument('--gru', action='store_true', default=False)
    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--dset_name', type=str, default='NGSIMDataset')
    parser.add_argument('--dset_tag', type=str, default="NGSIM")
    parser.add_argument('--dset_feature', type=int, default=4)
    parser.add_argument('--frame_skip', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--pretrain_epochs', type=int, default=0)

    args = parser.parse_args()

    loader_name = args.dset_name+'_loader.pth.tar'
    if os.path.exists(loader_name):
        assert os.path.isfile(loader_name)
        state = torch.load(loader_name)

        train_loader = state['train']
        test_loader = state['test']
    else:
        _, train_loader = data_loader(args, os.path.join(os.getcwd(), '..', '..', 'DataSet', 'dataset', args.dset_name, args.dset_tag, 'train'))
        _, test_loader = data_loader(args, os.path.join(os.getcwd(), '..', '..', 'DataSet', 'dataset', args.dset_name, args.dset_tag, 'test'))

        state = {}
        state['train'] = train_loader
        state['test'] = test_loader
        torch.save(state, loader_name)

    print(len(train_loader))
    print(len(test_loader))

    exec_model(train_loader, test_loader, args)


if __name__ == '__main__':
    main()
