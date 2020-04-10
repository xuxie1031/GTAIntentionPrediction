import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import time
from st_social import STSocialModel
# from graph_full import Graph
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

	net = STSocialModel(args.pred_len, args.in_channels, args.dyn_hidden_size, args.self_hidden_size, args.enc_hidden_size, args.dec_hidden_size, args.out_dim, args.gru, args.use_cuda, dev, dropout=args.dropout, residual=args.residual)
	optimizer = optim.Adam(net.parameters(), lr=args.lr)

	err_epochs = []
	for epoch in range(args.num_epochs):
		print('****** Training beginning ******')
		net.train()
		loss_epoch = 0
		num_batch = 0

		for batch in dataloader_train:
			input_data_list, pred_data_list, _, num_node_list = batch
                        
			batch_size = len(num_node_list)
			loss_batch = 0

			t_start = time.time()
			for i in range(batch_size):
				masks = data_masks(num_node_list[i], args.grid_size, args.enc_hidden_size)
				input_data, pred_data = input_data_list[i], pred_data_list[i]
                                
				data = torch.cat((input_data, pred_data), dim=0)
				data, _ = data_vectorize(data)
				input_data, pred_data = data[:-args.pred_len, :, :], data[-args.pred_len:, :, :]

				inputs = input_data[:, :, :2]
				ngbrs = data_ngbrs(inputs)

				if args.use_cuda:
					inputs = inputs.to(dev)
					ngbrs = ngbrs.to(dev)
					masks = masks.to(dev)
					pred_data = pred_data.to(dev)

				preds = net(inputs, ngbrs, masks)

				if epoch < args.pretrain_epochs:
					loss = mse_loss(preds, pred_data)
				else:
					loss = nll_loss(preds, pred_data)

				loss_batch += loss.item()

				optimizer.zero_grad()
				loss.backward()

				optimizer.step()

			t_end = time.time()
			loss_batch /= batch_size
			loss_epoch += loss_batch
			num_batch += 1

			print('epoch {}, batch {}, train_loss = {:.6f}, time/batch = {:.3f}'.format(epoch, num_batch, loss_batch, t_end-t_start))

		loss_epoch /= num_batch
		print('epoch {}, train_loss = {:.6f}\n'.format(epoch, loss_epoch))

		print('****** Testing beginning ******')
		net.eval()
		err_epoch = 0.0
		num_batch = 0

		for batch in dataloader_test:
			input_data_list, pred_data_list, _, num_node_list = batch

			batch_size = len(num_node_list)
			err_batch = 0

			t_start = time.time()
			for i in range(batch_size):
				masks = data_masks(num_node_list[i], args.grid_size, args.enc_hidden_size)
				input_data, pred_data = input_data_list[i], pred_data_list[i]

				input_data, first_values_dict = data_vectorize(input_data)

				inputs = input_data[:, :, :2]
				ngbrs = data_ngbrs(inputs)

				if args.use_cuda:
					inputs = inputs.to(dev)
					ngbrs = ngbrs.to(dev)
					masks = masks.to(dev)
					pred_data = pred_data.to(dev)

				preds = net(inputs, ngbrs, masks)
				ret_data = data_revert(preds[:, :, :2], first_values_dict)

				error = displacement_error(ret_data[:, :, :], pred_data[:, :, :2])[14]
                                err_batch += error.item()

			t_end = time.time()
			err_batch /= batch_size
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
	parser.add_argument('--obs_len', type=int, default=16)
	parser.add_argument('--pred_len', type=int, default=25)
	parser.add_argument('--in_channels', type=int, default=2)
	# parser.add_argument('--spatial_kernel_size', type=int, default=2)
	# parser.add_argument('--temporal_kernel_size', type=int, default=3)
	parser.add_argument('--dyn_hidden_size', type=int, default=32)
	parser.add_argument('--self_hidden_size', type=int, default=32)
	parser.add_argument('--enc_hidden_size', type=int, default=64)
	parser.add_argument('--dec_hidden_size', type=int, default=128)
	parser.add_argument('--out_dim', type=int, default=5)
	parser.add_argument('--grid_size', type=int, default=12)
	parser.add_argument('--lr', type=float, default=1e-3)
	# parser.add_argument('--grad_clip', type=float, default=10.0)
	parser.add_argument('--dropout', type=float, default=0.0)
	parser.add_argument('--residual', action='store_true', default=False)
	parser.add_argument('--gru', action='store_true', default=False)
	parser.add_argument('--use_cuda', action='store_true', default=True)
	parser.add_argument('--gpu', type=int, default=0)
	parser.add_argument('--dset_name', type=str, default='NGSIMDataset')
	parser.add_argument('--dset_tag', type=str, default="NGSIM")
	parser.add_argument('--dset_feature', type=int, default=4)
	parser.add_argument('--frame_skip', type=int, default=2)
	parser.add_argument('--num_epochs', type=int, default=100)
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
