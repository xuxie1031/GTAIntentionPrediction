import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import time
from .st_gcn3d import STGCN3DModel
from .graph import Graph
from .utils import *

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..', '..'))
from Dataset import *


def exec_model(dataloader_train, dataloader_test, args):
	net = STGCN3DModel(args.pred_len, args.in_channels, args.spatial_kernel, args.temporal_kernel, args.dec_hidden_size, args.out_dim, args.use_cuda, dropout=args.dropout)
	optimizer = optim.Adam(net.parameters(), lr=args.lr)

	err_epochs = []
	for epoch in range(args.num_epochs):
		net.train()
		print('****** Training beginning ******')
		loss_epoch = 0

		num_batch = 0
		for batch in dataloader_train:
			t_start = time.time()
			input_data_list, pred_data_list, _, num_node_list = batch

			loss_batch = 0.0
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

				if args.use_cuda:
					inputs = inputs.cuda()
					As = As.cuda()

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
			t_start = time.time()
			input_data_list, pred_data_list, _, num_node_list = batch

			err_batch = 0.0
			num2input_dict, num2pred_dict = data_batch(input_data_list, pred_data_list, num_node_list)
			for num in num2input_dict.keys():
				batch_size = len(num2input_dict[num])
				batch_input_data, batch_pred_data = torch.stack(num2input_dict[num]), torch.stack(num2pred_dict[num])

				batch_input_data, first_value_dicts = data_vectorize(batch_input_data)
				inputs = data_feeder(batch_input_data)

				g = Graph(batch_input_data[:, 0, :, :])
				As = g.normalize_undigraph()

				if args.use_cuda:
					inputs = inputs.cuda()
					As = As.cuda()

				pred_outs = net(inputs, As)

				pred_rets = data_revert(pred_outs, first_value_dicts)
				pred_rets = pred_rets[:, :, :, :2]

				error = 0.0
				for i in range(len(pred_rets)):
					error += displacement_error(pred_rets[i], batch_pred_data[i])
				err_batch += error.item() / batch_size

			t_end = time.time()
			err_epoch += err_batch
			num_batch += 1

			print('epoch {}, batch {}, test_error = {:.6f}, time/batch = {:.4f}'.format(epoch, num_batch, err_batch, t_end-t_start))

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