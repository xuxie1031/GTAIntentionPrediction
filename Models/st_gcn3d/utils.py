import torch
import numpy as np

import collections


class WriteOnceDict(dict):
	def __setitem__(self, key, value):
		if not key in self:
			super(WriteOnceDict, self).__setitem__(key, value)


def data_vectorize(data_seq):
    first_value_dict = WriteOnceDict()
    vectorized_seq = data_seq.clone()

    num_nodes = data_seq.size(1)
    for i, frame in enumerate(data_seq):
        for node in range(num_nodes):
            first_value_dict[node] = frame[node, :]
            vectorized_seq[i, node, :] = frame[node, :]-first_value_dict[node]
    
    return vectorized_seq, first_value_dict


def data_revert(data_seq, first_value_dict):
    reverted_seq = data_seq.clone()

    num_nodes = data_seq.size(1)
    for i, frame in enumerate(data_seq):
        for node in range(num_nodes):
            reverted_seq[i, node, :] = frame[node, :]+first_value_dict[node]
    
    return reverted_seq


def output_activation(x):
    muX = x[:, :, 0:1]
    muY = x[:, :, 1:2]
    sigX = x[:, :, 2:3]
    sigY = x[:, :, 3:4]
    rho = x[:, :, 4:5]
    sigX = torch.exp(sigX)
    sigY = torch.exp(sigY)
    rho = torch.tanh(rho)

    return torch.cat((muX, muY, sigX, sigY, rho), dim=2)


def nll_loss(pred_out, pred_data):
    pred_len, batch = pred_data.size(0), pred_data.size(1)
    acc = torch.zeros_like(pred_data)
    muX = pred_out[:, :, 0]
    muY = pred_out[:, :, 1]
    sigX = pred_out[:, :, 2]
    sigY = pred_out[:, :, 3]
    rho = pred_out[:, :, 4]
    ohr = torch.pow(1-torch.pow(rho, 2), -0.5)

    x = pred_data[:, :, 0]
    y = pred_data[:, :, 1]
    out = torch.pow(ohr, 2)*(torch.pow(sigX, 2)*torch.pow(x-muX, 2) + torch.pow(sigY, 2)*torch.pow(y-muY, 2) - \
          2*rho*torch.pow(sigX, 1)*torch.pow(sigY, 1)*(x-muX)*(y-muY)) - torch.log(sigX*sigY*ohr)
    acc[:, :, 0] = out
    loss = torch.sum(acc)/(pred_len*batch)
    return loss


def mse_loss(pred_out, pred_data):
    pred_len, batch = pred_data.size(0), pred_data.size(1)
    acc = torch.zeros_like(pred_data)
    muX = pred_out[:, :, 0]
    muY = pred_out[:, :, 1]

    x = pred_data[:, :, 0]
    y = pred_data[:, :, 1]
    out = torch.pow(x-muX, 2)+torch.pow(y-muY, 2)
    acc[:, :, 0] = out
    loss = torch.sum(acc)/(pred_len*batch)
    return loss


def displacement_error(pred_traj, pred_traj_gt, mode='avg'):
    loss = pred_traj_gt.permute(1, 0, 2)-pred_traj.permute(1, 0, 2)
    loss = loss**2
    # loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    loss = torch.sqrt(loss.sum(dim=2)).mean(dim=1)

    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'avg':
        return torch.mean(loss)
    elif mode == 'raw':
        return loss


def final_displacement_error(pred_pos, pred_pos_gt, mode='avg'):
    loss = pred_pos_gt-pred_pos
    loss = loss**2
    loss = torch.sqrt(loss.sum(dim=1))

    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'avg':
        return torch.mean(loss)
    elif mode == 'raw':
        return loss