import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import collections


class WriteOnceDict(dict):
    def __setitem__(self, key, value):
        if not key in self:
            super(WriteOnceDict, self).__setitem__(key, value)


def data_feeder(data_seq):
    T, V, C = data_seq.size()
    data = torch.zeros(T, V, V, 4)
    for i in range(V):
        for j in range(V):
            data[:, i, j, :2] = data_seq[:, i, :2]
            data[:, i, j, 2:] = data_seq[:, j, :2]
    data = data.permute(0, 3, 1, 2).contiguous()

    return data


def data_vectorize(data_seq):
    first_value_dict = WriteOnceDict()
    vecotorized_seq = []

    num_nodes = data_seq.size(1)
    frame0 = data_seq[0, :]
    for node in range(num_nodes):
        first_value_dict[node] = frame0[node]
    for i in range(1, len(data_seq)):
        frame = data_seq[i]
        vecotorized_frame = torch.zeros(num_nodes, data_seq.size(-1))
        for node in range(num_nodes):
            vecotorized_frame[node] = frame[node, :]-first_value_dict[node]
        vecotorized_seq.append(vecotorized_frame)
    
    return torch.stack(vecotorized_seq), first_value_dict
        

def vae_loss(preds, targets, mu, logvar, n_nodes, norms, pos_weights, device):
    N = preds.size(0)
    costs = torch.zeros(N).to(device)

    for i in range(N):
        # costs[i] = norms[i]*F.binary_cross_entropy_with_logits(preds[i], targets[i])
        costs[i] = F.binary_cross_entropy_with_logits(preds[i], targets[i])
    
    KLDs = -0.5 / n_nodes*torch.mean(torch.sum(1+2*logvar-mu.pow(2)-logvar.exp().pow(2), dim=2), dim=1)

    return torch.mean(costs+KLDs)
