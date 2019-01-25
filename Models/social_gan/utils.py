import torch
import torch.nn as nn

import random


def int_tuple(s):
    return tuple(int(i) for i in s.split(','))


def bool_flag(s):
    if s == '1':
        return True
    elif s == '0':
        return False
    raise ValueError('Invalid Value for Bool Flag')


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)


def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    
    return nn.Sequential(*layers)


def abs2rel(seq):
    rel_seq = torch.zeros_like(seq)
    rel_seq[1:, :, :] = seq[1:, :, :]-seq[:-1, :, :]
    return rel_seq


def rel2abs(rel_seq, start_pt):
    rel_seq = rel_seq.permute(1, 0, 2)
    abs_seq = rel_seq.cumsum(dim=1)
    start_pt = start_pt.unsqueeze(1)
    abs_seq = abs_seq+start_pt
    return abs_seq.permute(1, 0, 2)


def get_gaussian_noise(shape, use_cuda=True):
    if use_cuda:
        return torch.randn(*shape).cuda()
    else:
        return torch.randn(*shape)
 

 def bce_loss(input, target):
     neg_abs = -input.abs()
     loss = input.clamp(min=0)-input*target+(1+neg_abs.exp()).log()
     return loss.mean()


def gan_g_loss(scores_fake):
    y_fake = torch.ones_like(scores_fake)*random.uniform(0.7, 1.2)
    return bce_loss(scores_fake, y_fake)


def gan_d_loss(scores_real, scores_fake):
    y_real = torch.ones_like(scores_real)*random.uniform(0.7, 1.2)
    y_fake = torch.ones_like(scores_fake)*random.uniform(0, 0.3)
    loss_real = bce_loss(scores_real, y_real)
    loss_fake = bce_loss(scores_fake, y_fake)
    return loss_real+loss_fake


def l2_loss(traj_pred, traj_pred_gt, mode='average'):
    seq_len, batch, _ = traj_pred.size()

    loss = (traj_pred_gt.permute(1, 0, 2)-traj_pred.permute(1, 0, 2))**2
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.sum(loss)/(seq_len*batch)
    elif mode == 'raw':
        return loss.sum(dim=2).sum(dim=1)