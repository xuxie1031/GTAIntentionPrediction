import torch
import numpy as np

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


def veh_ped_seperate(data_seq, ids, offset=100):
    veh_seq_ids = (ids < offset).nonzero().squeeze()
    ped_seq_ids = (ids >= offset).nonzero().squeeze()

    veh_seq = torch.index_select(data_seq, 1, veh_seq_ids)
    ped_seq = torch.index_select(data_seq, 1, ped_seq_ids)

    return veh_seq, ped_seq


def get_coef(outputs):
    mux, muy, sx, sy, corr = outputs[:, :, 0], outputs[:, :, 1], outputs[:, :, 2], outputs[:, :, 3], outputs[:, :, 4]

    sx = torch.exp(sx)
    sy = torch.exp(sy)
    corr = torch.tanh(corr)

    return mux, muy, sx, sy, corr


def gaussian_likelihood_2d(outputs, target):
    seq_len = outputs.size(0)
    mux, muy, sx, sy, corr = get_coef(outputs)

    normx = target[:, :, 0]-mux
    normy = target[:, :, 1]-muy
    sxsy = sx*sy

    z = (normx/sx)**2+(normy/sy)**2-2*(corr*normx*normy/sxsy)
    negrho = 1-corr**2

    result = torch.exp(-z/(2*negrho))
    denom = 2*np.pi*(sxsy*torch.sqrt(negrho))

    result = result/denom

    epsilon = 1e-20
    result = -torch.log(torch.clamp(result, min=epsilon))

    loss = torch.mean(result)
    return loss


def sample_gaussian_2d(mux, muy, sx, sy, corr):
    o_mux, o_muy, o_sx, o_sy, o_corr = mux[0, :], muy[0, :], sx[0, :], sy[0, :], corr[0, :]
    o_mux, o_muy, o_sx, o_sy, o_corr = o_mux.cpu().numpy(), o_muy.cpu().numpy(), o_sx.cpu().numpy(), o_sy.cpu().numpy(), o_corr.cpu().numpy()

    batch = mux.size(1)
    next_x = torch.zeros(batch)
    next_y = torch.zeros(batch)

    for node in range(batch):
        mean = [o_mux[node], o_muy[node]]
        cov = [[o_sx[node]*o_sx[node], o_corr[node]*o_sx[node]*o_sy[node]], [o_corr[node]*o_sx[node]*o_sy[node], o_sy[node]*o_sy[node]]]

        next_v = np.random.multivariate_normal(mean, cov, 1)
        next_x[node] = next_v[0, 0]
        next_y[node] = next_v[0, 1]
    
    return next_x, next_y


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