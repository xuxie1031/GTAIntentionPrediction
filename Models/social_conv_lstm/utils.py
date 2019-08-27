import torch
import numpy as np

import collections


class WriteOnceDict(dict):
	def __setitem__(self, key, value):
		if not key in self:
			super(WriteOnceDict, self).__setitem__(key, value)


def data_vectorize(data_seq):
    first_value_dict = WriteOnceDict()
    vectorized_seq = []

    num_nodes = data_seq.size(1)
    frame0 = data_seq[0, :]
    for node in range(num_nodes):
        first_value_dict[node] = frame0[node, :]
    for i in range(1, len(data_seq)):
        frame = data_seq[i]
        vectorized_frame = torch.zeros(num_nodes, data_seq.size(-1))
        for node in range(num_nodes):
            vectorized_frame[node] = frame[node, :]-first_value_dict[node]
        vectorized_seq.append(vectorized_frame)
    
    return torch.stack(vectorized_seq)


def data_revert(data_seq, first_value_dict):
    reverted_seq = data_seq.clone()

    num_nodes = data_seq.size(1)
    for i, frame in enumerate(data_seq):
        for node in range(num_nodes):
            reverted_seq[i, node, :] = frame[node, :]+first_value_dict[node][:2]
    
    return reverted_seq


def veh_ped_seperate(data_seq, ids, offset=100):
    veh_seq_ids = (ids < offset).nonzero().squeeze()
    ped_seq_ids = (ids >= offset).nonzero().squeeze()

    veh_seq = torch.index_select(data_seq, 1, veh_seq_ids)
    ped_seq = torch.index_select(data_seq, 1, ped_seq_ids)

    return veh_seq, ped_seq


def get_conv_mask(last_frame, frames, num_nodes, encoder_dim, neighbor_size, grid_size, units=(1.0, 1.0)):
    width_unit, height_unit = units[0], units[1]
    last_frame_mask = np.zeros((num_nodes, grid_size, grid_size, encoder_dim), dtype=np.uint8)
    last_frame_np = last_frame.cpu().numpy()

    frames_permuted = frames.permute(1, 0, 2)

    width_bound, height_bound = neighbor_size/(width_unit*1.0)*2, neighbor_size/(height_unit*1.0)*2
    
    frames_nbrs = []
    for curr_idx in range(num_nodes):
        other_grid2lookup = {}
        for other_idx in range(num_nodes):
            current_x, current_y = last_frame_np[curr_idx, 0], last_frame_np[curr_idx, 1]

            width_low, width_high = current_x-width_bound/2, current_x+width_bound/2
            height_low, height_high = current_y-height_bound/2, current_y+height_bound/2

            other_x, other_y = last_frame_np[other_idx, 0], last_frame_np[other_idx, 1]
            if other_x >= width_high or other_x <= width_low or \
            other_y >= height_high or other_y <= height_low:
                continue
            
            cell_x = int(np.floor((other_x-width_low)/width_bound*grid_size))
            cell_y = int(np.floor((other_y-height_low)/height_bound*grid_size))

            if cell_x >= grid_size or cell_x < 0 or \
            cell_y >= grid_size or cell_y < 0:
                continue
            
            other_grid2lookup[cell_x*grid_size+cell_y] = other_idx
            last_frame_mask[curr_idx, cell_x, cell_y, :] = 1
        
        other_grid2lookup = collections.OrderedDict(sorted(other_grid2lookup.items()))
        other_grid2lookup_idx = torch.tensor(list(other_grid2lookup.values())).long()

        frames_nbr = torch.index_select(frames_permuted, 0, other_grid2lookup_idx)
        frames_nbrs.append(frames_nbr)
    
    frames_nbrs = torch.cat(frames_nbrs, dim=0).permute(1, 0, 2)
    last_frame_mask = torch.from_numpy(last_frame_mask)

    return frames_nbrs, last_frame_mask
            

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