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


def get_conv_mask(last_frame, frames, units, num_nodes, encoder_dim, neighbor_size, grid_size):
    width_unit, height_unit = units[0], units[1]
    last_frame_mask = np.zeros((num_nodes, width_unit, height_unit, encoder_dim))
    last_frame_np = last_frame.data.numpy()

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
            last_frame_mask[curr_idx, cell_x, cell_y, :] = 1.0
        
        other_grid2lookup = collections.OrderedDict(sorted(other_grid2lookup.items()))
        frames_nbr = torch.index_select(frames_permuted, 0, torch.tensor(list(other_grid2lookup.values())).long())
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