import torch
import numpy as np

import collections


def get_conv_mask(frame, units, num_nodes, encoder_dim, neighbor_size, grid_size):
    width_unit, height_unit = units[0], units[1]
    frame_mask = np.zeros((num_nodes, width_unit, height_unit, encoder_dim))
    frame_np = frame.data.numpy()

    width_bound, height_bound = neighbor_size/(width_unit*1.0)*2, neighbor_size/(height_unit*1.0)*2
    
    frame_nbrs = []
    for curr_idx in range(num_nodes):
        other_grid2lookup = {}
        for other_idx in range(num_nodes):
            current_x, current_y = frame_np[curr_idx, 0], frame_np[curr_idx, 1]

            width_low, width_high = current_x-width_bound/2, current_x+width_bound/2
            height_low, height_high = current_y-height_bound/2, current_y+height_bound/2

            other_x, other_y = frame_np[other_idx, 0], frame_np[other_idx, 1]
            if other_x >= width_high or other_x <= width_low or \
            other_y >= height_high or other_y <= height_low:
                continue
            
            cell_x = int(np.floor((other_x-width_low)/width_bound*grid_size))
            cell_y = int(np.floor((other_y-height_low)/height_bound*grid_size))

            if cell_x >= grid_size or cell_x < 0 or \
            cell_y >= grid_size or cell_y < 0:
                continue
            
            other_grid2lookup[cell_x*grid_size+cell_y] = other_idx
            frame_mask[curr_idx, cell_x, cell_y, :] = 1.0
        
        other_grid2lookup = collections.OrderedDict(sorted(other_grid2lookup.items()))
        frame_nbr = torch.index_select(frame, 0, torch.tensor(list(other_grid2lookup.values())).long())
        frame_nbrs.append(frame_nbr)
    
    frame_nbrs = torch.cat(frame_nbrs, dim=0)
    frame_mask = torch.from_numpy(frame_mask)

    return frame_nbrs, frame_mask
            

def output_activation(x):
    muX = x[:, :, 0:1]
    muY = x[:, :, 1:2]
    inv_sigX = x[:, :, 2:3]
    inv_sigY = x[:, :, 3:4]
    rho = x[:, :, 4:5]
    inv_sigX = torch.exp(inv_sigX)
    inv_sigY = torch.exp(inv_sigY)
    rho = torch.tanh(rho)

    return torch.cat((muX, muY, inv_sigX, inv_sigY, rho), dim=2)


