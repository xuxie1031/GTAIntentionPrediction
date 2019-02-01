import numpy as np
import torch
import itertools


def get_grid_mask(frame, num_nodes, neighbor_size, grid_size, units=(1.0, 1.0)):
    width_unit, height_unit = units[0], units[1]
    frame_mask = np.zeros((num_nodes, num_nodes, grid_size**2))
    frame_np = frame.cpu().numpy()

    width_bound, height_bound = neighbor_size/(width_unit*1.0)*2, neighbor_size/(height_unit*1.0)*2

    list_indices = list(range(num_nodes))
    for current_frame_idx, other_frame_idx in itertools.permutations(list_indices, 2):
        current_x, current_y = frame_np[current_frame_idx, 0], frame_np[current_frame_idx, 1]

        width_low, width_high = current_x-width_bound/2, current_x+width_bound/2
        height_low, height_high = current_y-height_bound/2, current_y+height_bound/2

        other_x, other_y = frame_np[other_frame_idx, 0], frame_np[other_frame_idx, 1]
        if other_x >= width_high or other_x <= width_low or \
        other_y >= height_high or other_y <= height_low:
            continue
        
        cell_x = int(np.floor((other_x-width_low)/width_bound*grid_size))
        cell_y = int(np.floor((other_y-height_low)/height_bound*grid_size))

        if cell_x >= grid_size or cell_x < 0 or \
        cell_y >= grid_size or cell_y < 0:
            continue

        frame_mask[current_frame_idx, other_frame_idx, cell_x+cell_y*grid_size] = 1
    
    return frame_mask


def get_grid_mask_seq(frames, neighbor_size, grid_size, use_cuda, units=(1.0, 1.0)):
    mask_seq = []
    for i in range(len(frames)):
        mask = torch.from_numpy(get_grid_mask(frames[i], len(frames[i]), neighbor_size, grid_size, units)).float()
        if use_cuda:
            mask = mask.cuda()
        mask_seq.append(mask)
    
    return mask_seq
