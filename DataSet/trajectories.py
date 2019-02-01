import torch
from torch.utils.data import Dataset

import numpy as np
import os


def seq_collate(data):
    (obs_seq_list, pred_seq_list, ids_list, num_nodes_list) = zip(*data)

    return (obs_seq_list, pred_seq_list, ids_list, num_nodes_list)


def read_file(path, delim=','):
    data  = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)

    return np.asarray(data)


class TrajectoryDataset(Dataset):
    def __init__(self, data_dir, obs_len=8, pred_len=12, min_agent=1, delim=','):
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len+pred_len
        self.min_agent = min_agent
        self.delim = delim

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, path) for path in all_files]
        num_nodes_seq_list = []
        ids_seq_list = []
        seq_list = []

        for path in all_files:
            data = read_file(path)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_seq = len(frames)-self.seq_len+1

            for idx in range(num_seq+1):
                curr_seq_data = np.concatenate(frame_data[idx:idx+self.seq_len], axis=0)
                agents_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq = np.zeros((len(agents_in_curr_seq), 2, self.seq_len))
                curr_ids = np.zeros((len(agents_in_curr_seq)))

                # collect sequence in dense fashion
                num_nodes = 0
                for agent_id in agents_in_curr_seq:
                    curr_agent_seq = curr_seq_data[agent_id == curr_seq_data[:, 1], :]
                    curr_agent_seq = np.around(curr_agent_seq, decimals=4)
                    agent_front = frames.index(curr_agent_seq[0, 0])-idx
                    agent_end = frames.index(curr_agent_seq[-1, 0])-idx+1
                    if agent_end-agent_front != self.seq_len:
                        continue
                    curr_agent_seq = np.transpose(curr_agent_seq[:, 2:4])
                    _idx = num_nodes
                    curr_seq[_idx, :, agent_front:agent_end] = curr_agent_seq
                    curr_ids[_idx] = agent_id
                    num_nodes += 1

                if num_nodes > self.min_agent:
                    num_nodes_seq_list.append(num_nodes)
                    ids_seq_list.append(curr_ids[:num_nodes])
                    seq_list.append(curr_seq[:num_nodes])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)

        self.num_nodes_seq_list = num_nodes_seq_list
        self.ids_seq_list = ids_seq_list
        self.obs_traj = torch.from_numpy(seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len:]).type(torch.float)
        cum_start_idx = [0]+np.cumsum(num_nodes_seq_list).tolist()
        self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx[:-1], cum_start_idx[1:])]


    def __len__(self):
        return self.num_seq
    

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        obs_traj = self.obs_traj[start:end, :]
        pred_traj = self.pred_traj[start:end, :]
        obs_traj = obs_traj.permute(2, 0, 1)
        pred_traj = pred_traj.permute(2, 0, 1)

        ids = self.ids_seq_list[index]
        ids = torch.from_numpy(ids).type(torch.float)

        num_nodes = self.num_nodes_seq_list[index]

        out = [
            obs_traj, pred_traj, ids, num_nodes
        ]

        return out