import torch
import numpy as np

def parse_file_trajs(filename, args, delim=','):
    seq_len = args.obs_len+args.pred_len

    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    f.close()
    data = np.asarray(data)

    seq_list = []
    ids_seq_list = []
    frame_seq_list = []
    num_nodes_seq_list = []

    frames = np.unique(data[:, 0]).tolist()
    frame_data = []
    for frame in frames:
        frame_data.append(data[frame == data[:, 0], :])
    num_seq = len(frames)-seq_len+1

    for idx in range(num_seq+1):
        curr_seq_data = np.concatenate(frame_data[idx:idx+seq_len], axis=0)
        agents_in_curr_seq = np.unique(curr_seq_data[:, 1])
        curr_seq = np.zeros((len(agents_in_curr_seq), 2, seq_len))
        curr_ids = np.zeros((len(agents_in_curr_seq)))

        num_nodes = 0
        for agent_id in agents_in_curr_seq:
            curr_agent_seq = curr_seq_data[agent_id == curr_seq_data[:, 1], :]
            curr_agent_seq = np.around(curr_agent_seq, decimals=4)
            agent_front = frames.index(curr_agent_seq[0, 0])-idx
            agent_end = frames.index(curr_agent_seq[-1, 0])-idx+1
            if agent_end-agent_front != seq_len: continue
            
            curr_agent_seq = np.transpose(curr_agent_seq[:, 4:6])
            _idx = num_nodes
            curr_seq[_idx, :, agent_front:agent_end] = curr_agent_seq
            curr_ids[_idx] = agent_id
            num_nodes += 1
        
        if num_nodes > 1:
            num_nodes_seq_list.append(num_nodes)
            ids_seq_list.append(curr_ids[:num_nodes])
            seq_list.append(curr_seq[:num_nodes])
            frame_seq_list.append(np.unique(curr_seq_data[:, 0]))
    
    return seq_list, frame_seq_list, ids_seq_list, num_nodes_seq_list