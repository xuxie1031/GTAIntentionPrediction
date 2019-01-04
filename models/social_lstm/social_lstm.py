import torch
import torch.nn as nn
import numpy as np

class SocialLSTM(nn.Module):
    def __init__(self, args):
        super(SocialLSTM, self).__init__()

        self.use_cuda = args.use_cuda
        self.hidden_size = args.hidden_size
        self.grid_size = args.grid_size
        self.embedding_size = args.embedding_size
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.gru = args.gru

        self.cell = nn.LSTMCell(2*self.embedding_size, self.hidden_size)
        if self.gru:
            self.cell = nn.GRUCell(2*self.embedding_size, self.hidden_size)

        self.input_embedding_layer = nn.Linear(self.input_size, self.embedding_size)
        self.tensor_embedding_layer = nn.Linear(self.grid_size*self.grid_size*self.hidden_size, self.embedding_size)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)


    def getSocialTensor(self, grid, hidden_states):
        num_nodes = grid.size()[0]

        social_tensor = torch.zeros(num_nodes, self.grid_size*self.grid_size, self.hidden_size)
        if self.use_cuda:
            social_tensor = social_tensor.cuda()

        for node in range(num_nodes):
            social_tensor[node] = torch.mm(torch.t(grid[node]), hidden_states)
        
        social_tensor = social_tensor.view(num_nodes, self.grid_size*self.grid_size*self.hidden_size)
        return social_tensor

    
    def forward(self, *args):
        input_data = args[0]
        grids = args[1]
        hidden_states = args[2]
        cell_states = args[3]
        seq_idxs = args[4]
        seq_len = args[5]
        num_nodes = args[6]

        outputs = torch.zeros(seq_len*num_nodes, self.output_size)
        if self.use_cuda:
            ouputs = ouputs.cuda()
        
        for framenum, frame in enumerate(input_data):
            if len(seq_idxs[framenum]) == 0:
                continue
            
            nodes_current = frame
            grid_current = grids[framenum]
            hidden_states_current = torch.index_select(hidden_states, 0, seq_idxs[framenum])
            if not self.gru:
                cell_states_current = torch.index_select(cell_states, 0, seq_idxs[framenum])
            
            social_tensor = self.getSocialTensor(grid_current, hidden_states_current)
            
            input_embedding = self.dropout(self.relu(self.input_embedding_layer(nodes_current)))
            social_embedding = self.dropout(self.relu(self.input_embedding_layer(social_tensor)))
            concat_embedding = torch.cat((input_embedding, social_embedding), 1)

            if not self.gru:
                h_nodes, c_nodes = self.cell(concat_embedding, (hidden_states_current, cell_states_current))
            else:
                h_nodes = self.cell(concat_embedding, (hidden_states_current))

            outputs[framenum*num_nodes+seq_idxs[framenum].data] = self.output_layer(h_nodes)
            hidden_states[seq_idxs[framenum].data] = h_nodes
            if not self.gru:
                cell_states[seq_idxs[framenum].data] = c_nodes
        
        outputs = outputs.view(seq_len, num_nodes, self.output_size)
        return outputs, hidden_states, cell_states
