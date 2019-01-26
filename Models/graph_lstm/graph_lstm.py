import torch
import torch.nn as nn


class GraphLSTM(nn.Module):
    def __init__(self, args):
        self.use_cuda = args.use_cuda
        self.graph_veh_hidden_size = args.graph_veh_hidden_size
        self.graph_ped_hidden_size = args.graph_ped_hidden_size

        self.dyn_veh_embedding_size = args.dyn_veh_embedding_size
        self.dyn_ped_embedding_size = args.dyn_ped_embedding_size
        self.graph_veh_embedding_size = args.graph_veh_embedding_size
        self.graph_ped_embedding_size = args.graph_ped_embedding_size

        self.mat_veh_hidden_size = args.mat_veh_hidden_size
        self.mat_ped_hidden_size = args.mat_ped_hidden_size
        self.cell_embedding_size = self.dyn_veh_embedding_size+self.mat_veh_hidden_size+self.mat_ped_hidden_size
        self.cell_hidden_size = args.cell_hidden_size
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.gru = args.gru

        self.cell = nn.LSTMCell(self.cell_embedding_size, self.cell_hidden_size)
        if self.gru:
            self.cell = nn.GRUCell(self.cell_embedding_size, self.cell_hidden_size)

        self.graph_veh_cell = nn.LSTMCell(self.dyn_veh_embedding_size, self.graph_veh_hidden_size)
        if self.gru:
            self.graph_veh_cell = nn.GRUCell(self.dyn_veh_embedding_size, self.graph_veh_hidden_size)
        
        self.graph_ped_cell = nn.LSTMCell(self.dyn_ped_embedding_size, self.graph_ped_hidden_size)
        if self.gru:
            self.graph_ped_cell = nn.LSTMCell(self.dyn_ped_embedding_size, self.graph_ped_hidden_size)

        self.dyn_veh_embedding_layer = nn.Linear(self.input_size, self.dyn_veh_embedding_size)
        self.dyn_ped_embedding_layer = nn.Linear(self.input_size, self.dyn_ped_embedding_size)

        self.graph_veh_embedding_layer = nn.Linear(self.input_size, self.graph_veh_embedding_size)
        self.graph_ped_embedding_layer = nn.Linear(self.input_size, self.graph_ped_embedding_size)

        self.w_graph_veh = nn.Parameter(torch.randn(self.graph_veh_hidden_size, self.mat_veh_hidden_size))
        self.w_graph_ped = nn.Parameter(torch.randn(self.graph_ped_hidden_size, self.mat_ped_hidden_size))

        self.output_layer = nn.Linear(self.cell_hidden_size, self.output_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

        if self.use_cuda:
            self.to(torch.device('cuda:0'))
    

    def forward(self, veh_input_data, ped_input_data, cell_state_tuple, graph_veh_state_tuple, graph_ped_state_tuple, seq_len, num_nodes):
        outputs = torch.zeros(seq_len*num_nodes, self.output_size)
        if self.use_cuda:
            outputs = outputs.cuda()

        for i in range(seq_len):
            veh_frame = veh_input_data[i]
            ped_frame = ped_input_data[i]

            # self vehicle
            veh_dyn_embedding = self.dropout(self.relu(self.dyn_veh_embedding_layer(veh_frame)))
            # ped_dyn_embedding = self.dropout(self.relu(self.dyn_ped_embedding_layer(ped_frame)))

            veh_graph_embedding = self.dropout(self.relu(self.graph_veh_embedding_layer(veh_frame)))
            ped_graph_embedding = self.dropout(self.relu(self.graph_ped_embedding_layer(ped_frame)))

            # other vehicle
            veh_graph_h = graph_veh_state_tuple[0]
            if not self.gru:
                veh_graph_c = graph_veh_state_tuple[1]
            
            if not self.gru:
                curr_veh_graph_h, curr_veh_graph_c = self.graph_veh_cell(veh_graph_embedding, (veh_graph_h, veh_graph_c))
            else:
                curr_veh_graph_h = self.graph_veh_cell(veh_graph_embedding, veh_graph_h)

            if not self.gru:
                graph_veh_state_tuple = (curr_veh_graph_h, curr_veh_graph_c)
            else:
                graph_veh_state_tuple = (curr_veh_graph_h, )

            veh_other_graph_embedding = []
            veh_nodes = torch.tensor(range(num_nodes)).long()
            for j in range(num_nodes):
                veh_other_idx = torch.cat((veh_nodes[:j], veh_nodes[j+1:]))
                veh_other_graph_h = torch.index_select(curr_veh_graph_h, 0, veh_other_idx)
                veh_other_graph_embedding.append(veh_other_graph_h.sum(dim=0))
            veh_graph_embedding = torch.stack(veh_other_graph_embedding)
            veh_graph_embedding = torch.mm(veh_graph_embedding, self.w_graph_veh)

            # ped
            ped_graph_h = graph_ped_state_tuple[0]
            if not self.gru:
                ped_graph_c = graph_ped_state_tuple[1]

            if not self.gru:
                curr_ped_graph_h, curr_ped_graph_c = self.graph_ped_cell(ped_graph_embedding, (ped_graph_h, ped_graph_c))
            else:
                curr_ped_graph_h = self.graph_ped_cell(ped_graph_embedding, ped_graph_h)

            if not self.gru:
                graph_ped_state_tuple = (curr_ped_graph_h, curr_ped_graph_c)
            else:
                graph_ped_state_tuple = (curr_ped_graph_h, )

            ped_graph_embedding = curr_ped_graph_h.sum(dim=0, keepdim=True).repeat(num_nodes, 1)
            ped_graph_embedding = torch.mm(ped_graph_embedding, self.w_graph_ped)

            cell_embedding = torch.cat((veh_dyn_embedding, veh_graph_embedding, ped_graph_embedding), dim=1)

            cell_h = cell_state_tuple[0]
            if not self.gru:
                cell_c = cell_state_tuple[1]

            if not self.gru:
                curr_cell_h, curr_cell_c = self.cell(cell_embedding, (cell_h, cell_c))
            else:
                curr_cell_h = self.cell(cell_embedding, cell_h)
            
            if not self.gru:
                cell_state_tuple = (curr_cell_h, curr_cell_c)
            else:
                cell_state_tuple = (curr_cell_h, )
            
            outputs[i*num_nodes+torch.tensor(range(num_nodes)).long()] = self.output_layer(curr_cell_h)
        
        outputs = outputs.view(seq_len, num_nodes, self.output_size)
        return outputs, cell_state_tuple, graph_veh_state_tuple, graph_ped_state_tuple