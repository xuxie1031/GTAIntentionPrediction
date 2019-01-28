import torch
import torch.nn as nn


class GraphLSTMBSL(nn.Module):
    def __init__(self, args):
        super(GraphLSTMBSL, self).__init__()

        self.use_cuda = args.use_cuda
        self.graph_hidden_size = args.graph_hidden_size
        self.dyn_embedding_size = args.dyn_embedding_size
        self.graph_embedding_size = args.graph_embedding_size

        self.mat_hidden_size = args.mat_hidden_size
        self.cell_embedding_size = self.dyn_embedding_size+self.mat_hidden_size
        self.cell_hidden_size = args.cell_hidden_size
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.gru = args.gru

        self.cell = nn.LSTMCell(self.cell_embedding_size, self.cell_hidden_size)
        if self.gru:
            self.cell = nn.GRUCell(self.cell_embedding_size, self.cell_hidden_size)

        self.graph_cell = nn.LSTMCell(self.graph_embedding_size, self.graph_hidden_size)
        if self.gru:
            self.graph_cell = nn.GRUCell(self.graph_embedding_size, self.graph_hidden_size)

        self.dyn_embedding_layer = nn.Linear(self.input_size, self.dyn_embedding_size)
        self.graph_embedding_layer = nn.Linear(self.input_size, self.graph_embedding_size)

        self.w_graph = nn.Parameter(torch.randn(self.graph_hidden_size, self.mat_hidden_size))

        self.output_layer = nn.Linear(self.cell_hidden_size, self.output_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

        if self.use_cuda:
            self.to(torch.device('cuda:0'))

    
    def forward(self, input_data, cell_state_tuple, graph_state_tuple, seq_len, num_nodes):
        outputs = torch.zeros(seq_len*num_nodes, self.output_size)
        if self.use_cuda:
            outputs = outputs.cuda()

        for i in range(seq_len):
            frame = input_data[i]

            dyn_embedding = self.dropout(self.relu(self.dyn_embedding_layer(frame)))
            graph_embedding = self.dropout(self.relu(self.graph_embedding_layer(frame)))

            graph_h = graph_state_tuple[0]
            if not self.gru:
                graph_c = graph_state_tuple[1]
            
            if not self.gru:
                curr_graph_h, curr_graph_c = self.graph_cell(graph_embedding, (graph_h, graph_c))
            else:
                curr_graph_h = self.graph_cell(graph_embedding, graph_h)
            
            if not self.gru:
                graph_state_tuple = (curr_graph_h, curr_graph_c)
            else:
                graph_state_tuple = (curr_graph_h, )

            other_graph_embedding = []
            nodes = torch.tensor(range(num_nodes)).long()
            for j in range(num_nodes):
                other_idx = torch.cat((nodes[:j], nodes[j+1:]))
                other_graph_h = torch.index_select(curr_graph_h, 0, other_idx)
                other_graph_embedding.append(other_graph_h.sum(dim=0))
            graph_embedding = torch.stack(other_graph_embedding)
            graph_embedding = torch.mm(graph_embedding, self.w_graph)

            cell_embedding = torch.cat((dyn_embedding, graph_embedding), dim=1)

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
            
            ouputs[i*num_nodes+torch.tensor(range(num_nodes)).long()]  =self.output_layer(curr_cell_h)

        ouputs = outputs.view(seq_len, num_nodes, self.output_size)
        return outputs, cell_state_tuple, graph_state_tuple