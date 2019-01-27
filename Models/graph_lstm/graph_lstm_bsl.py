import torch
import torch.nn as nn


class GraphLSTMBSL(nn.Module):
    def __init__(self, args):
        super(GraphLSTMBSL, self).__init__()

        self.use_cuda = args.use_cuda
        self.graph_hidden_size = args.graph_hidden_size

        self.dyn_embedding_size = args.dyn_embedding_size
        