import torch
import torch.nn as nn
from utils import *


class PoolHiddenNet(nn.Module):
    def __init__(
        self, input_dim, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
        activation='relu', batch_norm=True, dropout=0.0, use_cuda=True
    ):
        super(PoolHiddenNet, self).__init__()

        self.use_cuda = use_cuda
        self.input_dim = input_dim
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim

        mlp_pre_dim = embedding_dim+h_dim
        mlp_pre_pool_dims = [mlp_pre_dim, 512, bottleneck_dim]

        self.spatial_embedding = nn.Linear(input_dim, embedding_dim)
        self.mlp_pre_pool = make_mlp(
            mlp_pre_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )

    # 2D tensor repeat
    def repeat(self, tensor, num_reps):
        data_dim = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, data_dim)

        return tensor


    def forward(self, h_states, end_pos):
        num_nodes = end_pos.size(0)
        hidden_current = h_states.view(-1, self.h_dim)
        hidden_current_rep = hidden_current.repeat(num_nodes, 1)
        end_pos_rep1 = end_pos.repeat(num_nodes, 1)
        end_pos_rep2 = self.repeat(end_pos, num_nodes)
        rel_pos = end_pos_rep1-end_pos_rep2
        rel_embedding = self.spatial_embedding(rel_pos)
        mlp_h_input = torch.cat((rel_embedding, hidden_current_rep), dim=1)
        pool_current = self.mlp_pre_pool(mlp_h_input)
        pool_current = pool_current.view(num_nodes, num_nodes, -1).max(1)[0]

        return pool_current
