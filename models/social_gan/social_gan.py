import torch
import torch.nn as nn
from utils import *
from pool_net import *

class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1, dropout=0.0, use_cuda=True):
        super(Encoder, self).__init__()

        self.use_cuda = use_cuda
        self.input_dim = input_dim
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)

        self.spatial_embedding = nn.Linear(input_dim, embedding_dim)

    
    def init_hidden(self, batch, use_cuda=True):
        if use_cuda:
            return (
                torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
                torch.zeros(self.num_layers, batch, self.h_dim).cuda()
            )
        else:
            return (
                torch.zeros(self.num_layers, batch, self.h_dim),
                torch.zeros(self.num_layers, batch, self.h_dim)
            )

    
    def forward(self, obs_traj, obs_traj_idxs, num_nodes):
        (hidden_state, cell_state) = self.init_hidden(num_nodes, self.use_cuda)

        for framenum, frame in enumerate(obs_traj):
            if len(obs_traj_idxs[framenum]) == 0:
                continue

            batch = frame.size(1)
            h_curr = torch.index_select(hidden_state, 1, obs_traj_idxs[framenum])
            c_curr = torch.index_select(cell_state, 1, obs_traj_idxs[framenum])
            frame_embedding = self.spatial_embedding(frame.view(-1, self.input_dim))
            frame_embedding = frame_embedding.view(1, batch, self.embedding_dim)

            _, (h_curr, c_curr) = self.encoder(frame_embedding, (h_curr, c_curr))
            hidden_state[:, obs_traj_idxs[framenum].data, :] = h_curr
            cell_state[:, obs_traj_idxs[framenum].data, :] = c_curr
        
        final_h = hidden_state[:, obs_traj_idxs[-1].data, :]
        return final_h

    
class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim, embedding_dim=64, h_dim=128, mlp_dim=1024, num_layers=1,
                 dropout=0.0, bottleneck_dim=1024, activation='relu', batch_norm=True, use_cuda=True
    ):
        super(Decoder, self).__init__()

        self.use_cuda = use_cuda
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        
        self.decoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)

        self.pool_net = PoolHiddenNet(
            embedding_dim=embedding_dim,
            h_dim=h_dim,
            mlp_dim=mlp_dim,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )

        mlp_dims = [h_dim+bottleneck_dim, mlp_dim, h_dim]
        self.mlp = make_mlp(
            mlp_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )

        self.spatial_embedding = nn.Linear(input_dim, embedding_dim)
        self.hidden2pos = nn.Linear(h_dim, input_dim)


    def forward(self, last_pos, last_pos_rel, state_tuple):
        batch = last_pos.size(0)
        pred_traj_fake_rel = []
        decoder_input = self.spatial_embedding(last_pos_rel)
        decoder_input = decoder_input.view(1, batch, self.embedding_dim)

        for _ in range(self.seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            rel_pos = self.hidden2pos(output.view(-1, self.h_dim))
            curr_pos = last_pos+rel_pos

            decoder_h = state_tuple[0]
            pool_h = self.pool_net(decoder_h, curr_pos)
            decoder_h = torch.cat((decoder_h.view(-1, self.h_dim), pool_h), dim=1)
            decoder_h = self.mlp(decoder_h).unsqueeze(0)
            state_tuple = (decoder_h, state_tuple[1])

            decoder_input = self.spatial_embedding(rel_pos)
            decoder_input = decoder_input.view(1, batch, self.embedding_dim)
            pred_traj_fake_rel.append(rel_pos.view(batch, -1))
            last_pos = curr_pos
        
        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake_rel, state_tuple[0]


class TrajectoryGenerator(nn.Module):
    def __init__(self, obs_len, pred_len, input_dim, embedding_dim=64, encoder_h_dim=64,
                 decoder_h_dim=128, mlp_dim=1024, num_layers=1, noise_dim=(0, ),
                 dropout=0.0, bottleneck_dim=1024, activation='relu', batch_norm=True,
                 use_cuda=True
    ):
        super(TrajectoryGenerator, self).__init__()

        self.use_cuda = True
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.mlp_dim = mlp_dim
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.embedding_dim = embedding_dim
        self.noise_dim = noise_dim
        self.noise_first_dim = 0
        self.num_layers = num_layers
        self.bottleneck_dim = bottleneck_dim

        self.encoder = Encoder(
            input_dim,
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_cuda=use_cuda
        )

        self.decoder = Decoder(
            pred_len,
            input_dim,
            embedding_dim=embedding_dim,
            h_dim=decoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            batch_norm=batch_norm,
            use_cuda=use_cuda
        )

        self.pool_net = PoolHiddenNet(
            input_dim,
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            mlp_dim=mlp_dim,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout,
            use_cuda=use_cuda
        )

        if self.noise_dim[0] == 0:
            self.noise_dim = None
        else:
            self.noise_first_dim = self.noise_dim[0]

        pool_input_dim = encoder_h_dim+bottleneck_dim

        self.mlp_decoder_context = None
        if self.noise_dim or self.encoder_h_dim != self.decoder_h_dim:
            mlp_decoder_context_dims = [pool_input_dim, mlp_dim, self.decoder_h_dim-self.noise_first_dim]
            self.mlp_decoder_context = make_mlp(
                mlp_decoder_context_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )

        if self.use_cuda:
            self.to(torch.device('cuda:0'))


    def add_noise(self, input):
        if not self.noise_dim:
            return input
        
        noise_shape = (input.size(0), )+self.noise_dim
        z_decoder = get_gaussian_noise(noise_shape, self.use_cuda)

        decoder_h = torch.cat((input, z_decoder), dim=1)
        return decoder_h

    
    def forward(self, obs_traj, objs_traj_rel, obs_traj_idxs, num_nodes):
        batch = len(obs_traj_idxs[-1])
        if batch == 0:
            return None

        final_encoder_h = self.encoder(objs_traj_rel, obs_traj_idxs, num_nodes)

        end_pos = obs_traj[-1, :, :]
        pool_h = self.pool_net(final_encoder_h, end_pos)
        mlp_decoder_context_input = torch.cat((final_encoder_h.view(-1, self.encoder_h_dim), pool_h), dim=1)
        
        if self.mlp_decoder_context:
            noise_input = self.mlp_decoder_context(mlp_decoder_context_input)
        else:
            noise_input = mlp_decoder_context_input
        decoder_h = self.add_noise(noise_input).unsqueeze(0)
        
        if self.use_cuda:
            decoder_c = torch.zeros(self.num_layers, batch, self.decoder_h_dim).cuda()
        else:
            decoder_c = torch.zeros(self.num_layers, batch, self.decoder_h_dim)
        
        state_tuple = (decoder_h, decoder_c)
        last_pos = obs_traj[-1]
        last_pos_rel = objs_traj_rel[-1]

        decoder_out = self.decoder(last_pos, last_pos_rel, state_tuple)
        pred_traj_fake_rel, final_decoder_h = decoder_out

        if self.use_cuda:
            pred_traj_fake_rel_ret = torch.zeros(self.pred_len, num_nodes, self.input_dim).cuda()
        else:
            pred_traj_fake_rel_ret = torch.zeros(self.pred_len, num_nodes, self.input_dim)
        pred_traj_fake_rel_ret[:, obs_traj_idxs[-1].data, :] = pred_traj_fake_rel

        return pred_traj_fake_rel_ret


class TrajectoryDiscriminator(nn.Module):
    def __init__(self, obs_len, pred_len, input_dim, embedding_dim=64, h_dim=64, mlp_dim=1024,
                 num_layers=1, activation='relu', batch_norm=True, dropout=0.0, use_cuda=True,
                 d_type='local'
    ):
        super(TrajectoryDiscriminator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len+pred_len
        self.input_dim = input_dim
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.num_layers = num_layers
        self.d_type = d_type

        self.encoder = Encoder(
            input_dim, 
            embedding_dim=embedding_dim, 
            h_dim=h_dim, 
            mlp_dim=mlp_dim, 
            num_layers=num_layers,
            dropout=dropout
        )

        real_classifier_dims = [h_dim, mlp_dim, 1]
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )

        if d_type == 'global':
            mlp_pool_dims = [h_dim+embedding_dim, mlp_dim, h_dim]
            self.pool_net = PoolHiddenNet(
                input_dim,
                embedding_dim=embedding_dim,
                h_dim=h_dim,
                mlp_dim=mlp_pool_dims,
                bottleneck_dim=h_dim,
                activation=activation,
                batch_norm=batch_norm
            )

        if self.use_cuda:
            self.to(torch.device('cuda:0'))

    
    def forward(self, traj, traj_rel, traj_idx, num_nodes):
        final_h = self.encoder(traj_rel, traj_idx, num_nodes)

        if self.d_type == 'local':
            classifier_input = final_h.squeeze()
        else:
            classifier_input = self.pool_net(
                final_h.squeeze(), traj[0]
            )
        
        scores = self.real_classifier(classifier_input)
        if self.use_cuda:
            scores_ret = torch.zeros(scores.size(0), num_nodes).cuda()
        else:
            scores_ret = torch.zeros(scores.size(0), num_nodes)
        scores_ret[traj_idx[-1].data, :] = scores

        return scores_ret