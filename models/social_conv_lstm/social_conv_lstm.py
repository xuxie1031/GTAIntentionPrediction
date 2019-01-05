import torch
import torch.nn as nn
from utils import *


class SocialConvLSTM(nn.Module):
    def __init__(self, obs_len, pred_len, input_dim, output_dim, encoder_dim=64, decoder_dim=128,
                 grid_size=(8,8), soc_conv1_depth=64, soc_conv2_depth=16, 
                 dyn_embedding_dim=32, input_embedding_dim=32, use_cuda=True
    ):
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.grid_size = grid_size
        self.soc_conv1_depth = soc_conv1_depth
        self.soc_conv2_depth = soc_conv2_depth
        self.dyn_embedding_dim = dyn_embedding_dim
        self.input_embedding_dim = input_embedding_dim

        self.ip_emb = nn.Linear(input_dim, input_embedding_dim)

        self.enc_lstm = nn.LSTM(input_embedding_dim, encoder_dim, 1)
        
        self.dyn_emb = nn.Linear(encoder_dim, dyn_embedding_dim)

        self.soc_conv1 = nn.Conv2d(encoder_dim, soc_conv1_depth, 3)
        self.soc_conv2 = nn.Conv2d(soc_conv1_depth, soc_conv2_depth, 3)
        self.soc_maxpool = nn.MaxPool2d(2, stride=1)
        self.soc_embedding_dim = 3*3*soc_conv2_depth

        self.dec_lstm = nn.LSTM(self.soc_embedding_dim+self.dyn_embedding_dim, decoder_dim)

        self.op = nn.Linear(decoder_dim, output_dim)

        self.leaky_relu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()

        if self.use_cuda:
            self.to(torch.device('cuda:0'))


    def init_hidden(self, batch, use_cuda=True):
        if use_cuda:
            return (
                torch.zeros(1, batch, self.encoder_dim).cuda(),
                torch.zeros(1, batch, self.encoder_dim).cuda()
            )
        else:
            return (
                torch.zeros(1, batch, self.encoder_dim),
                torch.zeros(1, batch, self.encoder_dim)
            )

    def forward(self, obs_traj, obs_ngbrs_traj, masks, masks_idxs, obs_traj_idxs, obs_ngbrs_traj_idxs, num_nodes):
        # hidden_traj: (num_nodes, encode_dim)
        # hidden_ngbrs_traj: (num_nodes*num_nodes, encode_dim)
        # mask targets on the last frame

        batch = len(obs_traj_idxs[-1])
        if batch == 0:
            return None

        (hidden_state_traj, cell_state_traj) = self.init_hidden(num_nodes, self.use_cuda)
        for framenum, frame in enumerate(obs_traj):
            if len(obs_traj_idxs[framenum]) == 0:
                continue

            h_curr_traj = torch.index_select(hidden_state_traj, 1, obs_traj_idxs[framenum])
            c_curr_traj = torch.index_select(cell_state_traj, 1, obs_traj_idxs[framenum])
            
            embedding_traj = self.leaky_relu(self.ip_emb(frame.view(-1, self.input_dim))).unsqueeze(0)
            _, (h_curr_traj, c_curr_traj) = self.enc_lstm(embedding_traj, (h_curr_traj, c_curr_traj))

            hidden_state_traj[obs_traj_idxs[framenum].data] = h_curr_traj
            cell_state_traj[obs_traj_idxs[framenum].data] = c_curr_traj
        
        final_traj_h = torch.index_select(hidden_state_traj, 1, obs_traj_idxs[-1])
        traj_enc = self.leaky_relu(self.dyn_emb(final_traj_h.view(final_traj_h.size(1), final_traj_h.size(2))))

        (hidden_state_ngbrs_traj, cell_state_ngbrs_traj) = self.init_hidden(num_nodes*num_nodes, self.use_cuda)
        for framenum, frame in enumerate(obs_ngbrs_traj):
            if len(obs_ngbrs_traj_idxs[framenum]) == 0:
                continue

            h_curr_ngbrs_traj = torch.index_select(hidden_state_ngbrs_traj, 1, obs_ngbrs_traj_idxs[framenum])
            c_curr_ngbrs_traj = torch.index_select(cell_state_ngbrs_traj, 1, obs_ngbrs_traj_idxs[framenum])

            embedding_ngbrs_traj = self.leaky_relu(self.ip_emb(frame.view(-1, self.input_dim))).unsqueeze(0)
            _, (h_curr_ngbrs_traj, c_curr_ngbrs_traj) = self.enc_lstm(embedding_ngbrs_traj, (h_curr_ngbrs_traj, c_curr_ngbrs_traj))

            hidden_state_ngbrs_traj[obs_ngbrs_traj_idxs[framenum].data] = h_curr_ngbrs_traj
            cell_state_ngbrs_traj[obs_ngbrs_traj_idxs[framenum].data] = c_curr_ngbrs_traj
        
        final_ngbrs_traj_h = torch.index_select(hidden_state_ngbrs_traj, 1, masks_idxs)
        final_ngbrs_traj_h = final_ngbrs_traj_h.view(final_ngbrs_traj_h.size(1), final_ngbrs_traj_h.size(2))

        soc_enc = torch.zeros_like(masks).float()
        soc_enc = soc_enc.masked_scatter_(masks, final_ngbrs_traj_h)
        soc_enc = soc_enc.permute(0, 3, 2, 1)

        soc_enc = self.soc_maxpool(self.leaky_relu(self.soc_conv2(self.leaky_relu(self.soc_conv1(soc_enc)))))
        soc_enc = soc_enc.view(-1, self.soc_embedding_dim)

        enc = torch.cat((traj_enc, soc_enc), dim=1)
        enc = enc.repeat(self.pred_len, 1, 1)
        o_dec = self.dec_lstm(enc)
        o_pred = self.op(o_dec)

        return output_activation(o_pred)