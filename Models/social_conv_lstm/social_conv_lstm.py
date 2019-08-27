import torch
import torch.nn as nn
from utils import *


class SocialConvLSTM(nn.Module):
    def __init__(self, obs_len, pred_len, input_dim, output_dim, encoder_dim=64, decoder_dim=128,
                 dyn_embedding_dim=32, input_embedding_dim=32, grid_size=(8,8), soc_conv1_depth=64, soc_conv2_depth=16, 
                 use_cuda=True, device=None
    ):
        super(SocialConvLSTM, self).__init__()

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

        if use_cuda:
            self.to(device)


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


    # nbrs_traj and masks are obtained from obs_traj
    def forward(self, obs_traj, nbrs_traj, masks):
        _, (obs_enc, _) = self.enc_lstm(self.leaky_relu(self.ip_emb(obs_traj)))
        obs_enc = self.leaky_relu(self.dyn_emb(obs_enc.view(obs_enc.size(1), obs_enc.size(2))))

        _, (nbrs_enc, _) = self.enc_lstm(self.leaky_relu(self.ip_emb(nbrs_traj)))
        nbrs_enc = nbrs_enc.view(nbrs_enc.size(1), nbrs_enc.size(2))

        soc_enc = torch.zeros_like(masks).float()
        soc_enc = soc_enc.masked_scatter_(masks, nbrs_enc)
        soc_enc = soc_enc.permute(0, 3, 2, 1)

        soc_enc = self.soc_maxpool(self.leaky_relu(self.soc_conv2(self.leaky_relu(self.soc_conv1(soc_enc)))))
        soc_enc = soc_enc.view(-1, self.soc_embedding_dim)

        enc = torch.cat((soc_enc, obs_enc), 1)

        enc = enc.repeat(self.pred_len, 1, 1)
        h_dec, _ = self.dec_lstm(enc)
        o_pred = self.op(h_dec)

        return output_activation(o_pred)