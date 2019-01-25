import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import time
from social_conv_lstm import SocialConvLSTM
from utils import *


def exec_model(dataloader_train, dataloader_test, args):
    


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--output_dim', type=int, default=5)
    parser.add_argument('--input_embedding_dim', type=int, default=32)
    parser.add_argument('--dyn_embedding_dim', type=int, default=32)
    parser.add_argument('--encoder_dim', type=int, default=64)
    parser.add_argument('--decoder_dim', type=int, default=128)
    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--grad_clip', type=float, default=10.0)
    parser.add_argument('--lr', type=float, default=.003)
    parser.add_argument('--neighbor_size', type=int, default=32)
    parser.add_argument('--grid_size', type=int, default=8)
    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--pretrain_epochs', type=int, default=5)


if __name__ == '__main__':
    main()