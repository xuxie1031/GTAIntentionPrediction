import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import time
from social_conv_lstm import SocialConvLSTM
from utils import *


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--output_dim', type=int, default=5)
    parser.add_argument('--input_embedding_dim', type=int, default=32)
    parser.add_argument('--dyn_embedding_dim', type=int, default=32)
    parser.add_argument('--encoder_dim', type=int, default=64)
    parser.add_argument('--decoder_dim', type=int, default=128)
    


if __name__ == '__main__':
    main()