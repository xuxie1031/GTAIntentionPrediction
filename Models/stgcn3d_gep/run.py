import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import time
from st_gcn3d_gep import STGCN3DGEPModel
from graph import Graph
from utils import *

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..', '..'))


def exec_model(dataloader_train, dataloader_test, args):
    net = STGCN3DGEPModel(args)