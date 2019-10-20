import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.cluster import KMeans

import collections

import os
import json

import sys
sys.path.append(os.path.join(os.getcwd(), '..', 'grammar', 'src'))
from grammarutils import *
from GEP import GeneralizedEarley as GEP

class WriteOnceDict(dict):
	def __setitem__(self, key, value):
		if not key in self:
			super(WriteOnceDict, self).__setitem__(key, value)


def data_feeder_onehots(batch_onehots, vmax, num_node_list):
    N, T, C = batch_onehots.size()
    data = torch.zeros(N, T, vmax, C)
    for num in range(N):
        for i in range(T):
            data[num, i, :num_node_list[num]] = batch_onehots[num, i]
    data = data.permute(0, 3, 1, 2).contiguous()

    return data