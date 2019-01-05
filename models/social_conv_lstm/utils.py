import torch
import numpy as np


def output_activation(x):
    muX = x[:, :, 0:1]
    muY = x[:, :, 1:2]
    inv_sigX = x[:, :, 2:3]
    inv_sigY = x[:, :, 3:4]
    rho = x[:, :, 4:5]
    inv_sigX = torch.exp(inv_sigX)
    inv_sigY = torch.exp(inv_sigY)
    rho = torch.tanh(rho)

    return torch.cat((muX, muY, inv_sigX, inv_sigY, rho), dim=2)


