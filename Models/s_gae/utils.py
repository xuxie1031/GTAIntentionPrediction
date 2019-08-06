import torch
import torch.nn.functional as F


def vae_loss(preds, targets, mu, logvar, n_nodes, norms, pos_weights):
    N = preds.size(0)
    costs = torch.zeros(N)
    for i in range(N):
        costs[i] = norms[i]*F.binary_cross_entropy(preds[i], targets[i], pos_weights=pos_weights[i])
    
    KLDs = -0.5 / n_nodes*torch.mean(torch.sum(1+2*logvar-mu.pow(2)-logvar.exp().pow(2), dim=2), dim=1)

    return torch.mean(costs+KLDs)