import numpy as np
from scipy.stats import multivariate_normal
import itertools


def kernel_mat(seq_len, time_scale, l=1.0, sigmaf=1.0):
    gamma_ = 0.5/l**2
    times = np.asarray(range(seq_len), dtype=np.float64)*time_scale
    kernel = np.zeros((seq_len, seq_len))

    for i in range(seq_len):
        for j in range(i, seq_len):
            kernel[i, j] = sigmaf**2*np.exp(-gamma_*(times[i]-times[j])**2)
            kernel[j, i] = kernel[i, j]
    
    return kernel


def kernel_cov(batch_seq, kernel, obs_len, pred_len, sigman=1.0):
    batch = len(batch_seq)
    batch_mean = np.zeros((batch, pred_len), dtype=np.float64)
    batch_cov = np.zeros((batch, pred_len, pred_len), dtype=np.float64)

    kernel_obs = kernel[:obs_len, :obs_len]
    kernel_pred = kernel[:obs_len, obs_len:]
    kernel_cov = np.linalg.inv(kernel_obs+sigman**2*np.eye(obs_len))

    for i in range(batch):
        batch_mean[i, :] = kernel_pred.T.dot(kernel_cov.dot(batch_seq[i]))
        batch_cov[i, :, :] = kernel[obs_len:, obs_len:]-kernel_pred.T.dot(kernel_cov.dot(kernel_pred))
    
    return batch_mean, batch_cov


def phi(batch_seq, alpha_=0.5, h=1.0):
    batch = len(batch_seq)
    list_indices = list(range(batch))

    prod = 1.0
    for curr_idx, other_idx in itertools.combinations(list_indices, 2):
        curr_traj, other_traj = batch_seq[curr_idx], batch_seq[other_idx]
        dist_traj = np.linalg.norm(curr_traj-other_traj, axis=1)
        phi_traj = 1-alpha_*np.exp(-0.5/h**2*dist_traj)
        prod = prod*np.prod(phi_traj)
    
    return prod


def gaussian_probs(batch_seq, batch_mean, batch_cov):
    batch = len(batch_seq)
    probs = np.zeros((batch, ))

    for i in range(batch):
        seq = batch_seq[i]
        mean = batch_mean[i]
        cov = batch_cov[i]

        var = multivariate_normal(mean=mean, cov=cov)
        probs[i] = var.pdf(seq)
    
    return probs


def gaussian_samples(batch_mean, batch_cov, best_k):
    batch = len(batch_mean)

    samples = []
    for i in range(batch):
        mean = batch_mean[i]
        cov = batch_cov[i]

        k_samples = np.random.multivariate_normal(mean, cov, best_k)
        samples.append(k_samples)
    
    samples = np.asarray(samples)
    samples = samples.transpose(1, 0, 2)

    return samples


def posterior(batch_seq, batch_meanx, batch_covx, batch_meany, batch_covy, alpha_=0.5, h=1.0):
    batch = len(batch_seq)
    batch_seqx = batch_seq[:, :, 0]
    batch_seqy = batch_seq[:, :, 1]

    gaussian_probx = np.prod(gaussian_probs(batch_seqx, batch_meanx, batch_covx))
    gaussian_proby = np.prod(gaussian_probs(batch_seqy, batch_meany, batch_covy))
    gaussian_prob = gaussian_probx*gaussian_proby

    # phi_ = phi(batch_seq, alpha_=alpha_, h=h)

    # return phi_*gaussian_prob
    return gaussian_prob


def displacement_error(pred_traj, pred_traj_gt, mode='sum'):
    loss = pred_traj_gt.transpose(1, 0, 2)-pred_traj.transpose(1, 0, 2)
    loss = loss**2
    loss = np.sqrt(loss.sum(axis=1))

    if mode == 'sum':
        return np.sum(loss)
    elif mode == 'raw':
        return loss


def final_displacement_error(pred_pos, pred_pos_gt, mode='sum'):
    loss = pred_pos_gt-pred_pos
    loss = loss**2
    loss = np.sqrt(loss.sum(axis=1))

    if mode == 'sum':
        return np.sum(loss)
    elif mode == 'raw':
        return loss