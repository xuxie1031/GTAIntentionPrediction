import numpy as np

def w_mat(batch_prev_pos, batch_curr_pos, beta_=1.0, sigmaw=1.0):
    batch_v = batch_curr_pos-batch_prev_pos
    batch = len(batch_v)

    w = np.zeros((batch, batch), dtype=np.float64)
    for i in range(batch):
        for j in range(batch):
            if i == j: continue
            