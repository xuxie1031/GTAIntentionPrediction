import numpy as np

def w_mat(batch_prev_pos, batch_curr_pos, beta_=1.0, sigmaw=1.0):
    batch_v = batch_curr_pos-batch_prev_pos
    batch = len(batch_v)

    w = np.zeros((batch, batch))
    for i in range(batch):
        for j in range(batch):
            if i == j: continue
            
            self_v = batch_v[i]
            self_pos = batch_curr_pos[i]
            other_pos = batch_curr_pos[j]
            delta_pos = self_pos-other_pos

            w[i, j] = np.exp(-(np.linalg.norm(delta_pos))**2/(2*sigmaw**2))* \
                      (0.5*(1-np.inner(delta_pos/np.linalg.norm(delta_pos), self_v/np.linalg.norm(self_v))))**beta_

    return w


def d_vec(delta_pos, self_pos, batch_prev_pos, batch_curr_pos):
    batch_v = batch_curr_pos-batch_prev_pos
    delta_pos = batch_curr_pos-self_pos
    delta_v = delta_pos-batch_v

    d = delta_pos-(np.sum(delta_pos*delta_v, axis=1)/(np.linalg.norm(delta_v, axis=1))**2).repeat(2).reshape(-1, 2) * delta_v
    d = np.linalg.norm(d, axis=1)

    return d


def energy(w, node, delta_pos, self_pos, batch_prev_pos, batch_curr_pos, sigmad=1.0):
    d = d_vec(delta_pos, self_pos, batch_prev_pos, batch_curr_pos)
    e = w[node, :]*np.exp(-d**2/(2*sigmad**2))
    return np.sum(e)


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