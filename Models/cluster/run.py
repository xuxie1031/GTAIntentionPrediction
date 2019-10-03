from sklearn.cluster import KMeans

import torch
import numpy as np

import argparse
import os

def exec_model(args):
    feature_file = os.path.join(args.path_name, args.file_name)
    state = torch.load(feature_file)

    dim = state['dim']
    features = state['features']

    n_traj = len(features)
    features_all = np.concatenate(features, axis=0)
    features_all = features_all.reshape(-1, dim)

    kmeans = KMeans(n_clusters=args.k, random_state=0).fit(features_all)
    labels = kmeans.labels_
    labels = labels.reshape(n_traj, -1)

    # repeat removal
    for i in range(len(labels)):
        history = np.zeros(args.k)
        for j in range(len(labels[i])):
            if history[labels[i][j]] == 0: 
                history[labels[i][j]] = 1
            else:
                labels[i][j] = labels[i][j-1]

    for l_traj in labels:
        print(l_traj)

    np.savetxt(args.save_name, labels, fmt='%d', delimiter=',')

    print('****** saving model ******')
    if not os.path.exists('saved_models'): os.makedirs('saved_models')
    state = {}
    state['model'] = kmeans

    torch.save(state, os.path.join('saved_models', args.model_name))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--path_name', type=str, default='saved_features')
    parser.add_argument('--file_name', type=str, default='feature_NGSIM')
    parser.add_argument('--save_name', type=str, default='sentences_NGSIM')
    parser.add_argument('--model_name', type=str, default='Cluster.pth.tar')

    args = parser.parse_args()
    args.path_name = os.path.join('..', 's_gae', args.path_name)

    exec_model(args)


if __name__ == '__main__':
    main()
