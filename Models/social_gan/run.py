import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import time
from social_gan import *
from utils import *

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..', '..'))
from DataSet import *


def evaluate(args, batch, generator):
    input_data_list, pred_data_list, ids_list, num_nodes_list = batch

    err_batch = 0
    with torch.no_grad():
        for idx in range(len(input_data_list)):
            input_data = input_data_list[idx]
            pred_data = pred_data_list[idx]
            ids = ids_list[idx]
            num_nodes = num_nodes_list[idx]

            if args.use_cuda:
                input_data = input_data.cuda()
                pred_data = pred_data.cuda()
                ids = ids.cuda()
            traj_data = torch.cat((input_data, pred_data), dim=0)
            rel_traj_data = abs2rel(traj_data)

            rel_input_data = rel_traj_data[:args.obs_len, :, :]
            rel_pred_data = rel_traj_data[args.obs_len:, :, :]

            err_samples = 0
            for _ in range(args.num_samples):
                generator_out = generator(input_data, rel_input_data, num_nodes)

                rel_pred_fake = generator_out
                pred_fake = rel2abs(rel_pred_fake, input_data[-1])

                veh_pred_fake, _ = veh_ped_seperate(pred_fake, ids)
                veh_pred_data, _ = veh_ped_seperate(pred_data, ids)

                error = displacement_error(veh_pred_fake, veh_pred_data)
                # error = final_displacement_error(veh_pred_fake, veh_pred_data)
                err_samples += error.item()
            
            err_samples /= args.num_samples
            err_batch += err_samples
        
        err_batch /= args.batch_size
    
    return err_batch
        


def discriminator_step(args, batch, generator, discriminator, d_loss_fn, optimizer_d):
    input_data_list, pred_data_list, _, num_nodes_list = batch
    
    loss_batch_d = 0
    loss = torch.zeros(1)
    if args.use_cuda:
        loss = loss.cuda()

    for idx in range(len(input_data_list)):
        input_data = input_data_list[idx]
        pred_data = pred_data_list[idx]
        num_nodes = num_nodes_list[idx]

        if args.use_cuda:
            input_data = input_data.cuda()
            pred_data = pred_data.cuda()
        traj_data = torch.cat((input_data, pred_data), dim=0)
        rel_traj_data = abs2rel(traj_data)

        rel_input_data = rel_traj_data[:args.obs_len, :, :]
        rel_pred_data = rel_traj_data[args.obs_len:, :, :]

        generator_out = generator(input_data, rel_input_data, num_nodes)

        rel_pred_fake = generator_out
        pred_fake = rel2abs(rel_pred_fake, input_data[-1])

        traj_real = torch.cat((input_data, pred_data), dim=0)
        rel_traj_real = torch.cat((rel_input_data, rel_pred_data), dim=0)
        traj_fake = torch.cat((input_data, pred_fake), dim=0)
        rel_traj_fake = torch.cat((rel_input_data, rel_pred_fake), dim=0)

        scores_fake = discriminator(traj_fake, rel_traj_fake, num_nodes)
        scores_real = discriminator(traj_real, rel_traj_real, num_nodes)

        loss_data = d_loss_fn(scores_real, scores_fake)
        loss += loss_data

    loss_batch_d += loss.item()
    optimizer_d.zero_grad()
    loss.backward()
    if args.clip_th_d > 0:
        nn.utils.clip_grad_norm_(discriminator.parameters(), args.clip_th_d)
    optimizer_d.step()

    return loss_batch_d


def generator_step(args, batch, generator, discriminator, g_loss_fn, optimizer_g):
    input_data_list, pred_data_list, _, num_nodes_list = batch

    loss_batch_g = 0
    loss = torch.zeros(1)
    if args.use_cuda:
        loss = loss.cuda()

    for idx in range(len(input_data_list)):
        input_data = input_data_list[idx]
        pred_data = pred_data_list[idx]
        num_nodes = num_nodes_list[idx]

        if args.use_cuda:
            input_data = input_data.cuda()
            pred_data = pred_data.cuda()

        traj_data = torch.cat((input_data, pred_data), dim=0)
        rel_traj_data = abs2rel(traj_data)

        rel_input_data = rel_traj_data[:args.obs_len, :, :]
        rel_pred_data = rel_traj_data[args.obs_len:, :, :]

        l2_loss_rel_g = []
        for _ in range(args.best_k):
            generator_out = generator(input_data, rel_input_data, num_nodes)

            rel_pred_fake = generator_out
            pred_fake = rel2abs(rel_pred_fake, input_data[-1])

            if args.l2_loss_weight > 0:
                l2_loss_rel_g.append(args.l2_loss_weight*l2_loss(rel_pred_fake, rel_pred_data, mode='raw'))

        if args.l2_loss_weight > 0:
            l2_loss_rel_g = torch.stack(l2_loss_rel_g, dim=1)
            l2_loss_rel_g = torch.sum(l2_loss_rel_g, dim=0)
            l2_loss_rel_g = torch.min(l2_loss_rel_g)/num_nodes
            loss += l2_loss_rel_g
        
        traj_fake = torch.cat((input_data, pred_fake), dim=0)
        rel_traj_fake = torch.cat((rel_input_data, rel_pred_fake), dim=0)

        scores_fake = discriminator(traj_fake, rel_traj_fake, num_nodes)
        loss_data = g_loss_fn(scores_fake)
        loss += loss_data

    loss_batch_g += loss.item()
    optimizer_g.zero_grad()
    loss.backward()
    if args.clip_th_g > 0:
        nn.utils.clip_grad_norm_(generator.parameters(), args.clip_th_g)
    optimizer_g.step()

    return loss_batch_g


def exec_model(dataloader_train, dataloader_test, args):
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        input_dim=args.input_dim,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        batch_norm=args.batch_norm,
        use_cuda=args.use_cuda
    )
    generator.apply(init_weights)
    
    discriminator = TrajectoryDiscriminator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        input_dim=args.input_dim,
        embedding_dim=args.embedding_dim,
        h_dim=args.encoder_h_dim_d,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_norm=args.batch_norm,
        use_cuda=args.use_cuda
    )
    discriminator.apply(init_weights)

    g_loss_fn = gan_g_loss
    d_loss_fn = gan_d_loss

    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr_g)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr_d)

    for epoch in range(args.num_epochs):
        print('****** Training beginning ******')
        d_steps_left = args.steps_d
        g_steps_left = args.steps_g

        num_batch = 0
        for batch in dataloader_train:
            loss_batch_g = 0
            loss_batch_d = 0

            if d_steps_left > 0:
                loss_batch_d = discriminator_step(args, batch, generator, discriminator, d_loss_fn, optimizer_d)
                print('train epoch {}, batch {}, loss d = {:.6f}'.format(epoch, num_batch, loss_batch_d/dataloader_train.batch_size))
                d_steps_left -= 1
            elif g_steps_left > 0:
                loss_batch_g = generator_step(args, batch, generator, discriminator, g_loss_fn, optimizer_g)
                print('train epoch {}, batch {}, loss g = {:.6f}'.format(epoch, num_batch, loss_batch_g/dataloader_train.batch_size))
                g_steps_left -= 1
            num_batch += 1

            if d_steps_left > 0 or g_steps_left > 0:
                continue
            
            # check point
        
        print('****** Testing beginning ******')
        err_epoch = 0.0

        num_batch = 0
        for batch in dataloader_test:
            err_batch = evaluate(args, batch, generator)
            num_batch += 1

            print('test epoch {}, batch {}, error batch = {:.6f}'.format(epoch, num_batch, err_batch))
        
        err_epoch /= dataloader_test.batch_size

        d_steps_left = args.steps_d
        g_steps_left = args.steps_g


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--input_dim', type=int, default=2)
    
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_norm', type=bool_flag, default=1)
    parser.add_argument('--mlp_dim', type=int, default=1024)
    parser.add_argument('--bottleneck_dim', type=int, default=1024)

    parser.add_argument('--encoder_h_dim_g', type=int, default=64)
    parser.add_argument('--decoder_h_dim_g', type=int, default=128)
    parser.add_argument('--noise_dim', type=int_tuple, default=(0, ))
    parser.add_argument('--clip_th_g', type=float, default=0.0)
    parser.add_argument('--lr_g', type=float, default=5e-4)
    parser.add_argument('--steps_g', type=int, default=1)

    parser.add_argument('--encoder_h_dim_d', type=int, default=64)
    parser.add_argument('--clip_th_d', type=float, default=0.0)
    parser.add_argument('--lr_d', type=float, default=5e-4)
    parser.add_argument('--steps_d', type=int, default=2)

    parser.add_argument('--l2_loss_weight', type=float, default=0.0)
    parser.add_argument('--best_k', type=int, default=1)
    parser.add_argument('--num_samples', type=int, default=1)

    parser.add_argument('--use_cuda', type=int, default=1)

    args = parser.parse_args()

    _, train_loader = data_loader(args, os.path.join(os.getcwd(), '..', '..', 'DataSet', 'dataset', 'train'))
    _, test_loader = data_loader(args, os.path.join(os.getcwd(), '..', '..', 'DataSet', 'dataset', 'test'))

    exec_model(train_loader, test_loader, args)


if __name__ == '__main__':
    main()