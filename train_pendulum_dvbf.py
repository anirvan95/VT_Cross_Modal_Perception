import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import visdom
from torch.utils.data import DataLoader
import numpy as np
import random
from datetime import datetime
import imageio

import utils.dist as dist
import utils.compute_utils as utils
import utils.datasets as dset
from utils.plot_latent import plot_latent_dvbf

# Seeding for reproducibility
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

from train_pendulum_vae import VAE


class LSTMEncode(nn.Module):
    """
    LSTM Encoder to infer the y params
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMEncode, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden=None):
        output, hidden = self.lstm(input[:, None, :], hidden)
        output = self.fc(output[:, -1, :])  # Take the last output
        return output, hidden


class MLPTransition(nn.Module):
    """
    MLP Transition function to model the non-linear dynamics
    """
    def __init__(self, input_dim, output_dim):
        super(MLPTransition, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, output_dim)

        # Setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.act(self.fc2(h))
        h = self.act(self.fc3(h))
        h = self.fc4(h)
        z = h.view(x.size(0), self.output_dim)
        return z


class DVBF(nn.Module):
    def __init__(self, dim_z, dim_y, dim_h, use_cuda=False, load_vae=True, vae_file_path=None, freeze_vae=False):
        super(DVBF, self).__init__()
        self.dim_z = dim_z
        self.dim_y = dim_y
        self.dim_h = dim_h
        self.beta = 1
        self.lamb = 1

        self.use_cuda = use_cuda

        # distribution family of p(y)
        self.prior_dist_y = dist.Normal()
        self.q_dist_y = dist.Normal()
        # hyperparameters for prior p(z)
        self.register_buffer('prior_params_y', torch.zeros(self.dim_y, 2))

        # Recognition Model  - Visual Encoder & Decoder
        self.vae = VAE(dim_z, use_cuda=use_cuda)

        if load_vae:
            # By default, loads the latest checkpoint
            train_files = sorted(os.listdir(vae_file_path))
            checkpoint_path = train_files[-1]
            checkpt = torch.load(checkpoint_path)
            state_dict = checkpt['state_dict']
            self.vae.load_state_dict(state_dict, strict=False)
            print('VAE loaded successfully')
            if freeze_vae:
                for param in self.vae.parameters():
                    param.requires_grad = False

        # Recognition Model - Hidden States LSTM Network to identify hidden properties from changing state space z
        self.y_net = LSTMEncode(input_size=dim_z, hidden_size=128, output_size=dim_y*self.q_dist_y.nparams)
        
        # Transition Network, computes TRANSITION (from previous visual state z, inferred y (time-invariant properties) and action)
        self.transition_net = MLPTransition(input_dim=dim_z+dim_y+1, output_dim=dim_z)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()

    def get_prior_params_y(self, batch_size=1):
        expanded_size = (batch_size,) + self.prior_params_y.size()
        prior_params_y = self.prior_params_y.expand(expanded_size)
        return prior_params_y

    # TODO: Add model sample function to obtain the latent space analysis

    def filter(self, x, u, H):
        batch_size, T, _ = u.shape

        # Set the hidden layer of the LSTM to zero
        hidden_y_t = (torch.zeros(1, batch_size, self.dim_h, device='cuda'),
                  torch.zeros(1, batch_size, self.dim_h, device='cuda'))

        # Define the variables to store the filtering output
        prior_params_y_f = []
        y_params_f = []
        ys_f = []
        prior_params_z_f = []
        z_params_f = []
        zs_f = []
        x_hat_params_f = []
        xs_hat_f = []
        x_f = []
        ang_vel_f = []
        for t in range(0, T-1):
            u_t = u[:, t]
            if t > int(T * H):
                x_t = xs_hat_t_1 # inducing some future prediction aspect to improve dynamics learning
            else:
                x_t = x[:, t]

            prior_params_y_t = self.get_prior_params_y(batch_size)
            prior_params_z_t = self.vae.get_prior_params(batch_size)

            # Obtain the visual encoding
            zs_t, z_params_t = self.vae.encode(x_t)
            # Obtain the ys - time-invariant properties and hidden
            y_params_t, hidden_y_t = self.y_net.forward(zs_t, hidden_y_t)
            y_params_t = y_params_t.view(batch_size, self.dim_y, self.q_dist_y.nparams)
            # sample the latent code y
            ys_t = self.q_dist_y.sample(params=y_params_t)

            # Obtain the next step z's
            zs_t_1 = self.transition_net.forward(torch.cat([zs_t, u_t, ys_t], dim=-1))

            # Compute the predicted distribution from the sample points TODO

            # Pass through the decoder
            xs_hat_t_1, x_hat_params_t_1 = self.vae.decode(zs_t_1)

            prior_params_y_f.append(prior_params_y_t)
            y_params_f.append(y_params_t)
            ys_f.append(ys_t)
            ang_vel_f.append(zs_t_1)
            prior_params_z_f.append(prior_params_z_t)
            z_params_f.append(z_params_t)
            zs_f.append(zs_t)

            xs_hat_f.append(xs_hat_t_1)
            x_hat_params_f.append(x_hat_params_t_1)
            x_f.append(x[:, t+1])

        return prior_params_y_f, y_params_f, ys_f, prior_params_z_f, z_params_f, zs_f, xs_hat_f, x_hat_params_f, x_f, ang_vel_f

    def loss(self, x, u, H):
        batch_size, T, _ = u.shape
        prior_params_y_f, y_params_f, ys_f, prior_params_z_f, z_params_f, zs_f, xs_hat_f, x_hat_params_f, x_f, ang_vel_f = self.filter(x, u, H)
        prior_params_y = torch.stack(prior_params_y_f, dim=1).view(batch_size*(T-1), self.dim_y, 2)
        y_params = torch.stack(y_params_f, dim=1).view(batch_size*(T-1), self.dim_y, 2)
        ys = torch.stack(ys_f, dim=1).view(batch_size*(T-1), self.dim_y)
        prior_params_z = torch.stack(prior_params_z_f, dim=1).view(batch_size*(T-1), self.dim_z, 2)
        z_params = torch.stack(z_params_f, dim=1).view(batch_size*(T-1), self.dim_z, 2) # TODO : generalize here
        zs = torch.stack(zs_f, dim=1).view(batch_size*(T-1), self.dim_z)
        x_recon = torch.stack(x_hat_params_f, dim=1)
        x_recon_params = torch.stack(x_hat_params_f, dim=1).view(batch_size*(T-1), 32, 32)
        x_t_1 = torch.stack(x_f, dim=1).view(batch_size*(T-1), 32, 32)

        logpx = self.vae.x_dist.log_density(x_t_1, params=x_recon_params).view(batch_size, -1).sum(1)
        logpz = self.vae.prior_dist.log_density(zs, params=prior_params_z).view(batch_size, -1).sum(1)
        logqz_condx = self.vae.q_dist.log_density(zs, params=z_params).view(batch_size, -1).sum(1)

        logpy = self.prior_dist_y.log_density(ys, params=prior_params_y).view(batch_size, -1).sum(1)
        logqy_condx = self.q_dist_y.log_density(ys, params=y_params).view(batch_size, -1).sum(1)

        elbo = logpx + (logpz - logqz_condx + logpy - logqy_condx)*0.35

        return x_recon.sigmoid(), elbo, elbo.detach()


def format(x):
    img = torch.clip(x * 255., 0, 255).to(torch.uint8)
    return img.view(-1, 32, 32).numpy()


def display_samples(x, x_recon, filename):
    T = 14
    frames = []
    for i in range(T):
        gt = format(x[i])
        pred = format(x_recon[i])
        img = np.concatenate([gt, pred], axis=1).squeeze()
        # cv2.imshow(mat=img, winname='generated')
        # cv2.waitKey(5)
        # plt.imshow(img)
        # plt.show()
        frames.append(img)

    with imageio.get_writer(f"{filename}.mp4", mode="I") as writer:
        for idx, frame in enumerate(frames):
            writer.append_data(frame)


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-dist', default='normal', type=str, choices=['normal', 'laplace', 'flow'])
    parser.add_argument('-n', '--num-epochs', default=1000, type=int, help='number of training epochs')
    parser.add_argument('-b', '--batch-size', default=4000, type=int, help='batch size')
    parser.add_argument('-l', '--learning-rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('-z', '--latent-dim', default=3, type=int, help='size of latent dimension')
    parser.add_argument('--beta', default=3, type=float, help='ELBO penalty term')
    parser.add_argument('--tcvae', default=True, type=bool, help='Use TC VAE')
    parser.add_argument('--exclude-mutinfo', default=False, type=bool, help='Use Mutual Information term or not')
    parser.add_argument('--beta-anneal', default=True, type=bool, help='Use annealing of beta hyperparameter')
    parser.add_argument('--lambda-anneal', default=True, type=bool, help='Use annealing of lambda hyperparameter')
    parser.add_argument('--use_cuda', default=True, type=bool, help='Use cuda or not, set False for CPU testing')
    parser.add_argument('--mss', default=True, type=bool, help='Use Minibatch Stratified Sampling')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--visdom', default=True, type=bool, help='Use Visdom for real time plotting, makes it slower')
    parser.add_argument('--save', default='results/pendulum/dbvf_train/train_02_05_I')   # Important configuration, else will overwrite TODO do it automatically
    parser.add_argument('--log_freq', default=500, type=int, help='num iterations per log')
    args = parser.parse_args()

    if args.use_cuda:
        torch.cuda.set_device(args.gpu)

    train_loader = DataLoader(dataset=dset.Pendulum(), batch_size=args.batch_size, shuffle=True) # for debug uncomment this version
    vis = visdom.Visdom(port=8097)
    dbvf = DVBF(dim_z=3, dim_beta=5, use_cuda=args.use_cuda)
    load_dbvf = False
    if load_dbvf:
        # TODO: improve checkpoint loading
        checkpoint_path = 'results/pendulum/dbvf_train/train_24_01/checkpt-0000.pth'
        checkpt = torch.load(checkpoint_path)
        state_dict = checkpt['state_dict']
        dbvf.load_state_dict(state_dict, strict=False)
        print('DBVF loaded successfully')

    # setup the optimizer
    optimizer = optim.Adam(dbvf.parameters(), lr=args.learning_rate)
    # optimizer = optim.Adadelta(dbvf.parameters(), lr=args.learning_rate)
    train_elbo = []

    # training loop
    num_iterations = len(train_loader) * args.num_epochs
    iteration = 0
    elbo_running_mean = utils.RunningAverageMeter()
    while iteration < num_iterations:
        for i, values in enumerate(train_loader):
            H = torch.distributions.uniform.Uniform(0.5, 0.9).sample() # random masking to improve dynamics learning
            obs, state, parameter, action = values
            batch_time = time.time()
            dbvf.train()
            optimizer.zero_grad()
            # transfer to GPU
            if args.use_cuda:
                x = obs.cuda()
                u = action.cuda()
            else:
                x = obs
                u = action

            # do ELBO gradient and accumulate loss
            reconstruced, obj, elbo = dbvf.loss(x, u, H)
            if utils.isnan(obj).any():
                raise ValueError('NaN spotted in objective.')

            obj.mean().mul(-1).backward()
            elbo_running_mean.update(elbo.mean().item())
            optimizer.step()

            # report training diagnostics
            if iteration % args.log_freq == 0:
                train_elbo.append(elbo_running_mean.avg)
                print('[iteration %03d] time: %.2f training ELBO: %.4f (%.4f)' % (
                    iteration, time.time() - batch_time,
                    elbo_running_mean.val, elbo_running_mean.avg))

                dbvf.eval()

                utils.save_checkpoint({
                    'state_dict': dbvf.state_dict(),
                    'args': args}, args.save, 0)

                plot_latent_vs_gt_pendulum(dbvf, dataset.Pendulum(), vis)
                display_samples(x[0, :, :].cpu(), reconstruced[0, :, :].cpu(), f'dump/train_III/dvbf-iter-{iteration}')

            iteration += 1


if __name__ == '__main__':
    model = main()
