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
from utils.plot_latent import plot_latent_dvbf, display_video, plot_elbo
from utils.functions import bayes_fusion

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
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, output_dim)

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
    def __init__(self, dim_z, dim_y, dim_h, use_cuda=False, load_vae=False, vae_file_path=None, freeze_vae=False):
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
            checkpoint_paths = train_files[-1]
            checkpoints = sorted(os.listdir(os.path.join(vae_file_path, checkpoint_paths)))
            checkpt = torch.load(os.path.join(vae_file_path, checkpoint_paths, checkpoints[-1]))
            state_dict = checkpt['state_dict']
            self.vae.load_state_dict(state_dict, strict=False)
            print('VAE loaded successfully')
            if freeze_vae:
                for param in self.vae.parameters():
                    param.requires_grad = False

        # Recognition Model - Hidden States LSTM Network to identify hidden properties from changing state space z
        self.y_net = LSTMEncode(input_size=dim_z, hidden_size=128, output_size=dim_y * self.q_dist_y.nparams)

        # Transition Network, computes TRANSITION (from previous visual state z, inferred y (time-invariant properties) and action)
        self.transition_net = MLPTransition(input_dim=dim_z + dim_y + 1, output_dim=dim_z * self.vae.q_dist.nparams)

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
        # Convention
        # t = current time step
        # t_1 = next time step

        # First time step is handled differently
        # Obtain the measurement latent space
        zs_meas_0, z_meas_params_0 = self.vae.encode(x[:, 0])
        prior_params_z_t = self.vae.get_prior_params(batch_size)  # only for 1st time step
        zs_t = zs_meas_0
        z_params_0 = z_meas_params_0

        xs_hat_0, x_hat_params_0 = self.vae.decode(zs_t)

        prior_params_z_f.append(prior_params_z_t)
        z_params_f.append(z_params_0)
        zs_f.append(zs_t)
        xs_hat_f.append(xs_hat_0)
        x_hat_params_f.append(x_hat_params_0)
        x_f.append(x[:, 0])

        # Transfer the prior params z_t
        prior_params_z_t = z_params_0

        for t in range(0, T-1):     # Note the time index starts from 0 due to the correct control index
            u_t = u[:, t]
            x_t_1 = x[:, t+1]

            prior_params_y_t_1 = self.get_prior_params_y(batch_size)

            # Obtain the ys - time-invariant properties and hidden
            y_params_t_1, hidden_y_t_1 = self.y_net.forward(zs_t, hidden_y_t)
            y_params_t_1 = y_params_t_1.view(batch_size, self.dim_y, self.q_dist_y.nparams)

            # Sample the latent code y
            ys_t_1 = self.q_dist_y.sample(params=y_params_t_1)

            # Obtain the next step z's
            z_trans_params_t_1 = self.transition_net.forward(torch.cat([zs_t, u_t, ys_t_1], dim=-1))
            z_trans_params_t_1 = z_trans_params_t_1.view(batch_size, self.dim_z, self.vae.q_dist.nparams)

            if t > int(T * H):
                # inducing some future prediction aspect to improve dynamics learning
                z_params_t_1 = z_trans_params_t_1
            else:
                # Obtain measured latent space from the observations
                zs_meas_t_1, z_meas_params_t_1 = self.vae.encode(x_t_1)

                # Perform Bayesian Integration
                z_params_t_1 = bayes_fusion(z_trans_params_t_1, z_meas_params_t_1)

            # Pass through the decoder
            zs_t_1 = self.vae.q_dist.sample(params=z_params_t_1)
            xs_hat_t_1, x_hat_params_t_1 = self.vae.decode(zs_t_1)

            # Save the parameters
            prior_params_y_f.append(prior_params_y_t_1)
            y_params_f.append(y_params_t_1)
            ys_f.append(ys_t_1)

            prior_params_z_f.append(prior_params_z_t)
            z_params_f.append(z_params_t_1)
            zs_f.append(zs_t_1)

            xs_hat_f.append(xs_hat_t_1)
            x_hat_params_f.append(x_hat_params_t_1)
            x_f.append(x_t_1)

            # Transfer the variables from 1 time step to another
            prior_params_z_t = z_params_t_1
            hidden_y_t = hidden_y_t_1
            zs_t = zs_t_1

        return prior_params_y_f, y_params_f, ys_f, prior_params_z_f, z_params_f, zs_f, xs_hat_f, x_hat_params_f, x_f

    def loss(self, x, u, H):
        batch_size, T, _ = u.shape
        prior_params_y_f, y_params_f, ys_f, prior_params_z_f, z_params_f, zs_f, xs_hat_f, x_hat_params_f, x_f = self.filter(
            x, u, H)
        prior_params_y = torch.stack(prior_params_y_f, dim=1).view(batch_size * (T - 1), self.dim_y, 2)
        y_params = torch.stack(y_params_f, dim=1).view(batch_size * (T - 1), self.dim_y, 2)
        ys = torch.stack(ys_f, dim=1).view(batch_size * (T - 1), self.dim_y)
        prior_params_z = torch.stack(prior_params_z_f, dim=1).view(batch_size * T, self.dim_z, 2)
        z_params = torch.stack(z_params_f, dim=1).view(batch_size * T, self.dim_z, 2)
        zs = torch.stack(zs_f, dim=1).view(batch_size * T, self.dim_z)
        x_recon = torch.stack(x_hat_params_f, dim=1)
        x_recon_params = torch.stack(x_hat_params_f, dim=1).view(batch_size * T, 32, 32)
        x_t_1 = torch.stack(x_f, dim=1).view(batch_size * T, 32, 32)

        logpx = self.vae.x_dist.log_density(x_t_1, params=x_recon_params).view(batch_size, -1).sum(1)
        logpz = self.vae.prior_dist.log_density(zs, params=prior_params_z).view(batch_size, -1).sum(1)
        logqz_condx = self.vae.q_dist.log_density(zs, params=z_params).view(batch_size, -1).sum(1)

        logpy = self.prior_dist_y.log_density(ys, params=prior_params_y).view(batch_size, -1).sum(1)
        logqy_condx = self.q_dist_y.log_density(ys, params=y_params).view(batch_size, -1).sum(1)

        elbo = logpx + self.beta * (logpz - logqz_condx) + self.lamb * (logpy - logqy_condx)

        return x_recon, elbo, elbo.detach()


def anneal_kl(args, dvbf, iteration):
    # Annealing function for the beta and lamda terms
    # TODO: Improve the annealing function for the filter
    warmup_iter = 2000
    if args.lambda_anneal:
        dvbf.lamb = max(0, 0.95 - 1 / warmup_iter * iteration)  # 1 --> 0
    else:
        dvbf.lamb = 0
    if args.beta_anneal:
        dvbf.beta = min(args.beta, args.beta / warmup_iter * iteration)  # 0 --> 1
    else:
        dvbf.beta = args.beta


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-dist', default='normal', type=str, choices=['normal', 'laplace', 'flow'])
    parser.add_argument('-n', '--num-epochs', default=2000, type=int, help='number of training epochs')
    parser.add_argument('-b', '--batch-size', default=2000, type=int, help='batch size')
    parser.add_argument('--learning-rate', default=5e-5, type=float, help='learning rate')
    parser.add_argument('--dim_z', default=5, type=int, help='size of latent dimension z')
    parser.add_argument('--dim_y', default=4, type=int, help='size of latent dimension y')
    parser.add_argument('--dim_h', default=128, type=int, help='size of hidden layer dimension of LSTM')
    parser.add_argument('--beta', default=0.01, type=float, help='KL Scaling Factor of z')
    parser.add_argument('--lambda', default=1, type=float, help='KL Scaling Factor of y')
    parser.add_argument('--beta-anneal', default=True, type=bool, help='Use annealing of beta hyperparameter')
    parser.add_argument('--lambda-anneal', default=True, type=bool, help='Use annealing of lambda hyperparameter')
    parser.add_argument('--use_cuda', default=True, type=bool, help='Use cuda or not, set False for CPU testing')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--visdom', default=True, type=bool, help='Use Visdom for real time plotting, makes it slower')
    parser.add_argument('--save', default='results/pendulum/dbvf_train')
    parser.add_argument('--vae_path', default='results/pendulum/vae_train')
    parser.add_argument('--freeze_vae', default=False, type=bool,
                        help='Freeze the VAE during the filter training or not')
    parser.add_argument('--debug_dir', default='dump/pendulum/dbvf_train',
                        help='Freeze the VAE during the filter training or not')
    parser.add_argument('--log_freq', default=100, type=int, help='num iterations per log')

    args = parser.parse_args()

    if args.use_cuda:
        torch.cuda.set_device(args.gpu)

    train_loader = DataLoader(dataset=dset.Pendulum(), batch_size=args.batch_size, shuffle=True)  # for debug uncomment this version

    # Create folders for the saving the model
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    out_dir = os.path.join(args.save, date_time)
    debug_dir = os.path.join(args.debug_dir, date_time)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    dbvf = DVBF(dim_z=args.dim_z, dim_y=args.dim_y, dim_h=args.dim_h, load_vae=True, vae_file_path=args.vae_path,
                freeze_vae=False, use_cuda=args.use_cuda)

    '''
    TODO: Fix this in the event the training crashed
    load_dbvf = False
    if load_dbvf:
        checkpoint_path = 'results/pendulum/dbvf_train/train_24_01/checkpt-0000.pth'
        checkpt = torch.load(checkpoint_path)
        state_dict = checkpt['state_dict']
        dbvf.load_state_dict(state_dict, strict=False)
        print('DBVF loaded successfully')
    '''

    # setup the optimizer
    optimizer = optim.Adam(dbvf.parameters(), lr=args.learning_rate)

    # setup visdom for visualization
    if args.visdom:
        vis = visdom.Visdom(port=8097)

    train_elbo = []

    # training loop
    num_iterations = len(train_loader) * args.num_epochs
    iteration = 0
    elbo_running_mean = utils.RunningAverageMeter()
    while iteration < num_iterations:
        for i, values in enumerate(train_loader):
            # random masking at the end of the sequence to improve dynamics learning
            H = 0.5 + 0.4 * np.exp(-iteration / 1000)  # torch.distributions.uniform.Uniform(0.5, 0.9).sample()  #
            # H = torch.distributions.uniform.Uniform(0.7, 0.9).sample()
            obs, action, state, parameter = values
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
                    'args': args}, out_dir, iteration)

                plot_latent_dvbf(dbvf, dset.Pendulum(), vis)
                plot_elbo(train_elbo, vis)
                test_sample = reconstruced.sigmoid()
                video_path = debug_dir + '/dvbf_train_' + str(iteration)
                display_video(x[0, :, :].cpu(), test_sample[0, :, :].cpu(), video_path)

            iteration += 1


if __name__ == '__main__':
    model = main()
