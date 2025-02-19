import os
import time
import math
from numbers import Number
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import visdom
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import random
from datetime import datetime

import utils.dist as dist
import utils.compute_utils as utils
import utils.datasets as dset
from utils.flows import FactorialNormalizingFlow
from utils.plot_latent import plot_latent_vae, display_samples, plot_elbo

# Seeding for reproducibility
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)


class MLPEncoder(nn.Module):
    """
    MLP encoder for the VAE
    """
    def __init__(self, output_dim):
        super(MLPEncoder, self).__init__()
        self.output_dim = output_dim
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)

        # Setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(-1, 32 * 32)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        h = self.act(self.fc3(h))
        h = self.fc4(h)
        z = h.view(x.size(0), self.output_dim)
        return z


class MLPDecoder(nn.Module):
    """
    MLP decoder for the VAE
    """
    def __init__(self, input_dim):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1024)

        # Setup non-linearity
        self.act = nn.Tanh()

    def forward(self, z):
        h = z.view(z.size(0), -1)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        h = self.act(self.fc3(h))
        h = self.fc4(h)
        mu_img = h.view(z.size(0), 1, 32, 32)
        return mu_img


class VAE(nn.Module):
    def __init__(self, dim_z, use_cuda=False, prior_dist=dist.Normal(), q_dist=dist.Normal(), include_mutinfo=True, disentanglement=True):
        super(VAE, self).__init__()

        self.use_cuda = use_cuda
        self.dim_z = dim_z
        self.include_mutinfo = include_mutinfo
        self.disentanglement = disentanglement
        self.lamb = 0
        self.beta = 1
        self.x_dist = dist.Bernoulli() # Reconstructed Distribution

        # Model-specific
        # distribution family of p(z)
        self.prior_dist = prior_dist
        self.q_dist = q_dist
        # hyperparameters for prior p(z)
        self.register_buffer('prior_params', torch.zeros(self.dim_z, 2))

        self.encoder = MLPEncoder(dim_z * self.q_dist.nparams)
        self.decoder = MLPDecoder(dim_z)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()

    # return prior parameters wrapped in a suitable Variable
    def get_prior_params(self, batch_size=1):
        expanded_size = (batch_size,) + self.prior_params.size()
        prior_params = Variable(self.prior_params.expand(expanded_size))
        return prior_params

    # samples from the model p(x|z)p(z)
    def model_sample(self, batch_size=1):
        # sample from prior (value will be sampled by guide when computing the ELBO)
        prior_params = self.get_prior_params(batch_size)
        zs = self.prior_dist.sample(params=prior_params)
        # decode the latent code z
        x_params = self.decoder.forward(zs)
        return x_params

    # define the guide (i.e. variational distribution) q(z|x)
    def encode(self, x):
        x = x.view(x.size(0), 1, 32, 32)
        # use the encoder to get the parameters used to define q(z|x)
        z_params = self.encoder.forward(x).view(x.size(0), self.dim_z, self.q_dist.nparams)
        # sample the latent code z
        zs = self.q_dist.sample(params=z_params)
        return zs, z_params

    def decode(self, z):
        x_params = self.decoder.forward(z).view(z.size(0), 1, 32, 32)
        xs = self.x_dist.sample(params=x_params)
        return xs, x_params

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        zs, z_params = self.encode(x)
        xs, x_params = self.decode(zs)
        return xs, x_params, zs, z_params

    def elbo(self, x, dataset_size):
        # log p(x|z) + log p(z) - log q(z|x)
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 32, 32)
        prior_params = self.get_prior_params(batch_size)
        x_recon, x_params, zs, z_params = self.reconstruct_img(x)
        logpx = self.x_dist.log_density(x, params=x_params).view(batch_size, -1).sum(1)
        logpz = self.prior_dist.log_density(zs, params=prior_params).view(batch_size, -1).sum(1)
        logqz_condx = self.q_dist.log_density(zs, params=z_params).view(batch_size, -1).sum(1)

        # ---------------------------------------- No disentanglement --------------------------------------------------
        if not self.disentanglement:
            elbo = logpx + logpz - logqz_condx
            return elbo, elbo.detach()

        # ----------------------------------------- Disentanglement ----------------------------------------------------
        _logqz = self.q_dist.log_density(
            zs.view(batch_size, 1, self.dim_z),
            z_params.view(1, batch_size, self.dim_z, self.q_dist.nparams)
        )

        # compute log q(z) ~= log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m))
        _logqz = self.q_dist.log_density(
            zs.view(batch_size, 1, self.dim_z),
            z_params.view(1, batch_size, self.dim_z, self.q_dist.nparams)
        )
        # minibatch weighted sampling
        logqz_prodmarginals = (logsumexp(_logqz, dim=1, keepdim=False) - math.log(batch_size * dataset_size)).sum(1)
        logqz = (logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(batch_size * dataset_size))

        if self.include_mutinfo:
            # This is the complete disentanglement term - MI+TC+KL
            modified_elbo = logpx - \
                (logqz_condx - logqz) - \
                self.beta * (logqz - logqz_prodmarginals) - \
                (1 - self.lamb) * (logqz_prodmarginals - logpz)
        else:
            # This is partial disentanglement term - TC+KL
            modified_elbo = logpx - \
                self.beta * (logqz - logqz_prodmarginals) - \
                (1 - self.lamb) * (logqz_prodmarginals - logpz)

        return modified_elbo, modified_elbo.detach()


def log_importance_weight_matrix(batch_size, dataset_size):
    N = dataset_size
    M = batch_size - 1
    strat_weight = (N - M) / (N * M)
    W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
    W.view(-1)[::M+1] = 1 / N
    W.view(-1)[1::M+1] = strat_weight
    W[M-1, 0] = strat_weight
    return W.log()


def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


def anneal_kl(args, vae, iteration):
    # Annealing function for the beta and lamda terms

    warmup_iter = 2000
    if args.lambda_anneal:
        vae.lamb = max(0, 0.95 - 1 / warmup_iter * iteration)  # 1 --> 0
    else:
        vae.lamb = args.lamb
    if args.beta_anneal:
        vae.beta = min(args.beta, args.beta / warmup_iter * iteration)  # 0 --> 1
    else:
        vae.beta = args.beta


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-dist', default='normal', type=str, choices=['normal', 'laplace', 'flow'])
    parser.add_argument('-n', '--num-epochs', default=25, type=int, help='number of training epochs')
    parser.add_argument('-b', '--batch-size', default=100, type=int, help='batch size')
    parser.add_argument('-l', '--learning-rate', default=5e-4, type=float, help='learning rate')
    parser.add_argument('-z', '--latent-dim', default=3, type=int, help='size of latent dimension')
    parser.add_argument('--beta', default=2, type=float, help='Total Correlation Scaling Factor')
    parser.add_argument('--lamb', default=0, type=float, help='KL Scaling Factor')
    parser.add_argument('--disentanglement', default=True, type=bool, help='Use TC VAE (Disentanglement or Not)')
    parser.add_argument('--include-mutinfo', default=True, type=bool, help='Use Mutual Information term or not')
    parser.add_argument('--beta-anneal', default=False, type=bool, help='Use annealing of beta hyperparameter')
    parser.add_argument('--lambda-anneal', default=True, type=bool, help='Use annealing of lambda hyperparameter')
    parser.add_argument('--use_cuda', default=True, type=bool, help='Use cuda or not, set False for CPU testing')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--visdom', default=True, type=bool, help='Use Visdom for real time plotting, makes it slower')
    parser.add_argument('--save', default='results/pendulum/vae_train')
    parser.add_argument('--log_freq', default=200, type=int, help='num iterations per log')
    args = parser.parse_args()

    if args.use_cuda:
        torch.cuda.set_device(args.gpu)

    # data loader
    train_loader = DataLoader(dataset=dset.Pendulum(), batch_size=args.batch_size, shuffle=True) # for debug uncomment this version

    # Create folders for the saving the model
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    out_dir = os.path.join(args.save, date_time)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    vae = VAE(dim_z=args.latent_dim, use_cuda=args.use_cuda, include_mutinfo=args.include_mutinfo, disentanglement=args.disentanglement)

    # setup the optimizer
    optimizer = optim.Adam(vae.parameters(), lr=args.learning_rate)

    # setup visdom for visualization
    if args.visdom:
        vis = visdom.Visdom(port=8097)

    train_elbo = []

    # training loop
    dataset_size = len(train_loader.dataset)
    num_iterations = len(train_loader) * args.num_epochs
    iteration = 0
    # initialize loss accumulator
    elbo_running_mean = utils.RunningAverageMeter()
    while iteration < num_iterations:
        for i, values in enumerate(train_loader):
            obs, action, state, parameter = values
            batch_time = time.time()
            vae.train()
            anneal_kl(args, vae, iteration)
            optimizer.zero_grad()
            # transfer to GPU and merge the sequence dimension
            if args.use_cuda:
                x = obs.cuda()
            else:
                x = obs

            x = x.view(-1, 32, 32)
            # do ELBO gradient and accumulate loss
            obj, elbo = vae.elbo(x, dataset_size)
            if utils.isnan(obj).any():
                raise ValueError('NaN spotted in objective.')
            obj.mean().mul(-1).backward()
            elbo_running_mean.update(elbo.mean().item())
            optimizer.step()

            # report training diagnostics
            if iteration % args.log_freq == 0:
                train_elbo.append(elbo_running_mean.avg)
                print('[iteration %03d] time: %.2f \tbeta %.2f \tlambda %.2f training ELBO: %.4f (%.4f)' % (
                    iteration, time.time() - batch_time, vae.beta, vae.lamb,
                    elbo_running_mean.val, elbo_running_mean.avg))

                vae.eval()

                # Analyse results in the VISDOm
                if args.visdom:
                    display_samples(vae, x, vis)
                    plot_elbo(train_elbo, vis)
                    plot_latent_vae(vae, dset.Pendulum(), vis)


                # Save checkpoint
                utils.save_checkpoint({
                    'state_dict': vae.state_dict(),
                    'args': args}, out_dir, iteration)

            iteration += 1


if __name__ == '__main__':
    model = main()