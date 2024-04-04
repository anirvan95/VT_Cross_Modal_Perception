import os
import time
import math
from numbers import Number
import argparse
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.optim as optim
import visdom
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
np.random.seed(0)
import random
random.seed(0)
from tqdm import tqdm
import tcvae_utils.dist as dist
import tcvae_utils.utils as utils
import tcvae_utils.datasets as dset
from tcvae_utils.flows import FactorialNormalizingFlow
from tcvae_utils.compute_metric import elbo_decomposition_pendulum, mutual_info_metric_pendulum    # Compute after training
from tcvae_utils.plot_latent import plot_latent_vs_gt_pendulum, display_samples, plot_elbo


class MLPEncoder(nn.Module):
    def __init__(self, output_dim):
        super(MLPEncoder, self).__init__()
        self.output_dim = output_dim

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(-1, 32 * 32)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        h = self.fc3(h)
        h = self.fc4(h)
        z = h.view(x.size(0), self.output_dim)
        return z


class MLPDecoder(nn.Module):
    def __init__(self, input_dim):
        super(MLPDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 1024)
        )

    def forward(self, z):
        h = z.view(z.size(0), -1)
        h = self.net(h)
        mu_img = h.view(z.size(0), 1, 32, 32)
        return mu_img


class VAE(nn.Module):
    def __init__(self, z_dim, use_cuda=False, prior_dist=dist.Normal(), q_dist=dist.Normal(), include_mutinfo=True, tcvae=True, mss=False):
        super(VAE, self).__init__()

        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.include_mutinfo = include_mutinfo
        self.tcvae = tcvae
        self.lamb = 0
        self.beta = 1
        self.mss = mss
        self.x_dist = dist.Bernoulli()

        # Model-specific
        # distribution family of p(z)
        self.prior_dist = prior_dist
        self.q_dist = q_dist
        # hyperparameters for prior p(z)
        self.register_buffer('prior_params', torch.zeros(self.z_dim, 2))

        self.encoder = MLPEncoder(z_dim * self.q_dist.nparams)
        self.decoder = MLPDecoder(z_dim)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()

    # return prior parameters wrapped in a suitable Variable
    def _get_prior_params(self, batch_size=1):
        expanded_size = (batch_size,) + self.prior_params.size()
        prior_params = Variable(self.prior_params.expand(expanded_size))
        return prior_params

    # samples from the model p(x|z)p(z)
    def model_sample(self, batch_size=1):
        # sample from prior (value will be sampled by guide when computing the ELBO)
        prior_params = self._get_prior_params(batch_size)
        zs = self.prior_dist.sample(params=prior_params)
        # decode the latent code z
        x_params = self.decoder.forward(zs)
        return x_params

    # define the guide (i.e. variational distribution) q(z|x)
    def encode(self, x):
        x = x.view(x.size(0), 1, 32, 32)
        # use the encoder to get the parameters used to define q(z|x)
        z_params = self.encoder.forward(x).view(x.size(0), self.z_dim, self.q_dist.nparams)
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
        prior_params = self._get_prior_params(batch_size)
        x_recon, x_params, zs, z_params = self.reconstruct_img(x)
        logpx = self.x_dist.log_density(x, params=x_params).view(batch_size, -1).sum(1)
        logpz = self.prior_dist.log_density(zs, params=prior_params).view(batch_size, -1).sum(1)
        logqz_condx = self.q_dist.log_density(zs, params=z_params).view(batch_size, -1).sum(1)

        elbo = logpx + logpz - logqz_condx

        if self.beta == 1 and self.include_mutinfo and self.lamb == 0:
            return elbo, elbo.detach()

        # compute log q(z) ~= log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m))
        _logqz = self.q_dist.log_density(
            zs.view(batch_size, 1, self.z_dim),
            z_params.view(1, batch_size, self.z_dim, self.q_dist.nparams)
        )

        if not self.mss:
            # minibatch weighted sampling
            logqz_prodmarginals = (logsumexp(_logqz, dim=1, keepdim=False) - math.log(batch_size * dataset_size)).sum(1)
            logqz = (logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(batch_size * dataset_size))
        else:
            # minibatch stratified sampling
            logiw_matrix = Variable(log_importance_weight_matrix(batch_size, dataset_size).type_as(_logqz.data))
            logqz = logsumexp(logiw_matrix + _logqz.sum(2), dim=1, keepdim=False)
            logqz_prodmarginals = logsumexp(
                logiw_matrix.view(batch_size, batch_size, 1) + _logqz, dim=1, keepdim=False).sum(1)

        if not self.tcvae:
            if self.include_mutinfo:
                modified_elbo = logpx - self.beta * (
                    (logqz_condx - logpz) -
                    self.lamb * (logqz_prodmarginals - logpz)
                )
            else:
                modified_elbo = logpx - self.beta * (
                    (logqz - logqz_prodmarginals) +
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)
                )
        else:
            if self.include_mutinfo:
                modified_elbo = logpx - \
                    (logqz_condx - logqz) - \
                    self.beta * (logqz - logqz_prodmarginals) - \
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)
            else:
                modified_elbo = logpx - \
                    self.beta * (logqz - logqz_prodmarginals) - \
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)

        return modified_elbo, elbo.detach()


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
    warmup_iter = 5000
    if args.lambda_anneal:
        vae.lamb = max(0, 0.95 - 1 / warmup_iter * iteration)  # 1 --> 0
    else:
        vae.lamb = 0
    if args.beta_anneal:
        vae.beta = min(args.beta, args.beta / warmup_iter * iteration)  # 0 --> 1
    else:
        vae.beta = args.beta


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-dist', default='normal', type=str, choices=['normal', 'laplace', 'flow'])
    parser.add_argument('-n', '--num-epochs', default=50, type=int, help='number of training epochs')
    parser.add_argument('-b', '--batch-size', default=7000, type=int, help='batch size')
    parser.add_argument('-l', '--learning-rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('-z', '--latent-dim', default=3, type=int, help='size of latent dimension')
    parser.add_argument('--beta', default=1, type=float, help='ELBO penalty term')
    parser.add_argument('--tcvae', default=True, type=bool, help='Use TC VAE')
    parser.add_argument('--exclude-mutinfo', default=False, type=bool, help='Use Mutual Information term or not')
    parser.add_argument('--beta-anneal', default=True, type=bool, help='Use annealing of beta hyperparameter')
    parser.add_argument('--lambda-anneal', default=True, type=bool, help='Use annealing of lambda hyperparameter')
    parser.add_argument('--use_cuda', default=True, type=bool, help='Use cuda or not, set False for CPU testing')
    parser.add_argument('--mss', default=True, type=bool, help='Use Minibatch Stratified Sampling')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--visdom', default=True, type=bool, help='Use Visdom for real time plotting, makes it slower')
    parser.add_argument('--save', default='results/pendulum/train_04_03')   # Important configuration, else will overwrite TODO do it automatically
    parser.add_argument('--log_freq', default=200, type=int, help='num iterations per log')
    args = parser.parse_args()

    if args.use_cuda:
        torch.cuda.set_device(args.gpu)

    # data loader
    kwargs = {'num_workers': 4, 'pin_memory': args.use_cuda}
    # train_loader = DataLoader(dataset=dset.Pendulum(), batch_size=args.batch_size, shuffle=True, **kwargs)
    train_loader = DataLoader(dataset=dset.Pendulum(), batch_size=args.batch_size, shuffle=True) # for debug uncomment this version

    # setup the VAE
    if args.dist == 'normal':
        prior_dist = dist.Normal()
        q_dist = dist.Normal()
    elif args.dist == 'laplace':
        prior_dist = dist.Laplace()
        q_dist = dist.Laplace()
    elif args.dist == 'flow':
        prior_dist = FactorialNormalizingFlow(dim=args.latent_dim, nsteps=32)
        q_dist = dist.Normal()

    vae = VAE(z_dim=args.latent_dim, use_cuda=args.use_cuda, prior_dist=prior_dist, q_dist=q_dist, include_mutinfo=not args.exclude_mutinfo, tcvae=args.tcvae, mss=args.mss)

    # setup the optimizer
    optimizer = optim.Adam(vae.parameters(), lr=args.learning_rate)

    # setup visdom for visualization
    args.visdom = True
    vis = visdom.Visdom(port=8097)  # change here

    train_elbo = []

    # training loop
    dataset_size = len(train_loader.dataset)
    num_iterations = len(train_loader) * args.num_epochs
    iteration = 0
    # initialize loss accumulator
    elbo_running_mean = utils.RunningAverageMeter()
    while iteration < num_iterations:
        for i, values in enumerate(train_loader):
            obs, state, parameter = values
            iteration += 1
            batch_time = time.time()
            vae.train()
            anneal_kl(args, vae, iteration)
            optimizer.zero_grad()
            # transfer to GPU
            if args.use_cuda:
                x = obs.cuda()
            else:
                x = obs

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

                # plot training and test ELBOs
                if args.visdom:
                    display_samples(vae, x, vis)
                    plot_elbo(train_elbo, vis)

                utils.save_checkpoint({
                    'state_dict': vae.state_dict(),
                    'args': args}, args.save, 0)
                plot_latent_vs_gt_pendulum(vae, dset.Pendulum(), os.path.join(args.save, 'gt_vs_latent_{:05d}.png'.format(iteration)))


    # Report statistics after training
    vae.eval()
    utils.save_checkpoint({
        'state_dict': vae.state_dict(),
        'args': args}, args.save, 0)
    validation_loader = DataLoader(dset.Pendulum(), batch_size=2000, shuffle=False)
    logpx, dependence, information, dimwise_kl, analytical_cond_kl, marginal_entropies, joint_entropy = elbo_decomposition_pendulum(vae, validation_loader)
    torch.save({
        'logpx': logpx,
        'dependence': dependence,
        'information': information,
        'dimwise_kl': dimwise_kl,
        'analytical_cond_kl': analytical_cond_kl,
        'marginal_entropies': marginal_entropies,
        'joint_entropy': joint_entropy
    }, os.path.join(args.save, 'elbo_decomposition.pth'))
    '''
    TODO
    metric, marginal_entropies, cond_entropies = mutual_info_metric_pendulum(vae, dset.Pendulum())
    torch.save({
        'metric': metric,
        'marginal_entropies': marginal_entropies,
        'cond_entropies': cond_entropies},
        os.path.join(args.save, 'disentanglement_computation.pth'))
    '''

if __name__ == '__main__':
    model = main()