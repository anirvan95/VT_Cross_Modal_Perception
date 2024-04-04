import os
import time
import math
from numbers import Number
import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import visdom
import lib.dist as dist
import lib.utils as utils
import lib.datasets as dset
from lib.flows import FactorialNormalizingFlow


# from elbo_decomposition import elbo_decomposition
# from plot_latent_vs_true import plot_vs_gt_shapes, plot_vs_gt_faces  # noqa: F401


class GammaFunc(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GammaFunc, self).__init__()
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, output_dim)
        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.act(self.fc2(h))
        h = self.fc3(h)
        z = h.view(x.size(0), self.output_dim)
        return z


class ConvEncoder(nn.Module):
    def __init__(self, output_dim):
        super(ConvEncoder, self).__init__()
        self.output_dim = output_dim

        # self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)  # 32 x 32
        # self.bn1 = nn.BatchNorm2d(32)
        # self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)  # 16 x 16
        # self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(1, 64, 4, 2, 1)  # 8 x 8
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 4, 2, 1)  # 4 x 4
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 512, 4)  # 512, 1
        self.bn5 = nn.BatchNorm2d(512)
        self.conv_z = nn.Conv2d(512, output_dim, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(-1, 1, 16, 16)
        # h = self.act(self.bn1(self.conv1(h)))
        # h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.act(self.bn5(self.conv5(h)))
        z = self.conv_z(h).view(x.size(0), self.output_dim)
        return z


class ConvDecoder(nn.Module):
    def __init__(self, input_dim):
        super(ConvDecoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_dim, 512, 1, 1, 0)  # 1 x 1
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 64, 4, 1, 0)  # 4 x 4
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)  # 8 x 8
        self.bn3 = nn.BatchNorm2d(32)
        # self.conv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)  # 16 x 16
        # self.bn4 = nn.BatchNorm2d(32)
        # self.conv5 = nn.ConvTranspose2d(32, 32, 4, 2, 1)  # 32 x 32
        # self.bn5 = nn.BatchNorm2d(32)
        self.conv_final = nn.ConvTranspose2d(32, 1, 4, 2, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, z):
        h = z.view(z.size(0), z.size(1), 1, 1)
        h = self.act(self.bn1(self.conv1(h)))   # 512, 1
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        # h = self.act(self.bn4(self.conv4(h)))
        # h = self.act(self.bn5(self.conv5(h)))
        mu_img = self.conv_final(h)
        return mu_img


class VAE(nn.Module):
    def __init__(self, beta_dim, gamma_dim, use_cuda=False, prior_dist=dist.Normal(), q_dist=dist.Normal(),
                 include_mutinfo=True, tcvae=False, conv=False, mss=False):
        super(VAE, self).__init__()

        self.use_cuda = use_cuda
        self.beta_dim = beta_dim
        self.gamma_dim = gamma_dim
        self.z_dim = self.beta_dim + self.gamma_dim

        self.include_mutinfo = include_mutinfo
        self.tcvae = tcvae
        self.lamb = 0
        self.kappa = 6
        self.mss = mss
        self.x_dist = dist.Bernoulli()

        # Model-specific
        # distribution family of p(z) = p(gamma|beta)p(beta)
        self.prior_dist_gamma = prior_dist
        self.q_dist_gamma = q_dist
        self.prior_dist_beta = prior_dist
        self.q_dist_beta = q_dist

        # hyperparameters for prior p(gamma|beta)
        self.register_buffer('prior_params_gamma', torch.zeros(self.gamma_dim, 2))
        # hyperparameters for prior p(beta)
        self.register_buffer('prior_params_beta', torch.zeros(self.beta_dim, 2))

        self.encoder = ConvEncoder(self.beta_dim * self.q_dist_beta.nparams)
        self.gamma_func = GammaFunc(self.beta_dim, self.gamma_dim*self.q_dist_gamma.nparams)
        self.decoder = ConvDecoder(self.gamma_dim)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()

    # return prior parameters wrapped in a suitable Variable
    def _get_prior_params(self, batch_size=1):
        expanded_size_beta = (batch_size,) + self.prior_params_beta.size()
        prior_params_beta = Variable(self.prior_params_beta.expand(expanded_size_beta))
        prior_betas = self.prior_dist_beta.sample(params=prior_params_beta)
        prior_params_gamma = self.gamma_func(prior_betas).view(batch_size, self.gamma_dim, 2)

        # expanded_size_gamma = (batch_size,) + self.prior_params_gamma.size()
        # prior_params_gamma = Variable(self.prior_params_gamma.expand(expanded_size_gamma))

        return prior_params_beta, prior_params_gamma

    # samples from the model p(x|gamma)p(gamma)
    def model_sample(self, batch_size=1):
        # sample from prior (value will be sampled by guide when computing the ELBO)
        prior_params_beta, _ = self._get_prior_params(batch_size)
        betas = self.prior_dist_beta.sample(params=prior_params_beta)
        gamma_params = self.gamma_func(betas).view(betas.size(0), self.gamma_dim, self.q_dist_gamma.nparams)
        gammas = self.q_dist_gamma.sample(params=gamma_params)

        # decode the latent code z
        x_params = self.decoder.forward(gammas)
        return x_params

    # define the guide (i.e. variational distribution) q(z|x)
    def encode(self, x):
        x = x.view(x.size(0), 1, 16, 16)
        # use the encoder to get the parameters used to define q(z|x)
        beta_params = self.encoder.forward(x).view(x.size(0), self.beta_dim, self.q_dist_beta.nparams)
        # sample the latent code z
        betas = self.q_dist_beta.sample(params=beta_params)
        return betas, beta_params

    def decode(self, gamma):
        x_params = self.decoder.forward(gamma).view(gamma.size(0), 1, 16, 16)
        xs = self.x_dist.sample(params=x_params)
        return xs, x_params

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        betas, beta_params = self.encode(x)
        # gamma_params = self.gamma_func(betas).view(betas.size(0), self.gamma_dim, self.q_dist_gamma.nparams)
        # gammas = self.q_dist_gamma.sample(params=gamma_params)
        xs, x_params = self.decode(betas)
        return xs, x_params, betas, beta_params

    def _log_importance_weight_matrix(self, batch_size, dataset_size):
        N = dataset_size
        M = batch_size - 1
        strat_weight = (N - M) / (N * M)
        W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
        W.view(-1)[::M + 1] = 1 / N
        W.view(-1)[1::M + 1] = strat_weight
        W[M - 1, 0] = strat_weight
        return W.log()

    def elbo(self, x, dataset_size):
        # log p(x|z) + log p(z) - log q(z|x)
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 16, 16)
        prior_params_beta, prior_params_gamma = self._get_prior_params(batch_size)

        # xs, x_params, betas, beta_params, gammas, gamma_params = self.reconstruct_img(x)
        xs, x_params, betas, beta_params = self.reconstruct_img(x)

        # Likelihood of output - log p(x|z)
        logpx = self.x_dist.log_density(x, params=x_params).view(batch_size, -1).sum(1)

        # Priors and conditionals
        #logpgamma = self.prior_dist_gamma.log_density(gammas, params=prior_params_gamma).view(batch_size, -1).sum(1)
        #logqgamma_cond_beta_x = self.q_dist_gamma.log_density(gammas, params=gamma_params).view(batch_size, -1).sum(1)

        logpbeta = self.prior_dist_beta.log_density(betas, params=prior_params_beta).view(batch_size, -1).sum(1)
        logqbeta_condx = self.q_dist_beta.log_density(betas, params=beta_params).view(batch_size, -1).sum(1)

        # elbo = logpx + logpgamma - logqgamma_cond_beta_x + logpbeta - logqbeta_condx
        elbo = logpx + logpbeta - logqbeta_condx
        return elbo, elbo.detach()

        '''
        if self.kappa == 1 and self.include_mutinfo and self.lamb == 0:
            

        # compute log q(beta) ~= log 1/(NM) sum_m=1^M q(beta|x_m) = - log(MN) + logsumexp_m(q(beta|x_m))
        _logqbeta = self.q_dist_beta.log_density(
            betas.view(batch_size, 1, self.beta_dim),
            beta_params.view(1, batch_size, self.beta_dim, self.q_dist_beta.nparams)
        )

        if not self.mss:
            # minibatch weighted sampling
            logqbeta_prodmarginals = (logsumexp(_logqbeta, dim=1, keepdim=False) - math.log(batch_size * dataset_size)).sum(1)
            logqbeta = (logsumexp(_logqbeta.sum(2), dim=1, keepdim=False) - math.log(batch_size * dataset_size))
        else:
            # minibatch stratified sampling
            logiw_matrix = Variable(self._log_importance_weight_matrix(batch_size, dataset_size).type_as(_logqbeta.data))
            logqbeta = logsumexp(logiw_matrix + _logqbeta.sum(2), dim=1, keepdim=False)
            logqbeta_prodmarginals = logsumexp(logiw_matrix.view(batch_size, batch_size, 1) + _logqbeta, dim=1, keepdim=False).sum(1)

        modified_elbo = logpx + (logpgamma - logqgamma_cond_beta_x) - (logqbeta_condx - logqbeta) - self.kappa * (logqbeta - logqbeta_prodmarginals) - (1 - self.lamb) * (logqbeta_prodmarginals - logpbeta)


        if not self.tcvae:
            if self.include_mutinfo:
                modified_elbo = logpx - self.beta_param * ((logqz_condx - logpz) - self.lamb * (logqz_prodmarginals - logpz))
            else:
                modified_elbo = logpx - self.beta_param * ((logqz - logqz_prodmarginals) + (1 - self.lamb) * (logqz_prodmarginals - logpz))
        else:
            if self.include_mutinfo:
                modified_elbo = logpx - \
                                (logqz_condx - logqz) - \
                                self.beta_param * (logqz - logqz_prodmarginals) - \
                                (1 - self.lamb) * (logqz_prodmarginals - logpz)
            else:
                modified_elbo = logpx - \
                                self.beta_param * (logqz - logqz_prodmarginals) - \
                                (1 - self.lamb) * (logqz_prodmarginals - logpz)

        return modified_elbo, elbo.detach()
        '''


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


# for loading and batching datasets
def setup_data_loaders(args, use_cuda=False):
    if args.dataset == 'shapes':
        train_set = dset.Shapes()
    elif args.dataset == 'faces':
        train_set = dset.Faces()
    elif args.dataset == 'pendulum':
        train_set = dset.Pendulum()
    else:
        raise ValueError('Unknown dataset ' + str(args.dataset))

    # kwargs = {'num_workers': 4, 'pin_memory': use_cuda}
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    return train_loader


win_samples = None
win_test_reco = None
win_latent_walk = None
win_train_elbo = None

'''
def display_samples(model, x, vis):
    global win_samples, win_test_reco, win_latent_walk
    fig, ax = plt.subplots(1, 2)
    # plot random samples
    sample_mu = model.model_sample(batch_size=100).sigmoid()
    sample_mu = sample_mu
    images = list(sample_mu.view(-1, 1, 64, 64).data.cpu())
    # win_samples = vis.images(images, 10, 2, opts={'caption': 'samples'}, win=win_samples)
    tt = images[0]
    ax[0].imshow(tt[0, :, :])

    # plot the reconstructed distribution for the first 50 test images
    test_imgs = x[:50, :]
    _, reco_imgs, betas, beta_params, gammas, gamma_params = model.reconstruct_img(test_imgs)
    # _, reco_imgs, zs, _ = model.reconstruct_img(test_imgs)
    reco_imgs = reco_imgs.sigmoid()
    test_reco_imgs = torch.cat([test_imgs.view(1, -1, 64, 64), reco_imgs.view(1, -1, 64, 64)], 0).transpose(0, 1)
    beta_plotter = betas.cpu()
    ax[1].plot(beta_plotter[:, 0], beta_plotter[:, 1], '.r')
    plt.show()

    win_test_reco = vis.images(
        list(test_reco_imgs.contiguous().view(-1, 1, 64, 64).data.cpu()), 10, 2,
        opts={'caption': 'test reconstruction image'}, win=win_test_reco)

    # plot latent walks in betas (change one variable while all others stay the same)
    betas = betas[0:3]
    batch_size, beta_dim = betas.size()
    xs = []
    delta = torch.autograd.Variable(torch.linspace(-2, 2, 7), volatile=True).type_as(betas)
    for i in range(beta_dim):
        vec = Variable(torch.zeros(beta_dim)).view(1, beta_dim).expand(7, beta_dim).contiguous().type_as(betas)
        vec[:, i] = 1
        vec = vec * delta[:, None]
        betas_delta = betas.clone().view(batch_size, 1, beta_dim)
        betas_delta[:, :, i] = 0
        betas_walk = betas_delta + vec[None]

        gamma_params = model.gamma_func(betas_walk.view(-1, beta_dim)).view(21, 6, 2)
        gammas_walk = model.q_dist_gamma.sample(params=gamma_params)
        xs_walk = model.decoder.forward(gammas_walk).sigmoid()     # check dimenstion
        xs.append(xs_walk)

    xs = list(torch.cat(xs, 0).data.cpu())
    win_latent_walk = vis.images(xs, 7, 2, opts={'caption': 'latent walk'}, win=win_latent_walk)


def plot_elbo(train_elbo, vis):
    global win_train_elbo
    win_train_elbo = vis.line(torch.Tensor(train_elbo), opts={'markers': True}, win=win_train_elbo)
'''


def anneal_kl(args, vae, iteration):
    if args.dataset == 'shapes':
        warmup_iter = 7000
    elif args.dataset == 'faces':
        warmup_iter = 2500
    elif args.dataset == 'pendulum':
        warmup_iter = 5000

    if args.lambda_anneal:
        vae.lamb = max(0, 0.95 - 1 / warmup_iter * iteration)  # 1 --> 0
    else:
        vae.lamb = 0
    if args.kappa_anneal:
        vae.kappa = min(args.kappa, args.kappa / warmup_iter * iteration)  # 0 --> 1
    else:
        vae.kappa = args.kappa


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-d', '--dataset', default='pendulum', type=str, help='dataset name', choices=['shapes', 'faces', 'pendulum'])
    parser.add_argument('-dist', default='normal', type=str, choices=['normal', 'laplace', 'flow'])
    parser.add_argument('-n', '--num-epochs', default=2e4, type=int, help='number of training epochs')
    parser.add_argument('-b', '--batch-size', default=2048, type=int, help='batch size')
    parser.add_argument('-l', '--learning-rate', default=2e-4, type=float, help='learning rate')
    parser.add_argument('--kappa', default=6, type=float, help='ELBO penalty term')
    parser.add_argument('--tcvae', action='store_true')
    parser.add_argument('--exclude-mutinfo', action='store_true')
    parser.add_argument('--kappa-anneal', action='store_true')
    parser.add_argument('--lambda-anneal', action='store_true')
    parser.add_argument('--mss', action='store_true', help='use the improved minibatch estimator')
    parser.add_argument('--conv', action='store_false')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--visdom', action='store_false', help='whether plotting in visdom is desired')
    parser.add_argument('--save', default='pendulum_exp_2_vae')
    parser.add_argument('--log_freq', default=5, type=int, help='num iterations per log')
    args = parser.parse_args()

    # torch.cuda.set_device(args.gpu)
    device = 'cuda'

    # data loader
    train_loader = setup_data_loaders(args, use_cuda=False)

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

    vae = VAE(beta_dim=15, gamma_dim=15, use_cuda=False, prior_dist=prior_dist, q_dist=q_dist, include_mutinfo=not args.exclude_mutinfo, tcvae=args.tcvae, conv=False, mss=args.mss).to(device)

    # setup the optimizer
    optimizer = optim.Adam(vae.parameters(), lr=args.learning_rate)

    # setup visdom for visualization
    # vis = visdom.Visdom(env=args.save, port=8097)

    train_elbo = []

    # training loop
    dataset_size = len(train_loader.dataset)
    num_iterations = len(train_loader) * args.num_epochs
    iteration = 0
    # initialize loss accumulator
    elbo_running_mean = utils.RunningAverageMeter()
    while iteration < num_iterations:
        for i, x in enumerate(train_loader):
            iteration += 1
            batch_time = time.time()
            vae.train()
            anneal_kl(args, vae, iteration)
            optimizer.zero_grad()
            # transfer to GPU
            x = x.to(device)
            # do ELBO gradient and accumulate loss
            obj, elbo = vae.elbo(x, dataset_size)

            if utils.isnan(obj).any():
                raise ValueError('NaN spotted in objective.')
            obj.mean().mul(-1).backward()
            elbo_running_mean.update(elbo.mean())
            optimizer.step()

            # report training diagnostics
            if iteration % args.log_freq == 0:
                train_elbo.append(elbo_running_mean.avg)
                print('[iteration %03d] time: %.2f \tkappa %.2f \tlambda %.2f training ELBO: %.4f (%.4f)' % (
                    iteration, time.time() - batch_time, vae.kappa, vae.lamb,
                    elbo_running_mean.val, elbo_running_mean.avg))

                vae.eval()
                # display_samples(vae, x, vis)
                # plot_elbo(train_elbo, vis)

                utils.save_checkpoint({
                    'state_dict': vae.state_dict(),
                    'args': args}, args.save, 0)



    '''
    # Report statistics after training
    vae.eval()
    utils.save_checkpoint({
        'state_dict': vae.state_dict(),
        'args': args}, args.save, 0)
    dataset_loader = DataLoader(train_loader.dataset, batch_size=1000, num_workers=1, shuffle=False)
    logpx, dependence, information, dimwise_kl, analytical_cond_kl, marginal_entropies, joint_entropy = \
        elbo_decomposition(vae, dataset_loader)
    torch.save({
        'logpx': logpx,
        'dependence': dependence,
        'information': information,
        'dimwise_kl': dimwise_kl,
        'analytical_cond_kl': analytical_cond_kl,
        'marginal_entropies': marginal_entropies,
        'joint_entropy': joint_entropy
    }, os.path.join(args.save, 'elbo_decomposition.pth'))
    eval('plot_vs_gt_' + args.dataset)(vae, dataset_loader.dataset, os.path.join(args.save, 'gt_vs_latent.png'))
    '''
    return vae


if __name__ == '__main__':
    model = main()
