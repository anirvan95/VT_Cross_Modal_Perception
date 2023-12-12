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
from models.tcvae_model import VAE

# from elbo_decomposition import elbo_decomposition
# from plot_latent_vs_true import plot_vs_gt_shapes, plot_vs_gt_faces  # noqa: F401


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
    parser.add_argument('-l', '--learning-rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--kappa', default=6, type=float, help='ELBO penalty term')
    parser.add_argument('--tcvae', action='store_true')
    parser.add_argument('--exclude-mutinfo', action='store_true')
    parser.add_argument('--kappa-anneal', action='store_true')
    parser.add_argument('--lambda-anneal', action='store_true')
    parser.add_argument('--mss', action='store_true', help='use the improved minibatch estimator')
    parser.add_argument('--conv', action='store_false')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--visdom', action='store_false', help='whether plotting in visdom is desired')
    parser.add_argument('--save', default='runs/vae/pendulum_exp_2_vae')
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
