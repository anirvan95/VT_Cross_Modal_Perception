import os
import time
from datetime import datetime
import argparse
import numpy as np
import random
import visdom
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import utils.compute_utils as utils
from latent_filter import UniModalLF
from utils.datasets import CrossModal
from utils.plot_latent import extract_test_cases, plot_lf, save_lf

# Seeding for reproducibility
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

def compute_elbo(umlf, obs, action, labels, hyperparam, H):
    """
    Computes the ELBO for tactile modality
    :param umlf: tactile latent filter model
    :param obs: tactile observation
    :param action: actions
    :param labels: gt labels
    :param H: % Horizon to feed, rest mask and blind prediction, improves dynamics learning
    :return:
    ELBO Loss
    """
    batch_size, T, _ = action.shape

    tac_object_labels = labels[:, :, 9:] # Required for prior TODO: fix this, unifying into 1 single label
    pose_labels = labels[:, :, :6] # GT Pose of the object

    prior_params_y_f, y_params_f, ys_f, prior_params_z_f, z_params_f, zs_f, xs_hat_f, x_hat_params_f, x_f, hs_f, cs_f = umlf.filter(obs, action, tac_object_labels, H)

    prior_params_y = torch.stack(prior_params_y_f, dim=1).view(batch_size, (T - 1), umlf.dim_y, 2)
    y_params = torch.stack(y_params_f, dim=1).view(batch_size, (T - 1), umlf.dim_y, 2)
    ys = torch.stack(ys_f, dim=1).view(batch_size, (T - 1), umlf.dim_y)
    prior_params_z = torch.stack(prior_params_z_f, dim=1).view(batch_size, T, umlf.dim_z, 2)
    z_params = torch.stack(z_params_f, dim=1).view(batch_size, T, umlf.dim_z, 2)
    zs = torch.stack(zs_f, dim=1).view(batch_size, T, umlf.dim_z)
    x_recon = torch.stack(xs_hat_f, dim=1)

    x_recon_params = torch.stack(x_hat_params_f, dim=1)
    x_t_1 = torch.stack(x_f, dim=1)

    logpx = umlf.vae.x_dist.log_density(x_t_1, params=x_recon_params).view(batch_size, -1).sum(1)
    logpz = umlf.vae.prior_dist.log_density(zs, params=prior_params_z).view(batch_size, -1).sum(1)
    logqz_condx = umlf.vae.q_dist.log_density(zs, params=z_params).view(batch_size, -1).sum(1)

    logpy = umlf.prior_dist_y.log_density(ys, params=prior_params_y).view(batch_size, -1).sum(1)
    logqy_condx = umlf.q_dist_y.log_density(ys, params=y_params).view(batch_size, -1).sum(1)

    elbo = torch.mean(logpx)*0.1 + hyperparam['lamb']*(torch.mean(logpz - logqz_condx)) + hyperparam['beta']*torch.mean(logpy - logqy_condx)

    return x_recon, elbo, elbo.detach()

def anneal_kl(hyperparam, num_iterations, iteration):
    """
    Annealing function for the hyperparameters
    :param hyperparam: dict of the hyperparameters
    :param num_iterations: total number of iteration of training
    :param iteration: current iteration
    :return:
    """
    # Annealing function for the gamma, beta and lamda terms
    decay_rate_lamb = np.log(hyperparam['lamb_end'] / hyperparam['lamb_start']) / num_iterations
    hyperparam['lamb'] = hyperparam['lamb_start'] * np.exp(decay_rate_lamb * iteration)

    decay_rate_beta = np.log(hyperparam['beta_end'] / hyperparam['beta_start']) / num_iterations
    hyperparam['beta'] = hyperparam['beta_start'] * np.exp(decay_rate_beta * iteration)

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--modality', default='tactile', type=str, choices=['vision', 'tactile'])
    parser.add_argument('--num_objects', default=75, type=int, help='Number of objects')
    parser.add_argument('-dist', default='normal', type=str, choices=['normal', 'laplace', 'flow'])
    parser.add_argument('-n', '--num-epochs', default=500, type=int, help='number of training epochs')
    parser.add_argument('-b', '--batch-size', default=32, type=int, help='batch size')
    parser.add_argument('--learning-rate', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--dim_z', default=16, type=int, help='size of latent dimension z')
    parser.add_argument('--dim_y', default=16, type=int, help='size of latent dimension y')
    parser.add_argument('--dim_h', default=32, type=int, help='size of hidden layer dimension of LSTM')
    parser.add_argument('--lambs', default=100, type=float, help='KL Regularization factor start')
    parser.add_argument('--lambe', default=200, type=float, help='KL Regularization factor end')
    parser.add_argument('--betas', default=10, type=float, help='KL Regularization factor start')
    parser.add_argument('--betae', default=100, type=float, help='KL Regularization factor end')
    parser.add_argument('--stdx', default=0.75, type=float, help='Standard Deviation of observation')
    parser.add_argument('--stdy', default=0.75, type=float, help='Standard Deviation of y space')
    parser.add_argument('--lambda-anneal', default=True, type=bool, help='Use annealing of lambda hyperparameter or Constrained optimisation')
    parser.add_argument('--beta-anneal', default=True, type=bool, help='Use annealing of beta hyperparameter or Constrained optimisation')
    parser.add_argument('--use_cuda', default=True, type=bool, help='Use cuda or not, set False for CPU testing')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--visdom', default=False, type=bool, help='Use Visdom for real time plotting, makes it slower')
    parser.add_argument('--plotting', default=False, type=bool, help='Alternative to visdom for saving the plots, makes it slower')
    parser.add_argument('--save', default='results/unimodallf/tactile', help='Path to save the models')
    parser.add_argument('--debug_dir', default='dump/training/unimodallf/tactile', help='Path to save the dump files')
    parser.add_argument('--log_freq', default=50, type=int, help='num iterations per log')

    args = parser.parse_args()

    if args.use_cuda:
        torch.cuda.set_device(args.gpu)
        args.cuda = True

    # data loader
    print('Loading Dataset')
    train_dir = 'dataset/cm_dataset/training'
    file_paths = sorted([os.path.join(train_dir, file) for file in os.listdir(train_dir) if file.endswith('.npz')])
    dataset = CrossModal(file_paths)
    print('Done')

    train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=dataset, batch_size=3, shuffle=False)

    # Create folders for the saving the model
    now = datetime.now()
    date_time = now.strftime("%m_%d_%H_%M")
    out_dir = os.path.join(args.save, date_time)
    debug_dir = os.path.join(args.debug_dir, date_time)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    tactile_lf = UniModalLF(args)

    # Save the args in the debug directory
    args_dict = vars(args)  # Convert Namespace to dict
    with open(out_dir + '/args.txt', 'w') as file:
        for key, value in args_dict.items():
            file.write(f"{key}: {value}\n")

    # setup the optimizer
    optimizer = optim.Adam(tactile_lf.parameters(), lr=args.learning_rate)

    # setup hyperparameter dict
    hyperparam = {'lamb': args.lambs, 'lamb_start': args.lambs, 'lamb_end': args.lambe, 'beta': args.betas, 'beta_start': args.betas, 'beta_end': args.betae}

    # setup visdom for visualization
    if args.visdom:
        vis = visdom.Visdom(port=8097)

    if args.visdom or args.plotting:
        x_test, a_test, label_test, label_scatter, color_vals = extract_test_cases(test_loader)

    train_elbo = []

    # training loop
    num_iterations = len(train_loader) * args.num_epochs
    iteration = 0
    elbo_running_mean = utils.RunningAverageMeter()
    print('Total iteration of training: ', num_iterations)
    while iteration < num_iterations:
        for i, values in enumerate(train_loader):
            _, tac_obs, actions, labels = values
            batch_time = time.time()
            tactile_lf.train()
            optimizer.zero_grad()

            # transfer to GPU and ensure float32 type
            if args.use_cuda:
                tac_obs = tac_obs.cuda().to(dtype=torch.float32)
                actions = actions.cuda().to(dtype=torch.float32)
                labels = labels.cuda().to(dtype=torch.float32)
            else:
                tac_obs = tac_obs.to(dtype=torch.float32)
                actions = actions.to(dtype=torch.float32)
                labels = labels.to(dtype=torch.float32)

            # do ELBO gradient and accumulate loss
            reconstructed, obj, elbo = compute_elbo(tactile_lf, tac_obs, actions, labels, hyperparam, 1)

            if utils.isnan(obj).any():
                raise ValueError('NaN spotted in objective.')

            obj.mul(-1).backward()
            elbo_running_mean.update(elbo.mean().item())
            optimizer.step()
            if args.lambda_anneal:
                anneal_kl(hyperparam, num_iterations, iteration)

            # report training diagnostics
            if iteration % args.log_freq == 0:
                train_elbo.append(elbo_running_mean.avg)
                print(
                    '[iteration %03d] time: %.2f \tlambda %.2f training ELBO: %.4f (%.4f)' % (
                        iteration, time.time() - batch_time, hyperparam['lamb'],
                        elbo_running_mean.val, elbo_running_mean.avg))

                tactile_lf.eval()

                utils.save_checkpoint({
                    'state_dict': tactile_lf.state_dict(),
                    'args': args}, out_dir, iteration)

                if args.visdom:
                    # For real time evaluation, slightly faster
                    plot_lf(tactile_lf, vis, x_test, a_test, label_test, label_scatter, color_vals)
                elif args.plotting:
                    # For tuning of parameters
                    save_lf(tactile_lf, x_test, a_test, label_test, label_scatter, color_vals, iteration, out_dir)

            iteration += 1


if __name__ == '__main__':
    trainer = main()