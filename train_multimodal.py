import os
import re
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
from latent_filter_multimodal import MultiModalLF
from utils.datasets import CrossModal
from utils.plot_latent import validate_mmlf

# Seeding for reproducibility
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)


def compute_elbo(mmlf, vis_obs, tac_obs, actions, labels, hyperparam):
    """
    Computes the ELBO Loss for training the cross-modal model
    :param mmlf: model
    :param vis_obs:
    :param tac_obs:
    :param action:
    :param labels:
    :param hyperparam: lambda and beta parameter balance recon and regular.
    :param H: cross-modal activation signal
    :return: elbo loss
    """
    batch_size, T, _ = actions.shape

    object_labels = labels[:, :, 6:] # Required for hierarchical prior, contrastive version is more general
    pose_labels = labels[:, :, :6] # GT Pose of the object

    (prior_params_y_f, y_params_f, ys_f, hs_f, cs_f,
     vis_prior_params_z_f, vis_z_params_f, vis_zs_f, vis_xs_hat_f, vis_x_hat_params_f, vis_x_f,
     tac_prior_params_z_f, tac_z_params_f, tac_zs_f, tac_xs_hat_f, tac_x_hat_params_f, tac_x_f)= mmlf.filter(vis_obs, tac_obs, actions, object_labels)

    prior_params_y = torch.stack(prior_params_y_f, dim=1).view(batch_size, (T - 1), mmlf.dim_y, 2)
    y_params = torch.stack(y_params_f, dim=1).view(batch_size, (T - 1), mmlf.dim_y, 2)
    ys = torch.stack(ys_f, dim=1).view(batch_size, (T - 1), mmlf.dim_y)
    logpy = mmlf.prior_dist_y.log_density(ys, params=prior_params_y).view(batch_size, -1).sum(1)
    logqy_condx = mmlf.q_dist_y.log_density(ys, params=y_params).view(batch_size, -1).sum(1)

    # ############################################### Vision #######################################
    vis_prior_params_z = torch.stack(vis_prior_params_z_f, dim=1).view(batch_size, T, mmlf.vis_dim_z, 2)
    vis_z_params = torch.stack(vis_z_params_f, dim=1).view(batch_size, T, mmlf.vis_dim_z, 2)
    vis_zs = torch.stack(vis_zs_f, dim=1).view(batch_size, T, mmlf.vis_dim_z)

    vis_x_recon_params = torch.stack(vis_x_hat_params_f, dim=1)
    vis_x_t_1 = torch.stack(vis_x_f, dim=1)

    vis_logpx = mmlf.vis_vae.x_dist.log_density(vis_x_t_1, params=vis_x_recon_params).view(batch_size, -1).sum(1)
    vis_logpz = mmlf.vis_vae.prior_dist.log_density(vis_zs, params=vis_prior_params_z).view(batch_size, -1).sum(1)
    vis_logqz_condx = mmlf.vis_vae.q_dist.log_density(vis_zs, params=vis_z_params).view(batch_size, -1).sum(1)

    vis_recon = (torch.mean(vis_logpx) + hyperparam['lamb']*torch.mean(vis_logpz - vis_logqz_condx))

    # ############################################### Tactile #######################################
    tac_prior_params_z = torch.stack(tac_prior_params_z_f, dim=1).view(batch_size, T, mmlf.tac_dim_z, 2)
    tac_z_params = torch.stack(tac_z_params_f, dim=1).view(batch_size, T, mmlf.tac_dim_z, 2)
    tac_zs = torch.stack(tac_zs_f, dim=1).view(batch_size, T, mmlf.tac_dim_z)

    tac_x_recon_params = torch.stack(tac_x_hat_params_f, dim=1)
    tac_x_t_1 = torch.stack(tac_x_f, dim=1)

    tac_logpx = mmlf.tac_vae.x_dist.log_density(tac_x_t_1, params=tac_x_recon_params).view(batch_size, -1).sum(1)
    tac_logpz = mmlf.tac_vae.prior_dist.log_density(tac_zs, params=tac_prior_params_z).view(batch_size, -1).sum(1)
    tac_logqz_condx = mmlf.tac_vae.q_dist.log_density(tac_zs, params=tac_z_params).view(batch_size, -1).sum(1)

    tac_recon = (torch.mean(tac_logpx) + hyperparam['lamb'] * torch.mean(tac_logpz - tac_logqz_condx))

    elbo = (vis_recon + tac_recon + hyperparam['beta'] * torch.mean(logpy - logqy_condx))*0.05

    return elbo, elbo.detach()

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
    parser.add_argument('--num_objects', default=75, type=int, help='Number of objects')
    parser.add_argument('-dist', default='normal', type=str, choices=['normal', 'laplace', 'flow'])
    parser.add_argument('-n', '--num-epochs', default=250, type=int, help='number of training epochs')
    parser.add_argument('-b', '--batch-size', default=48, type=int, help='batch size')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--vis_dim_z', default=16, type=int, help='size of latent dimension visual z')
    parser.add_argument('--tac_dim_z', default=16, type=int, help='size of latent dimension tactile z')
    parser.add_argument('--dim_y', default=32, type=int, help='size of latent dimension tactile y')
    parser.add_argument('--dim_h', default=64, type=int, help='size of hidden layer dimension of tactile LSTM')
    parser.add_argument('--vis_stdx', default=0.5, type=float, help='standard deviation of visual observation')
    parser.add_argument('--tac_stdx', default=0.1, type=float, help='standard deviation of tactile observation')
    parser.add_argument('--stdy', default=0.1, type=float, help='standard deviation of tactile y space')
    parser.add_argument('--lambs', default=1e2, type=float, help='KL regularization factor start')
    parser.add_argument('--lambe', default=2e2, type=float, help='KL regularization factor end')
    parser.add_argument('--betas', default=1, type=float, help='KL regularization factor start')
    parser.add_argument('--betae', default=20, type=float, help='KL regularization factor end')
    parser.add_argument('--lambda-anneal', default=True, type=bool, help='Use annealing of lambda hyperparameter or Constrained optimisation')
    parser.add_argument('--beta-anneal', default=True, type=bool, help='Use annealing of beta hyperparameter or Constrained optimisation')
    parser.add_argument('--use_cuda', default=True, type=bool, help='Use cuda or not, set False for CPU testing')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--validate', default=True, type=bool, help='Perform validation or not, avoids overfitting')
    parser.add_argument('--showplots', default=True, type=bool, help='Use Visdom for real time plotting, makes it slower')
    parser.add_argument('--saveplots', default=False, type=bool, help='Alternative to visdom for saving the plots, makes it slower')
    parser.add_argument('--save', default='results/multimodallf/', help='Path to save the models')
    parser.add_argument('--debug_dir', default='dump/training/multimodallf/', help='Path to save the dump files')
    parser.add_argument('--log_freq', default=500, type=int, help='num iterations per log')

    args = parser.parse_args()

    if args.use_cuda:
        torch.cuda.set_device(args.gpu)
        args.cuda = True

    if args.showplots:
        vis = visdom.Visdom(port=8097)
    else:
        vis = None

    # Data Loader
    print('Loading Dataset')
    train_dir = 'dataset/cm_dataset/training'
    train_file_paths = sorted(
        [os.path.join(train_dir, file) for file in os.listdir(train_dir) if file.endswith('.npz')],
        key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group())  # Extract first numeric part
    )
    train_dataset = CrossModal(train_file_paths)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    if args.validate:
        val_dir = 'dataset/cm_dataset/validation'
        val_file_paths = sorted(
            [os.path.join(val_dir, file) for file in os.listdir(val_dir) if file.endswith('.npz')],
            key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group())  # Extract first numeric part
        )
        val_dataset = CrossModal(val_file_paths)
        test_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)

    print('Done Loading Dataset')

    # Create folders for the saving the model
    now = datetime.now()
    date_time = now.strftime("%m_%d_%H_%M")
    out_dir = os.path.join(args.save, date_time)
    checkpoint_dir = os.path.join(args.save, date_time, 'checkpoints')
    debug_dir = os.path.join(args.debug_dir, date_time)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    mmlf = MultiModalLF(args)

    # Save the args in the debug directory
    args_dict = vars(args)  # Convert Namespace to dict
    with open(out_dir + '/args.txt', 'w') as file:
        for key, value in args_dict.items():
            file.write(f"{key}: {value}\n")

    # Setup the optimizer
    optimizer = optim.Adam(mmlf.parameters(), lr=args.learning_rate)

    # Setup hyperparameter dict
    hyperparam = {'lamb': args.lambs, 'lamb_start': args.lambs, 'lamb_end': args.lambe, 'beta': args.betas, 'beta_start': args.betas, 'beta_end': args.betae}

    # Training loop
    num_iterations = len(train_loader) * args.num_epochs
    iteration = 0
    train_elbo = []
    elbo_running_mean = utils.RunningAverageMeter()
    print('Total iteration of training: ', num_iterations)
    while iteration < num_iterations:
        for i, values in enumerate(train_loader):
            vis_obs, tac_obs, actions, labels = values
            batch_time = time.time()
            mmlf.train()
            optimizer.zero_grad()

            # transfer to GPU and ensure float32 type
            if args.use_cuda:
                vis_obs = vis_obs.cuda().to(dtype=torch.float32)
                tac_obs = tac_obs.cuda().to(dtype=torch.float32)
                actions = actions.cuda().to(dtype=torch.float32)
                labels = labels.cuda().to(dtype=torch.float32)
            else:
                vis_obs = vis_obs.to(dtype=torch.float32)
                tac_obs = tac_obs.to(dtype=torch.float32)
                actions = actions.to(dtype=torch.float32)
                labels = labels.to(dtype=torch.float32)

            # ELBO gradient and accumulate loss
            obj, elbo = compute_elbo(mmlf, vis_obs, tac_obs, actions, labels, hyperparam)

            if utils.isnan(obj).any():
                raise ValueError('NaN spotted in objective.')

            obj.mul(-1).backward()
            elbo_running_mean.update(elbo.mean().item())
            optimizer.step()
            if args.lambda_anneal:
                anneal_kl(hyperparam, num_iterations, iteration)

            # Report training diagnostics
            if iteration % args.log_freq == 0:
                train_elbo.append(elbo_running_mean.avg)
                print(
                    '[iteration %03d] time: %.2f \tlambda %.2f \tbeta %.2f training ELBO: %.4f (%.4f)' % (
                        iteration, time.time() - batch_time, hyperparam['lamb'], hyperparam['beta'],
                        elbo_running_mean.val, elbo_running_mean.avg))

                utils.save_checkpoint({
                    'state_dict': mmlf.state_dict(),
                    'args': args}, checkpoint_dir, iteration)

                if args.validate:
                    mmlf.eval()
                    validate_mmlf(test_loader, mmlf, out_dir, iteration, save_plot=args.saveplots, show_plot=args.showplots, vis=vis)

            iteration += 1


if __name__ == '__main__':
    trainer = main()