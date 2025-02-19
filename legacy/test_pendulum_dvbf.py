from train_pendulum_vae import VAE
import torch
torch.manual_seed(0)
import torch.nn as nn
import numpy as np
import dbvf_utils.dist as dist
from torch.utils.data import DataLoader
import dbvf_utils.datasets as dataset
import tcvae_utils.utils as utils
import argparse
import torch.optim as optim
import time
import imageio


class LSTMEncode(nn.Module):
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
    def __init__(self, dim_z, dim_beta, use_cuda=False, load_vae=False):
        super(DVBF, self).__init__()
        self.dim_z = dim_z
        self.dim_beta = dim_beta
        self.use_cuda = use_cuda
        self.dt = 0.05
        # distribution family of p(beta)
        self.prior_dist_beta = dist.Normal()
        self.q_dist_beta = dist.Normal()
        # hyperparameters for prior p(z)
        self.register_buffer('prior_params_beta', torch.zeros(self.dim_beta, 2))

        # Recognition Model  - Visual Encoder & Decoder
        self.vae = VAE(dim_z, use_cuda=use_cuda)

        if load_vae:
            # TODO: improve checkpoint loading
            checkpoint_path = 'results/pendulum/tcvae_train/train_06_03/checkpt-0000.pth'
            checkpt = torch.load(checkpoint_path)
            state_dict = checkpt['state_dict']
            self.vae.load_state_dict(state_dict, strict=False)
            print('VAE loaded successfully')
            # TODO: should we add freezing of the VAE or not

        # Recognition Model - Hidden States LSTM Network to identify hidden properties from changing state space z
        self.beta_net = LSTMEncode(input_size=dim_z, hidden_size=128, output_size=dim_beta * self.q_dist_beta.nparams)

        # Transition Network, computes TRANSITION (from previous visual state z, inferred beta (time-invariant properties) and action)
        self.transition_net = MLPTransition(input_dim=dim_z + dim_beta + 1, output_dim=dim_z)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()

    def get_prior_params_beta(self, batch_size=1):
        expanded_size = (batch_size,) + self.prior_params_beta.size()
        prior_params_beta = self.prior_params_beta.expand(expanded_size)
        return prior_params_beta

    # TODO: Add model sample function to obtain the latent space analysis

    def filter(self, x, u, H):
        batch_size, T, _ = u.shape

        # Set the hidden layer of the LSTM to zero
        hidden_beta_t = (torch.zeros(1, batch_size, 128, device='cuda'),
                         torch.zeros(1, batch_size, 128, device='cuda'))

        # Define the variables to store the filtering output
        prior_params_beta_f = []
        beta_params_f = []
        betas_f = []
        prior_params_z_f = []
        z_params_f = []
        zs_f = []
        x_hat_params_f = []
        xs_hat_f = []
        x_f = []

        for t in range(0, H):
            if t < T-1:
                u_t = u[:, t]
                x_t = x[:, t]
                x_f.append(x[:, t + 1])
            else:
                x_t = xs_hat_t_1
                u_t = u[:, -1]*0
                x_f.append(x[:, -1])

            prior_params_beta_t = self.get_prior_params_beta(batch_size)
            prior_params_z_t = self.vae.get_prior_params(batch_size)

            # Obtain the visual encoding
            zs_t, z_params_t = self.vae.encode(x_t)
            # Obtain the betas - time-invariant properties and hidden
            beta_params_t, hidden_beta_t = self.beta_net.forward(zs_t, hidden_beta_t)
            beta_params_t = beta_params_t.view(batch_size, self.dim_beta, self.q_dist_beta.nparams)
            # sample the latent code beta
            betas_t = self.q_dist_beta.sample(params=beta_params_t)

            # Obtain the next step z's
            zs_t_1 = zs_t + self.transition_net.forward(
                torch.cat([zs_t, u_t, betas_t], dim=-1)) * self.dt  # transition predicts the angular velocity

            # Pass through the decoder
            xs_hat_t_1, x_hat_params_t_1 = self.vae.decode(zs_t_1)

            prior_params_beta_f.append(prior_params_beta_t)
            beta_params_f.append(beta_params_t)
            betas_f.append(betas_t)

            prior_params_z_f.append(prior_params_z_t)
            z_params_f.append(z_params_t)
            zs_f.append(zs_t)

            xs_hat_f.append(xs_hat_t_1)
            x_hat_params_f.append(x_hat_params_t_1)

        return prior_params_beta_f, beta_params_f, betas_f, prior_params_z_f, z_params_f, zs_f, xs_hat_f, x_hat_params_f, x_f

    def loss(self, x, u, H):
        batch_size, T, _ = u.shape
        prior_params_beta_f, beta_params_f, betas_f, prior_params_z_f, z_params_f, zs_f, xs_hat_f, x_hat_params_f, x_f = self.filter(
            x, u, H)
        '''
        prior_params_beta = torch.stack(prior_params_beta_f, dim=1).view(batch_size * (T - 1), self.dim_beta, 2)
        beta_params = torch.stack(beta_params_f, dim=1).view(batch_size * (T - 1), self.dim_beta, 2)
        betas = torch.stack(betas_f, dim=1).view(batch_size * (T - 1), self.dim_beta)
        prior_params_z = torch.stack(prior_params_z_f, dim=1).view(batch_size * (T - 1), self.dim_z, 2)
        z_params = torch.stack(z_params_f, dim=1).view(batch_size * (T - 1), self.dim_z, 2)  # TODO : generalize here
        zs = torch.stack(zs_f, dim=1).view(batch_size * (T - 1), self.dim_z)
        '''
        x_recon = torch.stack(xs_hat_f, dim=1)
        x_gt = torch.stack(x_f, dim=1)
        # x_recon_params = torch.stack(x_hat_params_f, dim=1).view(batch_size * (T - 1), 32, 32)
        # x_t_1 = torch.stack(x_f, dim=1).view(batch_size * (T - 1), 32, 32)
        '''
        logpx = self.vae.x_dist.log_density(x_t_1, params=x_recon_params).view(batch_size, -1).sum(1)
        logpz = self.vae.prior_dist.log_density(zs, params=prior_params_z).view(batch_size, -1).sum(1)
        logqz_condx = self.vae.q_dist.log_density(zs, params=z_params).view(batch_size, -1).sum(1)

        logpbeta = self.prior_dist_beta.log_density(betas, params=prior_params_beta).view(batch_size, -1).sum(1)
        logqbeta_condx = self.q_dist_beta.log_density(betas, params=beta_params).view(batch_size, -1).sum(1)

        elbo = logpx + (logpz - logqz_condx + logpbeta - logqbeta_condx) * 0.1
        '''
        return x_recon, x_gt


def format(x):
    resized_tensor = torch.nn.functional.interpolate(x[:, None, :, :], size=32, mode="bilinear", align_corners=False)
    img = torch.clip(resized_tensor[:, 0, :, :] * 255., 0, 255).to(torch.uint8)
    return img.view(-1, 32, 32).numpy()


def display_samples(x, x_recon, filename, T):
    frames = []
    gt = format(x)
    pred = format(x_recon[:, 0, :, :])
    for i in range(T-1):
        # img = gt[i] - pred[i]
        img = np.concatenate([gt[i], pred[i]], axis=0)
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
    parser.add_argument('-n', '--num-epochs', default=1, type=int, help='number of training epochs')
    parser.add_argument('-b', '--batch-size', default=1, type=int, help='batch size')
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
    parser.add_argument('--log_freq', default=1, type=int, help='num iterations per log')
    args = parser.parse_args()

    if args.use_cuda:
        torch.cuda.set_device(args.gpu)

    # data loader
    kwargs = {'num_workers': 4, 'pin_memory': args.use_cuda}
    # train_loader = DataLoader(dataset=dset.Pendulum(), batch_size=args.batch_size, shuffle=True, **kwargs)
    train_loader = DataLoader(dataset=dataset.Pendulum(), batch_size=args.batch_size, shuffle=True)  # for debug uncomment this version

    dbvf = DVBF(dim_z=3, dim_beta=5, use_cuda=args.use_cuda)
    load_dbvf = True
    if load_dbvf:
        # TODO: improve checkpoint loading
        checkpoint_path = 'results/pendulum/dbvf_train/train_24_02/checkpt-0000.pth'
        checkpt = torch.load(checkpoint_path)
        state_dict = checkpt['state_dict']
        dbvf.load_state_dict(state_dict, strict=False)
        print('DBVF loaded successfully')

    # training loop
    num_iterations = len(train_loader) * args.num_epochs
    iteration = 0
    # elbo_running_mean = utils.RunningAverageMeter()
    H = 20   # fraction of time given for system identification
    while iteration < num_iterations:
        for i, values in enumerate(train_loader):
            print(i)
            obs, state, parameter, action = values
            batch_time = time.time()
            dbvf.eval()
            # transfer to GPU
            if args.use_cuda:
                x = obs.cuda()
                u = action.cuda()
            else:
                x = obs
                u = action

            # do ELBO gradient and accumulate loss
            reconstruced, x_gt = dbvf.loss(x, u, H)
            display_samples(x_gt[0, :, :].cpu(), reconstruced[0, :, :].cpu(), f'dump/test_II/dvbf-{iteration}', H)

            iteration += 1


if __name__ == '__main__':
    model = main()
