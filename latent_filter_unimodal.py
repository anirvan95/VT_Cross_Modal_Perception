import numpy as np
import random

import torch
import torch.nn as nn
from torch.autograd import Variable

import utils.dist as dist
from utils.functions import bayes_fusion

# Seeding for reproducibility
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Dataset specific parameters
vis_obs_dim = [128, 128, 2]
tac_obs_dim = [80, 80, 1]
action_dim = 9
horizon = 99

from networks import *


class UniModalLF(nn.Module):
    def __init__(self, args):
        super(UniModalLF, self).__init__()
        self.dim_z = args.dim_z
        self.dim_y = args.dim_y
        self.dim_h = args.dim_h
        self.dim_a = action_dim

        self.x_std = args.stdx
        self.y_std = args.stdy

        self.modality = args.modality
        self.horizon = horizon

        # distribution family of p(y)
        self.prior_dist_y = dist.Normal()
        self.q_dist_y = dist.Normal()
        # hyperparameters for prior p(z) - Hierarchical prior or Normal prior
        # self.register_buffer('prior_params_y', torch.zeros(self.dim_y, 2))
        self.embedding = nn.Embedding(num_embeddings=args.num_objects, embedding_dim=self.dim_y)
        # Recognition Model  - Visual Encoder & Decoder
        self.vae = VAE(dim_z=self.dim_z, std=self.x_std, modality=self.modality, use_cuda=args.use_cuda)

        # Recognition Model - Hidden States LSTM Network to identify hidden properties from changing state space z
        self.y_net = LSTMEncode(input_size=self.dim_z + self.dim_a, hidden_size=self.dim_h, output_size=self.dim_y * self.q_dist_y.nparams)

        # Transition Network, computes TRANSITION (from previous visual state z, inferred y (time-invariant properties) and action)
        self.transition_net = MLPTransition(input_dim=self.dim_z + self.dim_y + self.dim_a, output_dim=self.dim_z * self.vae.q_dist.nparams)

        if args.use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()

    def get_prior_params_y(self, labels):
        object_label = labels[:, 0, 0] + labels[:, 0, 1] + labels[:, 0, 2] # possible to add the interaction as well to further disentangle
        object_label = object_label.to(torch.device('cuda')).long() # TODO: Fix with device selection
        prior_params_mu = torch.tanh(self.embedding(object_label))
        prior_params_sigma = torch.ones(labels.size(0), self.dim_y, 1)*np.log(self.y_std)
        prior_params_sigma = prior_params_sigma.to('cuda')
        prior_params_y = torch.cat((prior_params_mu[:, :, None], prior_params_sigma), dim=-1)

        return prior_params_y

    def filter(self, x, u, labels, H):
        batch_size, T, _ = u.shape

        # Set the hidden layer of the LSTM to zero
        hidden_y_t = (torch.zeros(batch_size, self.dim_h, device='cuda'),
                      torch.zeros(batch_size, self.dim_h, device='cuda'))

        # Define the variables to store the filtering output
        prior_params_y_f = []
        y_params_f = []
        ys_f = []
        hs_f = []
        cs_f = []
        prior_params_z_f = []
        z_params_f = []
        zs_f = []
        x_hat_params_f = []
        xs_hat_f = []
        x_f = []

        # Convention
        # t = previous time step
        # t_1 = current time step

        # First time step is handled differently
        # Obtain the measurement latent space
        zs_meas_t, z_meas_params_t = self.vae.encode(x[:, 0])
        prior_params_z_t = self.vae.get_prior_params(batch_size)  # only for 1st time step

        zs_t = zs_meas_t
        z_params_t = z_meas_params_t

        xs_hat_t, x_hat_params_t = self.vae.decode(zs_t)

        prior_params_z_f.append(prior_params_z_t)
        z_params_f.append(z_params_t)
        zs_f.append(zs_t)
        xs_hat_f.append(xs_hat_t)
        x_hat_params_f.append(x_hat_params_t)
        x_f.append(x[:, 0])

        # Transfer the prior params z_t
        prior_params_y_t_1 = self.get_prior_params_y(labels)

        for t in range(1, T):  # Note the time index starts from 0 due to the correct control index
            u_t_1 = u[:, t]
            u_t = u[:, t - 1]
            x_t_1 = x[:, t]

            y_params_t_1, hidden_y_t_1 = self.y_net.forward(torch.cat([zs_t, u_t], dim=-1), hidden_y_t)
            y_params_t_1 = y_params_t_1.view(batch_size, self.dim_y, self.q_dist_y.nparams)

            # Sample the latent code y
            ys_t_1 = self.q_dist_y.sample(params=y_params_t_1)

            # Obtain the next step z's
            z_trans_params_t_1 = self.transition_net.forward(torch.cat([zs_t, u_t_1, ys_t_1], dim=-1))
            z_trans_params_t_1 = z_trans_params_t_1.view(batch_size, self.dim_z, self.vae.q_dist.nparams)

            zs_meas_t_1, z_meas_params_t_1 = self.vae.encode(x_t_1)
            # z_params_t_1 = z_trans_params_t_1 # without Bayes Integration
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

            h, c = hidden_y_t
            hs_f.append(h)
            cs_f.append(c)

            hidden_y_t = hidden_y_t_1
            zs_t = zs_t_1
            prior_params_z_t = z_params_t_1

        return prior_params_y_f, y_params_f, ys_f, prior_params_z_f, z_params_f, zs_f, xs_hat_f, x_hat_params_f, x_f, hs_f, cs_f