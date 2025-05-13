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


class MultiModalLF(nn.Module):
    def __init__(self, args):
        super(MultiModalLF, self).__init__()
        self.dim_a = action_dim
        self.horizon = horizon

        # ############################################### Vision #######################################
        self.vis_dim_z = args.vis_dim_z
        self.vis_x_std = args.vis_stdx

        # Recognition Model  - Visual Encoder & Decoder
        self.vis_vae = VAE(dim_z=self.vis_dim_z, std=self.vis_x_std, modality='vision', use_cuda=args.use_cuda)


        # ############################################### Tactile #######################################
        self.tac_dim_z = args.tac_dim_z
        self.tac_x_std = args.tac_stdx

        # Recognition Model  - Visual Encoder & Decoder
        self.tac_vae = VAE(dim_z=self.tac_dim_z, std=self.tac_x_std, modality='tactile', use_cuda=args.use_cuda)


        self.y_std = args.stdy
        self.dim_y = args.dim_y
        self.dim_h = args.dim_h

        # distribution family of p(y)
        self.prior_dist_y = dist.Normal()
        self.q_dist_y = dist.Normal()

        # hyperparameters for prior p(z) - Hierarchical prior
        # self.register_buffer('prior_params_y', torch.zeros(self.dim_y, 2))
        self.embedding = nn.Embedding(num_embeddings=args.num_objects, embedding_dim=self.dim_y)

        # Recognition Model - Hidden States LSTM Network to identify hidden properties from changing state space z
        self.y_net = LSTMEncode(input_size=self.tac_dim_z + self.vis_dim_z + self.dim_a, hidden_size=self.dim_h,
                                    output_size=self.dim_y * self.q_dist_y.nparams)

        # Dynamic Network, computes next state (from previous visual state z, inferred y (time-invariant properties) and action)
        # Combines the visual and tactile, predicts visual and tactile space
        self.transition_net = MLPTransitionV2(input_dim=self.vis_dim_z + self.dim_y + self.tac_dim_z + + self.dim_a,
                                            output_dim=self.vis_dim_z * self.vis_vae.q_dist.nparams + self.tac_dim_z * self.tac_vae.q_dist.nparams)

        if args.use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()

    def get_prior_params_y(self, labels):
        device = labels.device
        object_label = ((labels[:, 0, 0]-1)*15 + (labels[:, 0, 1]-1)*5 + (labels[:, 0, 2]-1)).long()
        prior_params_mu = torch.tanh(self.embedding(object_label))
        prior_params_sigma = torch.ones(labels.size(0), self.dim_y, 1, device=device)*np.log(self.y_std)
        prior_params_y = torch.cat((prior_params_mu[:, :, None], prior_params_sigma), dim=-1)

        return prior_params_y


    def filter(self, vis_x, tac_x, u, labels):
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

        # ############################################### Vision #######################################
        vis_prior_params_z_f = []
        vis_z_params_f = []
        vis_zs_f = []
        vis_x_hat_params_f = []
        vis_xs_hat_f = []
        vis_x_f = []

        # ############################################### Tactile #######################################
        tac_prior_params_z_f = []
        tac_z_params_f = []
        tac_zs_f = []
        tac_x_hat_params_f = []
        tac_xs_hat_f = []
        tac_x_f = []

        # Convention
        # t = previous time step
        # t_1 = current time step

        # First time step is handled differently
        # Obtain the measurement latent space
        # ############################################### Vision #######################################
        vis_zs_meas_t, vis_z_meas_params_t = self.vis_vae.encode(vis_x[:, 0])
        vis_prior_params_z_t = self.vis_vae.get_prior_params(batch_size)  # only for 1st time step

        vis_zs_t = vis_zs_meas_t
        vis_z_params_t = vis_z_meas_params_t

        vis_xs_hat_t, vis_x_hat_params_t = self.vis_vae.decode(vis_zs_t)

        vis_prior_params_z_f.append(vis_prior_params_z_t)
        vis_z_params_f.append(vis_z_params_t)
        vis_zs_f.append(vis_zs_t)
        vis_xs_hat_f.append(vis_xs_hat_t)
        vis_x_hat_params_f.append(vis_x_hat_params_t)
        vis_x_f.append(vis_x[:, 0])

        # ############################################### Tactile #######################################
        tac_zs_meas_t, tac_z_meas_params_t = self.tac_vae.encode(tac_x[:, 0])
        tac_prior_params_z_t = self.tac_vae.get_prior_params(batch_size)  # only for 1st time step

        tac_zs_t = tac_zs_meas_t
        tac_z_params_t = tac_z_meas_params_t

        tac_xs_hat_t, tac_x_hat_params_t = self.tac_vae.decode(tac_zs_t)

        tac_prior_params_z_f.append(tac_prior_params_z_t)
        tac_z_params_f.append(tac_z_params_t)
        tac_zs_f.append(tac_zs_t)
        tac_xs_hat_f.append(tac_xs_hat_t)
        tac_x_hat_params_f.append(tac_x_hat_params_t)
        tac_x_f.append(tac_x[:, 0])

        # Transfer the prior params y_t
        prior_params_y_t_1 = self.get_prior_params_y(labels)

        # Filtering Step
        for t in range(1, T):
            u_t_1 = u[:, t]
            u_t = u[:, t - 1]

            vis_x_t_1 = vis_x[:, t]
            tac_x_t_1 = tac_x[:, t]

            y_params_t_1, hidden_y_t_1 = self.y_net.forward(torch.cat([vis_zs_t, tac_zs_t, u_t],dim=-1), hidden_y_t)
            y_params_t_1 = y_params_t_1.view(batch_size, self.dim_y, self.q_dist_y.nparams)

            # Sample the visual latent code y
            ys_t_1 = self.q_dist_y.sample(params=y_params_t_1)

            # Obtain the next step visual and tactile z's via common dynamics function
            fused_trans_params_t_1 = self.transition_net.forward(torch.cat([vis_zs_t, tac_zs_t, ys_t_1, u_t_1], dim=-1))
            fused_trans_params_t_1 = fused_trans_params_t_1.view(batch_size, self.vis_dim_z*self.vis_vae.q_dist.nparams+self.tac_dim_z*self.tac_vae.q_dist.nparams)
            vis_trans_params_t_1 = fused_trans_params_t_1[:, :self.vis_dim_z*self.vis_vae.q_dist.nparams]
            vis_trans_params_t_1 = vis_trans_params_t_1.view(batch_size, self.vis_dim_z, self.vis_vae.q_dist.nparams)
            tac_trans_params_t_1 = fused_trans_params_t_1[:, self.vis_dim_z*self.vis_vae.q_dist.nparams:]
            tac_trans_params_t_1 = tac_trans_params_t_1.view(batch_size, self.tac_dim_z, self.tac_vae.q_dist.nparams)

            vis_zs_meas_t_1, vis_z_meas_params_t_1 = self.vis_vae.encode(vis_x_t_1)
            vis_z_params_t_1 = bayes_fusion(vis_trans_params_t_1, vis_z_meas_params_t_1)
            # Pass through the visual decoder
            vis_zs_t_1 = self.vis_vae.q_dist.sample(params=vis_z_params_t_1)
            vis_xs_hat_t_1, vis_x_hat_params_t_1 = self.vis_vae.decode(vis_zs_t_1)

            tac_zs_meas_t_1, tac_z_meas_params_t_1 = self.tac_vae.encode(tac_x_t_1)
            tac_z_params_t_1 = bayes_fusion(tac_trans_params_t_1, tac_z_meas_params_t_1)
            # Pass through the tactile decoder
            tac_zs_t_1 = self.tac_vae.q_dist.sample(params=tac_z_params_t_1)
            tac_xs_hat_t_1, tac_x_hat_params_t_1 = self.tac_vae.decode(tac_zs_t_1)

            # Save the parameters
            # ############################################### Vision #######################################
            vis_prior_params_z_f.append(vis_prior_params_z_t)
            vis_z_params_f.append(vis_z_params_t_1)
            vis_zs_f.append(vis_zs_t_1)

            vis_xs_hat_f.append(vis_xs_hat_t_1)
            vis_x_hat_params_f.append(vis_x_hat_params_t_1)
            vis_x_f.append(vis_x_t_1)

            vis_zs_t = vis_zs_t_1
            vis_prior_params_z_t = vis_z_params_t_1

            # ############################################### Tactile #######################################
            tac_prior_params_z_f.append(tac_prior_params_z_t)
            tac_z_params_f.append(tac_z_params_t_1)
            tac_zs_f.append(tac_zs_t_1)

            tac_xs_hat_f.append(tac_xs_hat_t_1)
            tac_x_hat_params_f.append(tac_x_hat_params_t_1)
            tac_x_f.append(tac_x_t_1)

            tac_zs_t = tac_zs_t_1
            tac_prior_params_z_t = tac_z_params_t_1

            h, c = hidden_y_t
            hs_f.append(h)
            cs_f.append(c)

            prior_params_y_f.append(prior_params_y_t_1)
            y_params_f.append(y_params_t_1)
            ys_f.append(ys_t_1)

            hidden_y_t = hidden_y_t_1

        return (prior_params_y_f, y_params_f, ys_f, hs_f, cs_f,
                vis_prior_params_z_f, vis_z_params_f, vis_zs_f, vis_xs_hat_f, vis_x_hat_params_f, vis_x_f,
                tac_prior_params_z_f, tac_z_params_f, tac_zs_f, tac_xs_hat_f, tac_x_hat_params_f, tac_x_f)

