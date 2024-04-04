# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 10:57:19 2023

@author: simon
"""
from typing import Tuple, Dict
from train_tcvae import VAE
import torch
import torch.nn as nn
import numpy as np
import lib.dist as dist


class ConvEncoder(nn.Module):
    def __init__(self, output_dim):
        super(ConvEncoder, self).__init__()
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(1, 64, 4, 2, 1)  # 8 x 8
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 4, 2, 1)  # 4 x 4
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 512, 4)  # 512, 1
        self.bn3 = nn.BatchNorm2d(512)
        self.conv_z = nn.Conv2d(512, output_dim, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(-1, 1, 16, 16)
        # h = self.act(self.bn1(self.conv1(h)))
        # h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
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


class DVBF(nn.Module):
    def __init__(self, dim_x: Tuple, dim_u: int, dim_z: int, dim_alpha: int, dim_beta: int, batch_size: int, hidden_transition: int = 128, hidden_size=128):
        super(DVBF, self).__init__()
        self.dim_x = np.prod(dim_x).item()
        self.dim_z = dim_z
        self.dim_alpha = dim_alpha
        self.dim_beta = 15    # Fixed due to prior training of the VAE
        self.dim_u = dim_u
        self.batch_size = batch_size
        self.ci = 1.0

        # Dynamic Network, ENCODE TRANSITION (from previous LS and action)
        self.alpha_net = nn.Sequential(
            nn.Linear(in_features=dim_z+dim_u, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=dim_alpha),
            nn.ReLU(),
            nn.Softmax()
        )
        # Add the VAE model here, load the params and freeze it

        prior_dist = dist.Normal()
        q_dist = dist.Normal()

        self.vae = VAE(beta_dim=15, gamma_dim=15, use_cuda=False, prior_dist=prior_dist, q_dist=q_dist)
        checkpoint_path = 'pendulum_exp_2_vae/checkpt-0000.pth'
        checkpt = torch.load(checkpoint_path)
        state_dict = checkpt['state_dict']
        self.vae.load_state_dict(state_dict, strict=False)

        # Recognition Model, ENCODER
        self.beta_net = ConvEncoder(self.dim_beta * 2)
        
        # Transition Network, COMPUTE TRANSITION (from alpha (dynamic properties) and beta (time-invariant properties))
        self.transition_net = nn.Sequential(
            nn.Linear(in_features=dim_alpha+dim_beta, out_features=hidden_transition),
            nn.ReLU(),
            nn.Linear(in_features=hidden_transition, out_features=dim_z),
            nn.ReLU(),
            nn.Softmax()
        )
        
        # Reconstruction Model, DECODER
        self.observation_model = ConvDecoder(self.dim_z)

    def generate_samples(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(logvar)
        return mu + eps * std

    def get_initial_samples(self, x):
        x_initial = x[:, 0, :]
        # print(x_initial.shape)
        x_initial = x_initial.view(x.shape[0], 1, 16, 16)
        betas_prior, beta_params_prior = self.vae.encode(x_initial)
        mu_beta_prior = beta_params_prior[:, :, 0]
        logvar_beta_prior = beta_params_prior[:, :, 1]
        alphas = torch.zeros((x.shape[0], self.dim_alpha)).to('cpu')
        betas_prior = self.generate_samples(mu_beta_prior, logvar_beta_prior)
        z_t = self.encode_transition(alphas, betas_prior)
        return mu_beta_prior, logvar_beta_prior, z_t, betas_prior

    def encode_dynamic(self, z_t, u_t):
        alpha = self.alpha_net(torch.cat([z_t, u_t], dim=-1))
        return alpha

    def encode_transition(self, alpha, beta):
        z = self.transition_net(torch.cat([alpha, beta], dim=-1))
        return z

    def sample_beta(self, x_t=None):
        if x_t is not None:
            # data = torch.cat([x_t], dim=1)
            beta_params = self.beta_net(x_t)
            mu_beta, logvar_beta = torch.split(beta_params, split_size_or_sections=self.dim_beta, dim=1)
        else:
            mu_beta = torch.zeros((self.batch_size, self.dim_beta)).to('cpu')
            logvar_beta = torch.ones((self.batch_size, self.dim_beta)).to('cpu')
        return mu_beta, logvar_beta

    def filter(self, x: torch.Tensor, u: torch.Tensor):
        num_obs = x.shape[1]
        N, T, _ = u.shape
        mu_beta_prior, logvar_beta_prior, z_t, betas_prior = self.get_initial_samples(x)
        z = [z_t]
        betas = [betas_prior]
        beta_means = [mu_beta_prior]
        beta_vars = [logvar_beta_prior]
        betas_prior = [betas_prior]
        betas_prior_means = [mu_beta_prior]
        beta_prior_vars = [logvar_beta_prior]

        for t in range(1, T):
            u_t = u[:, t - 1]
            if t < num_obs:
                z_t, mu_beta_t, logvar_beta_t, beta_t = self.forward(z=z_t, u=u_t, x=x[:, t], return_q=True)
                betas_priot_t, beta_prior_params_t = self.vae.encode(x[:, t])
                # Pass the observation to prior beta distribution
                mu_beta_prior_t = beta_prior_params_t[:, :, 0]
                logvar_beta_prior_t = beta_prior_params_t[:, :, 1]
            else:
                z_t, mu_beta_t, logvar_beta_t, beta_t = self.forward(z=z_t, u=u_t, return_q=True)
                betas_priot_t, beta_prior_params_t = self.vae.encode(x[:, t])
                mu_beta_prior_t = beta_prior_params_t[:, :, 0]
                logvar_beta_prior_t = beta_prior_params_t[:, :, 1]

            z.append(z_t)
            betas.append(beta_t)
            beta_means.append(mu_beta_t)
            beta_vars.append(logvar_beta_t)

            betas_prior.append(betas_priot_t)
            betas_prior_means.append(mu_beta_prior_t)
            beta_prior_vars.append(logvar_beta_prior_t)

        z = torch.stack(z, dim=1)
        beta_means = torch.stack(beta_means, dim=1)
        beta_vars = torch.stack(beta_vars, dim=1)
        betas = torch.stack(betas, dim=1)

        beta_prior_means = torch.stack(betas_prior_means, dim=1)
        beta_prior_vars = torch.stack(beta_prior_vars, dim=1)
        betas_prior = torch.stack(betas_prior, dim=1)

        return z, dict(beta_means=beta_means, beta_vars=beta_vars, betas=betas, beta_prior_means=beta_prior_means, beta_prior_vars=beta_prior_vars, betas_prior=betas_prior)

    def forward(self, z: torch.Tensor, u: torch.Tensor, x: torch.Tensor = None, return_q=False):
        alpha = self.encode_dynamic(z, u)
        mu_beta, logvar_beta = self.sample_beta(x)
        beta = self.generate_samples(mu_beta, logvar_beta)
        z = self.encode_transition(alpha, beta)
        if return_q:
            return z, mu_beta, logvar_beta, beta
        else:
            return z

    '''
    def reconstruct(self, z: torch.Tensor):
        x_rec_mean = self.observation_model(z).view(-1, self.dim_x)
        return x_rec_mean
    '''

    def reconstruct(self, z: torch.Tensor, return_dist=False):
        x_rec_mean = self.observation_model(z.view(-1, self.dim_z))
        x_rec_mean = x_rec_mean.view(x_rec_mean.shape[0], -1)
        if return_dist:
            p_x = torch.distributions.MultivariateNormal(x_rec_mean, covariance_matrix=torch.eye(self.dim_x).to(x_rec_mean))
            return p_x, x_rec_mean
        else:
            return x_rec_mean

    def loss(self, x, u):
        z, info = self.filter(x, u)
        beta_means, beta_logvars, betas, beta_prior_means, beta_prior_logvars, betas_prior = info['beta_means'], info['beta_vars'], info['betas'], info['beta_prior_means'], info['beta_prior_vars'], info['betas_prior']
        x_hat = self.reconstruct(z)
        # p_x, x_hat = self.reconstruct(z, return_dist=True)
        # logprob_x = p_x.log_prob(x.view(-1, self.dim_x))

        # Reconstruction Error Loss
        rec_loss = torch.nn.functional.mse_loss(x_hat, x.view(-1, self.dim_x))
        beta_mean, beta_logvars = beta_means.view(-1, self.dim_beta), beta_logvars.view(-1, self.dim_beta)
        beta_prior_means, beta_prior_logvars = beta_prior_means.view(-1, self.dim_beta), beta_prior_logvars.view(-1, self.dim_beta)
        kl_loss = 0.5 * torch.mean(((beta_mean-beta_prior_means).pow(2) + beta_logvars.exp())/beta_prior_logvars.exp() - beta_logvars + beta_prior_logvars - 1)

        loss = (rec_loss + kl_loss)*100

        return loss