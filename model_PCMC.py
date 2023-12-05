# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 10:57:19 2023

@author: simon
"""

from typing import Tuple, Dict
from ReconstructionVAE import VAE
import torch
import torch.nn as nn
import numpy as np


class DVBF(nn.Module):
    def __init__(self, dim_x: Tuple, dim_u: int, dim_z: int, dim_alpha: int, dim_beta: int, batch_size: int, hidden_transition: int = 128, hidden_size=128):
        super(DVBF, self).__init__()
        self.dim_x = np.prod(dim_x).item()
        self.dim_z = dim_z
        self.dim_alpha = dim_alpha
        self.dim_beta = dim_beta
        self.dim_u = dim_u
        self.batch_size = batch_size
        self.ci = 1.0
        
        
        # Dynamic Network, ENCODE TRANSITION (from previous LS and action)
        self.alpha_net = nn.Sequential(
            nn.Linear(in_features=dim_z+dim_u, out_features=hidden_transition),
            nn.ReLU(),
            nn.Linear(in_features=hidden_transition, out_features=dim_alpha),
            nn.ReLU(),
            nn.Softmax()
        )
        
        # Recognition Model, ENCODER
        self.beta_net = nn.Sequential(
            nn.Linear(in_features=self.dim_x, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=2*dim_beta),
        )
        
        # Transition Network, COMPUTE TRANSITION (from alpha (dynamic properties) and beta (time-invariant properties))
        self.transition_net = nn.Sequential(
            nn.Linear(in_features=dim_alpha+dim_beta, out_features=hidden_transition),
            nn.ReLU(),
            nn.Linear(in_features=hidden_transition, out_features=dim_z),
            nn.ReLU(),
            nn.Softmax()
        )
        
        # Reconstruction Model, DECODER
        self.observation_model = nn.Sequential(
            nn.Linear(in_features=dim_z, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=self.dim_x)
        )



    def generate_samples(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(logvar)
        return mu + eps * std

    def get_initial_samples(self, x: torch.Tensor) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
        beta_params = VAE(x[:,1])  #???????
        mu, logvar = torch.split(beta_params, split_size_or_sections=self.dim_beta, dim=1)
        alpha1 = torch.zeros((self.dim_alpha))
        beta1 = self.generate_samples(mu, logvar)
        z1 = self.encode_transition(alpha1, beta1)
        return mu, logvar, z1, beta1

    def encode_dynamic(self, z_t, u_t):
        alpha = self.alpha_net(torch.cat([z_t, u_t], dim=-1))
        return alpha

    def encode_transition(self, alpha, beta):
        z = self.transition_net(torch.cat([alpha, beta], dim=-1))
        return z

    def sample_beta(self, z_t, x_t=None):
        if x_t is not None:
            data = torch.cat([x_t], dim=1)
            beta_params = self.beta_net(data)
            mu_beta, logvar_beta = torch.split(beta_params, split_size_or_sections=self.dim_beta, dim=1)
        else:
            mu_beta = torch.zeros((self.batch_size, self.dim_beta)).to(z_t)
            logvar_beta = torch.ones((self.batch_size, self.dim_beta)).to(z_t)
        return mu_beta, logvar_beta

    def filter(self, x: torch.Tensor, u: torch.Tensor):
        num_obs = x.shape[1]
        N, T, _ = u.shape
        mu, logvar, z_t, beta_t = self.get_initial_samples(x)
        z = [z_t]
        beta = [beta_t]
        beta_means = [mu]
        beta_vars = [logvar]

        for t in range(1, T):
            u_t = u[:, t - 1]
            if t < num_obs:
                z_t, mu_beta_t, logvar_beta_t, beta_t = self.forward(z=z_t, u=u_t, x=x[:, t], return_q=True)
            else:
                z_t, mu_beta_t, logvar_beta_t, beta_t = self.forward(z=z_t, u=u_t, return_q=True)
            z.append(z_t)
            beta.append(beta_t)
            beta_means.append(mu_beta_t)
            beta_vars.append(logvar_beta_t)
        z = torch.stack(z, dim=1)
        beta_means = torch.stack(beta_means, dim=1)
        beta_vars = torch.stack(beta_vars, dim=1)
        beta = torch.stack(beta, dim=1)
        return z, dict(beta_means=beta_means, beta_vars=beta_vars, betas=beta)

    def forward(self, z: torch.Tensor, u: torch.Tensor, x: torch.Tensor = None, return_q=False):
        alpha = self.encode_dynamic(z, u)
        mu_beta, logvar_beta = self.sample_beta(z, x)
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
        x_rec_mean = self.observation_model(z).view(-1, self.dim_x)
        if return_dist:
            p_x = torch.distributions.MultivariateNormal(x_rec_mean, covariance_matrix=torch.eye(self.dim_x).to(x_rec_mean))
            return p_x, x_rec_mean
        else:
            return x_rec_mean

    def loss(self, x, u):
        z, info = self.filter(x, u)
        beta_means, beta_logvars, beta = info['beta_means'], info['beta_vars'], info['betas']
        x_hat = self.reconstruct(z)
        # p_x, x_hat = self.reconstruct(z, return_dist=True)
        # logprob_x = p_x.log_prob(x.view(-1, self.dim_x))

        # Reconstruction Error Loss
        rec_loss = torch.nn.functional.mse_loss(x_hat, x.view(-1, self.dim_x))
        beta_mean, beta_logvars = beta_means.view(-1, self.dim_beta), beta_logvars.view(-1, self.dim_beta)
        tilde_mean, tilde_logvars = torch.split(VAE(x), split_size_or_sections=self.dim_beta, dim=1)
        kl_loss = 0.5 * torch.mean(((beta_mean-tilde_mean).pow(2) + beta_logvars.exp())
                                   /tilde_logvars.exp() - beta_logvars + tilde_logvars - 1)

        loss = (rec_loss + kl_loss)*100

        return loss