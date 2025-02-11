# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 17:24:44 2024

@author: simon
"""

import imageio
import numpy as np
import torch
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader, Dataset
from models.pcmc_model import PCMC
import lib.dist as dist



#################  FUNCTIONS  #############################################################################################

#Funtion to lead the dataset
def load_data(file: str, batch_size=1, device='cpu') -> Dataset:
    x = torch.from_numpy(np.load(file)['obs']).to(device)
    x = x[:10000, :, :, :, :]  #????????????????????
    x = x.to(torch.float32) / 255
    x = x.view([10000, 15, 32*32])
    u = torch.from_numpy(np.load(file)['actions']).to(device)
    u = u[:10000, :, :]  #????????????????????
    u = u.to(torch.float32) / 1.5
    datasets = TensorDataset(x, u)

    train_loader = DataLoader(datasets['training_params'], batch_size=batch_size, shuffle=True)
    return train_loader

#Funtion to access the latent space of trained pcmc given a specific image and its previous latent space.
def latent_space(trained_model, z: torch.Tensor, u: torch.Tensor, x: torch.Tensor = None, return_q=False):
        alpha = trained_model.encode_dynamic(z, u)
        mu_beta, logvar_beta = trained_model.sample_beta(x)
        beta = trained_model.generate_samples(mu_beta, logvar_beta)
        z = trained_model.encode_transition(alpha, beta)
        if return_q:
            return z, mu_beta, logvar_beta, beta
        else:
            return z
 
##########################################################################################################

#parameter init
dim_z = 3
dim_x = (32, 32)
dim_u = 1
dim_a = 1
dim_beta = 15   # Fixed from before
dim_alpha = 3
batch_size = 300

device = 'cuda'
prior_dist = dist.Normal()
q_dist = dist.Normal()

           
if device == 'cuda':
    use_cuda = True
else:
    use_cuda = False

trained_pcmc = PCMC(dim_x=32*32, dim_u=dim_u, dim_z=dim_z, dim_alpha=dim_alpha, dim_beta=dim_beta, batch_size=batch_size, device=device).to(device)

'''
The idea here is to obtain the parameters of an already trained network but with - state_dict = checkpt['state_dict'] -
it gives me TypeError: 'PCMC' object is not subscriptable. This means that it doesn't work as the VAE. Needs to be fixed.
'''

checkpoint_path = "D:/CrossModalGit/VT_Cross_Modal_Perception-main/runs/pcmc/dvbf_exp_2.th"
checkpt = torch.load(checkpoint_path)
state_dict = checkpt['state_dict']
trained_pcmc.load_state_dict(state_dict, strict=False)


dataset_path = "D:/CrossModalGit/VT_Cross_Modal_Perception-main/datasets/pendulum/training_params.npz"
train_loader = load_data(file = dataset_path, batch_size = 1, device = device)

'''
My naive idea was to skim through all images in the dataset and append in a list both the betas and the zetas.
The problem is that, to do so, for each image I need to find the latent space associated with the previous image 
of the sequence. If I manage to do that, the idea is to put every latent dimension in a separate column of a 
dataframe and plot all combinations as a  function of the corresponding input dimension (E.g. angular velocity).
Hopefully, we are going to find some structure
'''


count = 0
return_q = True
for batch in train_loader:
    count += 1
    if count > 1000:
        break
    x, u = batch[0], batch[1]
    
    mu_beta_prior, logvar_beta_prior, z_t, betas_prior = self.get_initial_samples(x)
    zetas = [z_t]
    betas = [betas_prior]
    beta_means = [mu_beta_prior]
    beta_vars = [logvar_beta_prior]
    
    if return_q:
        z, mu_beta, logvar_beta, beta = latent_space(trained_pcmc, z = z, u = u, x = x, return_q=return_q)
    else:
        z = latent_space(trained_pcmc, z = z, u = u, x = x, return_q=return_q)
    
