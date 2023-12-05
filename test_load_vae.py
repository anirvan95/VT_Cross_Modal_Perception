import torch
from train_tcvae import VAE
import lib.dist as dist

checkpoint_path = 'test1/checkpt-0000.pth'
prior_dist = dist.Normal()
q_dist = dist.Normal()

vae = VAE(beta_dim=6, gamma_dim=6, use_cuda=False, prior_dist=prior_dist, q_dist=q_dist)

# Load the trained parameters

checkpt = torch.load(checkpoint_path)
args = checkpt['args']
state_dict = checkpt['state_dict']

vae.load_state_dict(state_dict, strict=False)

# Freeze the model
for param in vae.parameters():
    param.requires_grad = False


# Check freezing
def is_frozen(module):
    for param in module.parameters():
        if param.requires_grad:
            return False
    return True


print('Frozen status: ', is_frozen(vae))