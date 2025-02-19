import torch
from train_tcvae import VAE
import lib.dist as dist
from torch.utils.data import DataLoader
import lib.dist as dist
import lib.utils as utils
import lib.datasets as dset
import matplotlib.pyplot as plt


checkpoint_path = 'runs/pendulum_exp_2_vae/checkpt-0000.pth'
prior_dist = dist.Normal()
q_dist = dist.Normal()

vae = VAE(beta_dim=15, gamma_dim=15, use_cuda=False, prior_dist=prior_dist, q_dist=q_dist)

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

# Load some samples and check reconstruction
train_set = dset.Pendulum()
train_loader = DataLoader(dataset=train_set, batch_size=2, shuffle=True)
plt.ion()

fig, ax = plt.subplots(1, 2)

for i, x in enumerate(train_loader):
    x = x.to('cpu')
    _, reco_params, _, _ = vae.reconstruct_img(x)
    reco_imgs = reco_params.sigmoid()
    # test_reco_imgs = torch.cat([reco_imgs.view(1, -1, 16, 16), reco_imgs.view(1, -1, 16, 16)], 0).transpose(0, 1)

    ax[0].imshow(x[0, 0, :, :])
    ax[1].imshow(reco_imgs[0, 0, :, :])
    plt.pause(1)
    ax[0].cla()
    ax[1].cla()