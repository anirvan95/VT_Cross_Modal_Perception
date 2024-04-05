import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import brewer2mpl
bmap = brewer2mpl.get_map('Set1', 'qualitative', 3)
colors = bmap.mpl_colors
import numpy as np
plt.style.use('ggplot')

VAR_THRESHOLD = 1e-4
latent_walks = 15
win_samples = None
win_test_reco = None
win_latent_walk = None
win_train_elbo = None


def display_samples(model, x, vis):
    global win_samples, win_test_reco, win_latent_walk

    # plot random samples
    sample_mu = model.model_sample(batch_size=100).sigmoid()
    images = np.array(sample_mu.view(-1, 1, 32, 32).data.cpu())
    win_samples = vis.images(images, 10, 2, opts={'caption': 'samples'}, win=win_samples)

    # plot the reconstructed distribution for the first 50 test images
    test_imgs = x[:50, :]
    _, reco_imgs, zs, _ = model.reconstruct_img(test_imgs)
    reco_imgs = reco_imgs.sigmoid()
    test_reco_imgs = torch.cat([test_imgs.view(1, -1, 32, 32), reco_imgs.view(1, -1, 32, 32)], 0).transpose(0, 1)
    win_test_reco = vis.images(np.array(test_reco_imgs.contiguous().view(-1, 1, 32, 32).data.cpu()), 10, 2, opts={'caption': 'test reconstruction image'}, win=win_test_reco)

    # plot latent walks
    zs_sample = zs[0:1, :]
    _, z_dim = zs_sample.size()
    xs = []
    with torch.no_grad():
        delta = torch.autograd.Variable(torch.linspace(-2, 2, latent_walks)).type_as(zs_sample) # variations in each dimension of latent space

    zs_delta = zs_sample.clone().view(1, 1, z_dim)
    for i in range(z_dim):
        vec = torch.zeros(z_dim).view(1, z_dim).expand(latent_walks, z_dim).contiguous().type_as(zs_sample)
        vec[:, i] = 1
        vec = vec * delta[:, None]
        # zs_delta[:, :, i] = 0
        zs_walk = zs_delta + vec[None]
        xs_walk = model.decoder.forward(zs_walk.view(-1, z_dim)).sigmoid()
        xs.append(xs_walk)

    xs = np.array(torch.cat(xs, 0).data.cpu())
    win_latent_walk = vis.images(xs, latent_walks, z_dim, opts={'caption': 'latent walk'}, win=win_latent_walk)


def plot_elbo(train_elbo, vis):
    global win_train_elbo
    win_train_elbo = vis.line(torch.Tensor(train_elbo), opts={'markers': True}, win=win_train_elbo)


def plot_latent_vs_gt_pendulum(vae, pendulum_dataset, save, z_inds=None):
    validation_loader = DataLoader(pendulum_dataset, batch_size=7000, shuffle=False)

    N = len(validation_loader.dataset)  # number of data samples
    K = vae.z_dim                    # number of latent variables
    nparams = vae.q_dist.nparams
    vae.eval()

    # print('Computing q(z|x) distributions.')
    qz_params = torch.Tensor(N, K, nparams)
    states = torch.Tensor(N, 2)
    parameters = torch.Tensor(N, 3)

    n = 0
    for obs, state, parameter in validation_loader:
        batch_size = obs.size(0)
        xs = obs.cuda()
        with torch.no_grad():
            xs = xs.view(batch_size, 1, 32, 32)
        qz_params[n:n + batch_size] = vae.encoder.forward(xs).view(batch_size, vae.z_dim, nparams).data
        states[n: n+batch_size] = state[:, 0, :]
        parameters[n: n+batch_size] = parameter[:, 0, :]
        n += batch_size

    # z_j is inactive if Var_x(E[z_j|x]) < eps.
    qz_means = qz_params[:, :, 0]
    '''
    var = torch.std(qz_means.contiguous().view(N, K), dim=0).pow(2)
    active_units = torch.arange(0, K)[var > VAR_THRESHOLD].long()
    print('Active units: ' + ','.join(map(str, active_units.tolist())))
    n_active = len(active_units)
    print('Number of active units: {}/{}'.format(n_active, vae.z_dim))
    z_inds = active_units
    '''
    # subplots where subplot[i, j] is gt_i vs. z_j
    # Plotting only visual cues - length, and theta
    fig, ax = plt.subplots(2, K)
    colors = np.array([['*r', '*g', '*b'],
              ['xm', 'xc', 'xb']])
    for j in range(K):
        ratio_theta = qz_means[:, j]/(states[:, 0]+10)
        ratio_theta = ratio_theta.numpy()
        ax[0, j].plot(ratio_theta, colors[0, j])

    for j in range(K):
        ratio_length = qz_means[:, j]/parameters[:, 0]
        ratio_length = ratio_length.numpy()
        ax[1, j].plot(ratio_length, colors[1, j])

    plt.savefig(save)
    plt.close()

