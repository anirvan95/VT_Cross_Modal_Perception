import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import numpy as np

VAR_THRESHOLD = 1e-4
latent_walks = 15
win_samples = None
win_test_reco = None
win_latent_walk = None
win_train_elbo = None
win_correlation = None
win_space = None


def generate_gradient_colors(N):
    t = np.linspace(0, 1, N)
    red = np.interp(t, [0, 0.5, 1], [255, 0, 0])
    green = np.interp(t, [0, 0.5, 1], [0, 255, 0])
    blue = np.interp(t, [0, 0.5, 1], [0, 0, 255])

    colors = np.column_stack((red, green, blue))
    colors = colors.astype(int)

    return colors


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
    _, dim_z = zs_sample.size()
    xs = []
    with torch.no_grad():
        delta = torch.autograd.Variable(torch.linspace(-2, 2, latent_walks)).type_as(zs_sample) # variations in each dimension of latent space

    zs_delta = zs_sample.clone().view(1, 1, dim_z)
    for i in range(dim_z):
        vec = torch.zeros(dim_z).view(1, dim_z).expand(latent_walks, dim_z).contiguous().type_as(zs_sample)
        vec[:, i] = 1
        vec = vec * delta[:, None]
        # zs_delta[:, :, i] = 0
        zs_walk = zs_delta + vec[None]
        xs_walk = model.decoder.forward(zs_walk.view(-1, dim_z)).sigmoid()
        xs.append(xs_walk)

    xs = np.array(torch.cat(xs, 0).data.cpu())
    win_latent_walk = vis.images(xs, latent_walks, dim_z, opts={'caption': 'latent walk'}, win=win_latent_walk)


def plot_elbo(train_elbo, vis):
    global win_train_elbo
    win_train_elbo = vis.line(torch.Tensor(train_elbo), opts={'markers': True}, win=win_train_elbo)


def plot_latent_vae(vae, pendulum_dataset, vis):
    global win_correlation, win_space
    validation_loader = DataLoader(pendulum_dataset, batch_size=500, shuffle=False)
    vae.eval()
    # Extract test cases from the whole batch
    for i, values in enumerate(validation_loader):
        obs, action, state, parameter = values
        if i == 0:
            x = obs[0:1, :, :, :]
            s = state[0:1, :, :]
            p = parameter[0:1, :, :]
        else:
            x = torch.cat([x, obs[0:1, :, :, :]], dim=0)
            s = torch.cat([s, state[0:1, :, :]], dim=0)
            p = torch.cat([p, parameter[0:1, :, :]], dim=0)

    x = x.cuda()
    x = x.view(-1, 32, 32)
    states = s.view(-1, 2)
    parameters = p.view(-1, 3)

    xs, x_params, zs, z_params = vae.reconstruct_img(x)

    qz_means = z_params[:, :, 0]
    dim_z = qz_means.shape[1]

    colors_label = np.array([['*r', '*g', '*b', '*m', '*c', '*k'],
                             ['xr', 'xg', 'xb', 'xm', 'xc', 'xb'],
                             ['.r', '.g', '.b', '.m', '.c', '.b']])


    # Plotting trend (correlation) of the grount truth states with the learn latent space
    fig, ax = plt.subplots(2, dim_z)
    for j in range(dim_z):
        ax[0, j].plot(qz_means[:, j].cpu().detach(), states[:, 0], colors_label[0, j])

    for j in range(dim_z):
        ax[1, j].plot(qz_means[:, j].cpu().detach(), parameters[:, 0], colors_label[2, j])

    win_correlation = vis.matplot(plt, win=win_correlation)
    plt.close()

    # Plot the latent space with coloring according to the ground truth values of theta
    sorted_values, indices = torch.sort(states[:, 0])
    qz_means_sorted = qz_means[indices]
    var = torch.std(qz_means.contiguous(), dim=0).pow(2)
    var_sort, info_dim = torch.sort(var)
    qz_means_sorted = qz_means_sorted[:, info_dim[-3:]]

    colors = generate_gradient_colors(states.shape[0])

    win_space = vis.scatter(qz_means_sorted.cpu().detach(), opts={'caption': 'latent theta space', 'markercolor': colors, 'markersize': 2}, win=win_space)


def plot_latent_dvbf(dbvf, pendulum_dataset, vis):
    global win_static_correlation, win_dynamic_correlation, win_space_theta, win_space_atheta
    validation_loader = DataLoader(pendulum_dataset, batch_size=500, shuffle=False)
    dbvf.eval()
    batch_size = 250
    T = 15
    # Extract test cases from the whole batch
    for i, values in enumerate(validation_loader):
        obs, state, parameter, action = values
        if i == 0:
            x = obs[0:1, :, :, :]
            u = action[0:1, :, :]
            s = state[0:1, :, :]
            p = parameter[0:1, :, :]

        else:
            x = torch.cat([x, obs[0:1, :, :, :]], dim=0)
            u = torch.cat([u, action[0:1, :, :]], dim=0)
            s = torch.cat([s, state[0:1, :, :]], dim=0)
            p = torch.cat([p, parameter[0:1, :, :]], dim=0)

    x = x.cuda()
    u = u.cuda()

    prior_params_beta_f, beta_params_f, betas_f, prior_params_z_f, z_params_f, zs_f, xs_hat_f, x_hat_params_f, x_f, ang_vel_f = dbvf.filter(x, u, 1)

    # prior_params_beta = torch.stack(prior_params_beta_f, dim=1).view(batch_size * (T - 1), dbvf.dim_beta, 2)
    ang_vels = torch.stack(ang_vel_f, dim=1).view(batch_size * (T - 1), dbvf.dim_z)
    beta_params = torch.stack(beta_params_f, dim=1).view(batch_size * (T - 1), dbvf.dim_beta, 2)
    # betas = torch.stack(betas_f, dim=1).view(batch_size * (T - 1), dbvf.dim_beta)
    # prior_params_z = torch.stack(prior_params_z_f, dim=1).view(batch_size * (T - 1), dbvf.dim_z, 2)
    z_params = torch.stack(z_params_f, dim=1).view(batch_size * (T - 1), dbvf.dim_z, 2)  # TODO : generalize here
    # zs = torch.stack(zs_f, dim=1).view(batch_size * (T - 1), dbvf.dim_z)
    # x_recon = torch.stack(xs_hat_f, dim=1)
    # x_recon_params = torch.stack(x_hat_params_f, dim=1).view(batch_size * (T - 1), 32, 32)
    # x_t_1 = torch.stack(x_f, dim=1).view(batch_size * (T - 1), 32, 32)

    qz_means = z_params[:, :, 0]
    qbeta_means = beta_params[:, :, 0]
    Kz = qz_means.shape[1]
    Kbeta = qbeta_means.shape[1]

    colors_label = np.array([['*r', '*g', '*b', '*m', '*c', '*k'],
                             ['xr', 'xg', 'xb', 'xm', 'xc', 'xb'],
                             ['.r', '.g', '.b', '.m', '.c', '.b']])

    states = s[:, :-1, :].reshape(batch_size * (T - 1), 2)
    parameters = p[:, :-1, :].reshape(batch_size * (T-1), 3)

    fig, ax = plt.subplots(3, Kz)
    # Directly Observable Cues
    for j in range(Kz):
        ax[0, j].plot(qz_means[:, j].cpu().detach(), states[:, 0], colors_label[0, j])

    for j in range(Kz):
        ax[1, j].plot(ang_vels[:, j].cpu().detach(), states[:, 1], colors_label[1, j])

    for j in range(Kz):
        ax[2, j].plot(qz_means[:, j].cpu().detach(), parameters[:, 0], colors_label[2, j])


    win_static_correlation = vis.matplot(plt, win=win_static_correlation)
    plt.close()

    # Hidden Cues
    fig, ax = plt.subplots(2, Kbeta)
    for j in range(Kbeta):
        ax[0, j].plot(qbeta_means[:, j].cpu().detach(), parameters[:, 1], colors_label[0, j])

    for j in range(Kbeta):
        ax[1, j].plot(qbeta_means[:, j].cpu().detach(), parameters[:, 2], colors_label[1, j])

    win_dynamic_correlation = vis.matplot(plt, win=win_dynamic_correlation)
    plt.close()

    # Plot the latent space with coloring according to the ground truth values of theta
    sorted_values, indices = torch.sort(states[:, 0])
    qz_means_sorted = qz_means[indices]
    colors = generate_gradient_colors(states.shape[0])

    win_space_theta = vis.scatter(qz_means_sorted.cpu().detach(), opts={'caption': 'latent theta space', 'markercolor': colors, 'markersize': 2}, win=win_space_theta)

    # Plot the latent space with coloring according to the ground truth values of angular theta
    # Plot only top 3 space with high variance
    sorted_vals, ind_ang_theta = torch.sort(states[:, 1])
    ang_vels_sorted = ang_vels[ind_ang_theta]
    colors = generate_gradient_colors(states.shape[0])