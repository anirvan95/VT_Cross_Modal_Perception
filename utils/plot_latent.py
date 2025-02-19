import os
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
import torch
import numpy as np
import imageio
import umap

VAR_THRESHOLD = 1e-4
latent_walks = 15
horizon = 26  # 1 Hz
action_dim = 2 # pose + param
obs_dim = [30, 30, 2] # Spatial and Vibration

win_samples = None
win_test_reco = None
win_latent_walk = None
win_train_elbo = None
win_correlation = None
win_space = None
win_static_correlation = None
win_dynamic_correlation = None
win_space_z = None
win_space_y = None
win_embed_space_z = None
win_embed_space_y = None
win_space_h = None
win_embed_space_h = None

win_label = None
win_test_reco = None
loss = torch.nn.MSELoss()

layout_def = {
    'scene': {
        'xaxis': {'range': [-0.1, 0.1]},
        'yaxis': {'range': [-0.1, 0.1]},
        'zaxis': {'range': [0, 0.1]},
    }
}

def generate_gradient_colors(N):
    t = np.linspace(0, 1, N)
    red = np.interp(t, [0, 0.5, 1], [255, 0, 0])
    green = np.interp(t, [0, 0.5, 1], [0, 255, 0])
    blue = np.interp(t, [0, 0.5, 1], [0, 0, 255])

    colors = np.column_stack((red, green, blue))
    colors = colors.astype(int)

    return colors

def compute_distances(features, labels):
    unique_classes = np.unique(labels)

    # Initialize the distance matrix
    distance_matrix = np.zeros((len(unique_classes), len(unique_classes)))

    # Compute intra-class distances
    for i, cls in enumerate(unique_classes):
        class_indices = np.where(labels == cls)[0]
        class_features = features[class_indices]

        if len(class_features) > 1:  # Ensure there are at least 2 samples
            distances = pdist(class_features)  # Pairwise distances
            intra_class_distance = np.mean(distances)  # Average intra-class distance
            distance_matrix[i, i] = intra_class_distance

    # Compute inter-class distances
    for i in range(len(unique_classes)):
        for j in range(i + 1, len(unique_classes)):
            class_i_features = features[labels == unique_classes[i]]
            class_j_features = features[labels == unique_classes[j]]

            # Compute pairwise distances between the two classes
            distances = pdist(np.vstack([class_i_features, class_j_features]))
            inter_class_distance = np.mean(distances)  # Average inter-class distance
            distance_matrix[i, j] = inter_class_distance
            distance_matrix[j, i] = inter_class_distance  # Symmetric matrix

    return distance_matrix, unique_classes

def extract_test_cases(test_loader):
    xf = []
    af = []
    labelf = []
    first = True
    # Extract test cases from the whole batch
    for i, values in enumerate(test_loader):
        obs, action, label = values
        if first:
            xf = obs[0:1, :, :, :, :]
            af = action[0:1, :, :]
            labelf = label[0:1, :, :]
            first = False
        else:
            xf = torch.cat([xf, obs[0:1, :, :, :, :]], dim=0)
            af = torch.cat([af, action[0:1, :, :]], dim=0)
            labelf = torch.cat([labelf, label[0:1, :, :]], dim=0)

    batch_size, T, _ = af.shape
    x = xf
    color_vals = []
    label_ind = []
    cmap1 = plt.get_cmap('tab20b')
    cmap2 = plt.get_cmap('tab20c')
    for i in range(labelf.shape[0]):
        object_label = labelf[i, 0, 0]
        col = cmap1(object_label / 27)
        col_rgb = np.array([col[0], col[1], col[2]]) * 255
        for j in range(0, horizon):
            label_ind.append(object_label)
            col_lighter = col_rgb
            color_vals.append(np.array([col_lighter[0], col_lighter[1], col_lighter[2]]))

    label_ind = np.array(label_ind)
    indices = np.arange(0, label_ind.shape[0])

    color_vals = np.array(color_vals)
    color_vals = color_vals.astype(int)
    label_scatter = torch.concat([torch.from_numpy(indices[:, None]), torch.from_numpy(label_ind[:, None])], axis=-1)

    return x, af, labelf, label_scatter, color_vals


def plot_vae(vae, vis, x, u, labels, label_scatter, color_vals):
    global win_test_reco, win_label, win_space_z, win_embed_space_z

    x = x.cuda()
    x = x.view(-1, 30, 30, 2)
    batch_size = x.size(0)

    with torch.no_grad():
        zs, z_params = vae.encode(x)
        xs, x_params = vae.decode(zs)

    # #################################### Plot few reconstruction of the tactile data #################################################
    print('MSE Loss -', loss(xs, x).cpu().numpy())

    fig, ax = plt.subplots(3, 3, figsize=(5, 5))
    # Plot mean reconstruction
    tactile_recon_mean = torch.mean(xs, dim=(1,3))
    tactile_recon_gt = torch.mean(x, dim=(1,3))

    index = 0
    for i in range(3):
        for j in range(3):
            ax[i, j].plot(tactile_recon_mean[index, :].cpu().detach(), '.r')
            ax[i, j].plot(tactile_recon_gt[index, :].cpu().detach(), '.b')
            index += int(x.shape[0] / 9)  # 9 the samples
    win_test_reco = vis.matplot(plt, win=win_test_reco)
    plt.close()

    # #################################### Plot the latent space #################################################
    qz_means = z_params[:, :, 0].cpu()

    z_embedded = umap.UMAP(n_components=2).fit_transform(qz_means.numpy())
    z_embedded = np.concatenate([z_embedded, np.zeros([batch_size, 1])], axis=-1)
    var_z = torch.std(qz_means.contiguous(), dim=0).pow(2)
    var_sort, dim_z = torch.sort(var_z)
    # print('Z Var: ', var_sort)
    qz_means_inf = qz_means[:, dim_z[-3:]]
    color_vals_z = color_vals
    win_space_z = vis.scatter(qz_means_inf,
                              opts={'caption': 'latent z space', 'markercolor': color_vals_z, 'markersize': 2,
                                    'layout': layout_def},
                              win=win_space_z)

    win_embed_space_z = vis.scatter(z_embedded,
                                    opts={'caption': 'latent z space', 'markercolor': color_vals_z, 'markersize': 2,
                                          'layout': layout_def},
                                    win=win_embed_space_z)

    win_label = vis.scatter(label_scatter, opts={'caption': 'labels', 'markercolor': color_vals, 'markersize': 10},
                            win=win_label)



def plot_lf(dvbf, vis, x, u, labels, label_scatter, color_vals):
    global win_space_y, win_test_reco, win_label, win_space_z, win_embed_space_y, win_embed_space_z, win_embed_space_h

    x = x.cuda()
    u = u.cuda()
    labels = labels.cuda()
    batch_size = x.shape[0]
    with torch.no_grad():
        prior_params_y_f, y_params_f, ys_f, prior_params_z_f, z_params_f, zs_f, xs_hat_f, x_hat_params_f, x_f, hs_f, cs_f = dvbf.filter(x, u, labels, H=1)

    y_params = torch.stack(y_params_f, dim=1)
    z_params = torch.stack(z_params_f, dim=1)
    x_recon = torch.stack(xs_hat_f, dim=1)

    # #################################### Plot few reconstruction of the tactile data #################################################
    print('MSE Loss -', loss(x_recon, x).cpu().numpy())
    # Plot mean reconstruction
    tactile_recon_mean = torch.mean(x_recon, dim=(2, 3, 4))
    tactile_recon_gt = torch.mean(x, dim=(2, 3, 4))
    fig, ax = plt.subplots(3, 3, figsize=(5, 5))
    index = 0
    for i in range(3):
        for j in range(3):
            ax[i, j].plot(tactile_recon_mean[index, :].cpu().detach(), '.r')
            ax[i, j].plot(tactile_recon_gt[index, :].cpu().detach(), '.b')
            index += int(x.shape[0] / 9)  # 9 the samples
    win_test_reco = vis.matplot(plt, win=win_test_reco)
    plt.close()

    # #################################### Plot the latent space #################################################
    # time_indx = np.arange(0, 26)  # Plot the inferred last time steps of the latent
    y_params = y_params.cpu()
    qy_means = y_params[:, :, :, 0]
    qy_means = qy_means.reshape(y_params.shape[0]*(horizon-1), dvbf.dim_y)

    #y_embedded = umap.UMAP(n_components=2).fit_transform(qy_means.numpy())
    #y_embedded = np.concatenate([y_embedded, np.zeros([batch_size*(horizon-1), 1])], axis=-1)
    var_y = torch.std(qy_means.contiguous(), dim=0).pow(2)
    var_sort, dim_y = torch.sort(var_y)
    # print('Y Var: ', var_sort)
    qy_means_inf = qy_means[:, dim_y[-3:]]
    color_vals_y = color_vals.reshape([batch_size, horizon, 3])
    color_vals_y = color_vals_y[:, 0:-1, :]
    color_vals_y = color_vals_y.reshape([batch_size*(horizon-1), 3])
    win_space_y = vis.scatter(qy_means_inf,
                              opts={'caption': 'latent y space', 'markercolor': color_vals_y, 'markersize': 2.5,
                                    'layout': layout_def},
                              win=win_space_y)
    '''
    win_embed_space_y = vis.scatter(y_embedded,
                                    opts={'caption': 'latent y space', 'markercolor': color_vals_y, 'markersize': 2.5,
                                          'layout': layout_def},
                                    win=win_embed_space_y)
    '''
    z_params = z_params.cpu()
    qz_means = z_params[:, :, :, 0]
    qz_means = qz_means.reshape(z_params.shape[0] * horizon, z_params.shape[-2])

    #z_embedded = umap.UMAP(n_components=2).fit_transform(qz_means.numpy())
    #z_embedded = np.concatenate([z_embedded, np.zeros([batch_size*horizon, 1])], axis=-1)
    var_z = torch.std(qz_means.contiguous(), dim=0).pow(2)
    var_sort, dim_z = torch.sort(var_z)
    # print('Z Var: ', var_sort)
    qz_means_inf = qz_means[:, dim_z[-3:]]

    win_space_z = vis.scatter(qz_means_inf,
                              opts={'caption': 'latent z space', 'markercolor': color_vals, 'markersize': 2, 'layout': layout_def},
                              win=win_space_z)
    '''
    win_embed_space_z = vis.scatter(z_embedded,
                              opts={'caption': 'latent z space', 'markercolor': color_vals, 'markersize': 2, 'layout': layout_def},
                              win=win_embed_space_z)
    '''


def save_lf(dvbf, x, u, labels, label_scatter, color_vals, iteration, out_dir):

    x = x.cuda()
    u = u.cuda()
    labels = labels.cuda()
    batch_size = x.shape[0]
    with torch.no_grad():
        prior_params_y_f, y_params_f, ys_f, prior_params_z_f, z_params_f, zs_f, xs_hat_f, x_hat_params_f, x_f, hs_f, cs_f = dvbf.filter(x, u, labels, H=1)

    y_params = torch.stack(y_params_f, dim=1)
    z_params = torch.stack(z_params_f, dim=1)
    x_recon = torch.stack(xs_hat_f, dim=1)

    # #################################### Plot few reconstruction of the tactile data #################################################
    # print('MSE Loss -', loss(x_recon, x).cpu().numpy())
    # Plot mean reconstruction
    tactile_recon_mean = torch.mean(x_recon, dim=(2, 3, 4))
    tactile_recon_gt = torch.mean(x, dim=(2, 3, 4))
    fig, ax = plt.subplots(3, 3, figsize=(5, 5))
    index = 0
    for i in range(3):
        for j in range(3):
            ax[i, j].plot(tactile_recon_mean[index, :].cpu().detach(), '.r')
            ax[i, j].plot(tactile_recon_gt[index, :].cpu().detach(), '.b')
            index += int(x.shape[0] / 9)  # 9 the samples
    recon_fig_name = os.path.join(out_dir, 'reconstructed_'+str(iteration)+'.png')
    plt.savefig(recon_fig_name)
    plt.close()

    # #################################### Plot the latent space #################################################
    # time_indx = np.arange(0, 26)  # Plot the inferred last time steps of the latent
    y_params = y_params.cpu()
    qy_means = y_params[:, :, :, 0]
    qy_means = qy_means.reshape(y_params.shape[0]*(horizon-1), dvbf.dim_y)

    #y_embedded = umap.UMAP(n_components=2).fit_transform(qy_means.numpy())
    #y_embedded = np.concatenate([y_embedded, np.zeros([batch_size*(horizon-1), 1])], axis=-1)
    var_y = torch.std(qy_means.contiguous(), dim=0).pow(2)
    var_sort, dim_y = torch.sort(var_y)
    # print('Y Var: ', var_sort)
    qy_means_inf = qy_means[:, dim_y[-3:]]
    color_vals_y = color_vals.reshape([batch_size, horizon, 3])
    color_vals_y = color_vals_y[:, 0:-1, :]
    color_vals_y = color_vals_y.reshape([batch_size*(horizon-1), 3])
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(qy_means_inf[:, 0], qy_means_inf[:, 1], qy_means_inf[:, 2], c=color_vals_y/255, s=10)
    plt.savefig(os.path.join(out_dir, 'latent_y_space_'+str(iteration)+'.png'))
    plt.close()

    z_params = z_params.cpu()
    qz_means = z_params[:, :, :, 0]
    qz_means = qz_means.reshape(z_params.shape[0] * horizon, z_params.shape[-2])

    #z_embedded = umap.UMAP(n_components=2).fit_transform(qz_means.numpy())
    #z_embedded = np.concatenate([z_embedded, np.zeros([batch_size*horizon, 1])], axis=-1)
    var_z = torch.std(qz_means.contiguous(), dim=0).pow(2)
    var_sort, dim_z = torch.sort(var_z)
    # print('Z Var: ', var_sort)
    qz_means_inf = qz_means[:, dim_z[-3:]]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(qz_means_inf[:, 0], qz_means_inf[:, 1], qz_means_inf[:, 2], c=color_vals/255, s=5)
    plt.savefig(os.path.join(out_dir, 'latent_z_space_' + str(iteration) + '.png'))
    plt.close()

    # Perform quick distance computation of the last time steps

    features_np = torch.concat([z_params[:, -2:-1, :, 0:1], y_params[:, -2:-1, :, 0:1]], axis=2)
    features_np = features_np[:, 0, :, 0]
    features_np = features_np.cpu().detach().numpy()
    labels_np = labels[:, -1, 0]
    labels_np = labels_np.cpu().detach().numpy()
    distance_matrix, unique_classes = compute_distances(features_np, labels_np)
    plt.figure(figsize=(15, 10))
    sns.heatmap(distance_matrix, annot=True, cmap='coolwarm', xticklabels=unique_classes, yticklabels=unique_classes)
    plt.gca().invert_yaxis()
    plt.title('Distance Matrix (Intra-Class on Diagonal, Inter-Class Off-Diagonal)')
    plt.xlabel('Class Label')
    plt.ylabel('Class Label')
    plt.savefig(os.path.join(out_dir, 'distance_metric_' + str(iteration) + '.png'))
    plt.close()


def format(x):
    img = torch.clip(x * 255., 0, 255).to(torch.uint8)
    return img.view(-1, 32, 32).numpy()


def display_video(x, x_recon, filename):
    T = 14
    frames = []
    for i in range(T):
        gt = format(x[i])
        pred = format(x_recon[i])
        img = np.concatenate([gt, pred], axis=1).squeeze()
        # cv2.imshow(mat=img, winname='generated')
        # cv2.waitKey(5)
        # plt.imshow(img)
        # plt.show()
        frames.append(img)

    with imageio.get_writer(f"{filename}.mp4", mode="I") as writer:
        for idx, frame in enumerate(frames):
            writer.append_data(frame)