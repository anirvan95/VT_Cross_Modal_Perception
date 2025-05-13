import os
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
import torch
import random
import numpy as np
import imageio
import umap

layout_def = {
    'scene': {
        'xaxis': {'range': [-0.1, 0.1]},
        'yaxis': {'range': [-0.1, 0.1]},
        'zaxis': {'range': [0, 0.1]},
    }
}

# Seeding for reproducibility
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Dataset specific parameters
vis_obs_dim = [64, 64, 2]
tac_obs_dim = [80, 80, 1]
action_dim = 9
horizon = 99

win_vis_test_reco = None
win_tac_test_reco = None
vis_win_space_y = None
tac_win_space_y = None
vis_win_space_z = None
tac_win_space_z = None
win_label = None

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


def validate_cmlf(test_loader, cmlf, out_dir, H, iteration, save_plot=False, show_plot=False, vis=None):
    first = True
    with torch.no_grad():
        for i, values in enumerate(test_loader):
            vis_obs, tac_obs, actions, labels = values

            vis_obs = vis_obs.cuda().to(dtype=torch.float32)
            tac_obs = tac_obs.cuda().to(dtype=torch.float32)
            actions = actions.cuda().to(dtype=torch.float32)
            labels = labels.cuda().to(dtype=torch.float32)

            object_labels = labels[:, :, 6:]  # Required for hierarchical prior, contrastive version is more general
            pose_labels = labels[:, :, :6]  # GT Pose of the object

            (vis_prior_params_y_f, vis_y_params_f, vis_ys_f, vis_prior_params_z_f, vis_z_params_f, vis_zs_f, vis_xs_hat_f,
             vis_x_hat_params_f, vis_x_f, vis_hs_f, vis_cs_f,
             tac_prior_params_y_f, tac_y_params_f, tac_ys_f, tac_prior_params_z_f, tac_z_params_f, tac_zs_f, tac_xs_hat_f,
             tac_x_hat_params_f, tac_x_f, tac_hs_f, tac_cs_f,
             tac2vis_y_params_f, lstvis_y_params_f) = cmlf.filter(vis_obs, tac_obs, actions, object_labels, H)

            # Move results to CPU to reduce GPU memory usage
            vis_y_params_f = torch.stack(vis_y_params_f, dim=1).cpu()
            vis_z_params_f = torch.stack(vis_z_params_f, dim=1).cpu()
            vis_x_recon_f = torch.stack(vis_xs_hat_f, dim=1).cpu()
            x_gt_f = torch.stack(vis_x_f, dim=1).cpu()
            tac_y_params_f = torch.stack(tac_y_params_f, dim=1).cpu()
            tac_z_params_f = torch.stack(tac_z_params_f, dim=1).cpu()
            tac_x_recon_f = torch.stack(tac_xs_hat_f, dim=1).cpu()
            tac_gt_f = torch.stack(tac_x_f, dim=1).cpu()

            if first:
                vis_y_params, vis_z_params, vis_x_recons, vis_x_gts = vis_y_params_f, vis_z_params_f, vis_x_recon_f, x_gt_f
                tac_y_params, tac_z_params, tac_x_recons, tac_x_gts = tac_y_params_f, tac_z_params_f, tac_x_recon_f, tac_gt_f
                labels_test = object_labels.cpu()
                first = False
            else:
                vis_y_params = torch.cat((vis_y_params, vis_y_params_f), dim=0)
                vis_z_params = torch.cat((vis_z_params, vis_z_params_f), dim=0)
                vis_x_recons = torch.cat((vis_x_recons, vis_x_recon_f), dim=0)
                vis_x_gts = torch.cat((vis_x_gts, x_gt_f), dim=0)
                tac_y_params = torch.cat((tac_y_params, tac_y_params_f), dim=0)
                tac_z_params = torch.cat((tac_z_params, tac_z_params_f), dim=0)
                tac_x_recons = torch.cat((tac_x_recons, tac_x_recon_f), dim=0)
                tac_x_gts = torch.cat((tac_x_gts, tac_gt_f), dim=0)
                labels_test = torch.cat((labels_test, object_labels.cpu()), dim=0)

    num_test_samples = labels_test.shape[0]
    object_ind = ((labels_test[:, :, 0]-1)*15 + (labels_test[:, :, 1]-1)*5 + (labels_test[:, :, 2]-1))
    object_ind = object_ind.reshape(-1)
    cmap1 = plt.get_cmap('tab20b')
    color_vals =  cmap1(object_ind.numpy() / 75)
    color_vals = color_vals[:, 0:3]
    color_vals = color_vals.reshape(num_test_samples, horizon, 3)
    # test_object_labels = object_ind.reshape(num_test_samples, horizon)

    # if show_plot:
    #     global win_label
    #     label_scatter = test_object_labels.reshape(-1).numpy()
    #     label_ind = np.arange(num_test_samples*horizon)
    #     label_scatter = np.concatenate((label_ind[:, None], label_scatter[:, None]), axis=-1)
    #     color_scatter = color_vals.reshape(num_test_samples*horizon, 3)
    #     color_scatter = color_scatter * 255
    #     color_scatter = color_scatter.astype(int)
    #     win_label = vis.scatter(label_scatter, opts={'caption': 'labels', 'markercolor': color_scatter, 'markersize': 10}, win=win_label)

    vis_recon_mean = torch.mean(vis_x_recons, dim=(2, 3, 4))
    vis_gt_mean = torch.mean(vis_x_gts, dim=(2, 3, 4))

    tactile_recon_mean = torch.mean(tac_x_recons, dim=(2, 3, 4))
    tactile_gt_mean = torch.mean(tac_x_gts, dim=(2, 3, 4))
    # ############################################### Plot reconstruction #############################################
    # #################################################### Vision #####################################################
    fig, ax = plt.subplots(3, 3, figsize=(5, 5))
    index = 0
    for i in range(3):
        for j in range(3):
            ax[i, j].plot(vis_recon_mean[index, :].detach().numpy(), '.r')
            ax[i, j].plot(vis_gt_mean[index, :].detach(), '.b')
            index += int(num_test_samples / 9)  # 9 the samples

    if show_plot:
        global win_vis_test_reco
        win_vis_test_reco = vis.matplot(plt, win=win_vis_test_reco)
        plt.close()
    elif save_plot:
        recon_fig_name = os.path.join(out_dir, 'vis_reconstructed_' + str(iteration) + '.png')
        plt.savefig(recon_fig_name)
        plt.close()
    # #################################################### Tactile ####################################################
    fig, ax = plt.subplots(3, 3, figsize=(5, 5))
    index = 0
    for i in range(3):
        for j in range(3):
            ax[i, j].plot(tactile_recon_mean[index, :].detach(), '.c')
            ax[i, j].plot(tactile_gt_mean[index, :].cpu().detach(), '.m')
            index += int(num_test_samples / 9)  # 9 the samples

    if show_plot:
        global win_tac_test_reco
        win_tac_test_reco = vis.matplot(plt, win=win_tac_test_reco)
        plt.close()
    elif save_plot:
        recon_fig_name = os.path.join(out_dir, 'tac_reconstructed_' + str(iteration) + '.png')
        plt.savefig(recon_fig_name)
        plt.close()

    # ############################################### Plot Y-Latent Space #############################################
    color_vals_y = color_vals[:, 0:-1, :]
    color_vals_y = color_vals_y.reshape(num_test_samples * (horizon - 1), 3)
    color_vals_y_visdom = color_vals_y*255
    color_vals_y_visdom = color_vals_y_visdom.astype(int)
    # #################################################### Vision #####################################################
    vis_qy_means = vis_y_params[:, :, :, 0]
    vis_qy_means = vis_qy_means.reshape(num_test_samples * (horizon - 1), cmlf.vis_dim_y)
    vis_var_y = torch.std(vis_qy_means.contiguous(), dim=0).pow(2)
    var_sort, dim_y = torch.sort(vis_var_y)
    vis_qy_means_inf = vis_qy_means[:, dim_y[-3:]]
    if show_plot:
        global vis_win_space_y
        vis_win_space_y = vis.scatter(vis_qy_means_inf,
                                  opts={'caption': 'latent vis y space', 'markercolor': color_vals_y_visdom, 'markersize': 2.5,
                                        'layout': layout_def},
                                  win=vis_win_space_y)
    elif save_plot:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(vis_qy_means_inf[:, 0], vis_qy_means_inf[:, 1], vis_qy_means_inf[:, 2], c=color_vals_y, s=10)
        plt.savefig(os.path.join(out_dir, 'vis_latent_y_space_'+str(iteration)+'.png'))
        plt.close()

    # #################################################### Tactile #####################################################
    tac_qy_means = tac_y_params[:, :, :, 0]
    tac_qy_means = tac_qy_means.reshape(num_test_samples * (horizon - 1), cmlf.tac_dim_y)
    tac_var_y = torch.std(tac_qy_means.contiguous(), dim=0).pow(2)
    var_sort, dim_y = torch.sort(tac_var_y)
    tac_qy_means_inf = tac_qy_means[:, dim_y[-3:]]
    if show_plot:
        global tac_win_space_y
        tac_win_space_y = vis.scatter(tac_qy_means_inf,
                                      opts={'caption': 'latent vis y space', 'markercolor': color_vals_y_visdom,
                                            'markersize': 2.0,
                                            'layout': layout_def},
                                      win=tac_win_space_y)
    elif save_plot:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(tac_qy_means_inf[:, 0], tac_qy_means_inf[:, 1], tac_qy_means_inf[:, 2], c=color_vals_y, s=10)
        plt.savefig(os.path.join(out_dir, 'tac_latent_y_space_' + str(iteration) + '.png'))
        plt.close()

    # # ############################################### Plot Z-Latent Space #############################################
    # color_vals_z = color_vals.reshape(num_test_samples*horizon, 3)
    # color_vals_z_visdom = color_vals_z*255
    # color_vals_z_visdom = color_vals_z_visdom.astype(int)
    # # #################################################### Vision #####################################################
    # vis_qz_means = vis_z_params[:, :, :, 0]
    # vis_qz_means = vis_qz_means.reshape(num_test_samples * horizon, cmlf.vis_dim_z)
    # vis_var_z = torch.std(vis_qz_means.contiguous(), dim=0).pow(2)
    # vis_var_sort, dim_z = torch.sort(vis_var_z)
    # vis_qz_means_inf = vis_qz_means[:, dim_z[-3:]]
    # if show_plot:
    #     global vis_win_space_z
    #     vis_win_space_z = vis.scatter(vis_qz_means_inf,
    #                                   opts={'caption': 'latent vis z space', 'markercolor': color_vals_z_visdom,
    #                                         'markersize': 2.5,
    #                                         'layout': layout_def},
    #                                   win=vis_win_space_z)
    # elif save_plot:
    #     fig = plt.figure(figsize=(10, 8))
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.scatter(vis_qz_means_inf[:, 0], vis_qz_means_inf[:, 1], vis_qz_means_inf[:, 2], c=color_vals_z, s=10)
    #     plt.savefig(os.path.join(out_dir, 'vis_latent_z_space_' + str(iteration) + '.png'))
    #     plt.close()
    #
    # # #################################################### Vision #####################################################
    # tac_qz_means = tac_z_params[:, :, :, 0]
    # tac_qz_means = tac_qz_means.reshape(num_test_samples * horizon, cmlf.tac_dim_z)
    # tac_var_z = torch.std(tac_qz_means.contiguous(), dim=0).pow(2)
    # tac_var_sort, dim_z = torch.sort(tac_var_z)
    # tac_qz_means_inf = tac_qz_means[:, dim_z[-3:]]
    # if show_plot:
    #     global tac_win_space_z
    #     tac_win_space_z = vis.scatter(tac_qz_means_inf,
    #                                   opts={'caption': 'latent vis z space', 'markercolor': color_vals_z_visdom,
    #                                         'markersize': 2.0,
    #                                         'layout': layout_def},
    #                                   win=tac_win_space_z)
    # elif save_plot:
    #     fig = plt.figure(figsize=(10, 8))
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.scatter(tac_qz_means_inf[:, 0], tac_qz_means_inf[:, 1], tac_qz_means_inf[:, 2], c=color_vals_z, s=10)
    #     plt.savefig(os.path.join(out_dir, 'tac_latent_z_space_' + str(iteration) + '.png'))
    #     plt.close()



def format(x):
    img = torch.clip(x * 255., 0, 255).to(torch.uint8)
    return img.view(-1, 32, 32).numpy()


def display_video(x, x_recon, filename):
    # TODO: Fix display video for presentation
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