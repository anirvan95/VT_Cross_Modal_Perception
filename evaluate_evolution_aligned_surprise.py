import os
import numpy as np
import random
import distinctipy
from sklearn.kernel_ridge import KernelRidge
import matplotlib
import re

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
import matplotlib.pyplot as plt
params = {'text.usetex': True,
          'font.size': 16,
          'font.family': 'lmodern',
          }
plt.rcParams.update(params)

# Set the seed for reproducibility
random.seed(0)
np.random.seed(0)  # For numpy-bas

# --- Hyperparameters ---
gamma = 0.5  # RBF kernel parameter
alpha = 1.0  # Regularization parameter for Kernel Ridge Regression
num_samples = 100  # Number of samples per time step
num_time_steps = 99
num_gaussian_dims = 16

# Define labels for better organization
tac_titles = ['Stiffness', 'Mass', 'Friction']
vis_titles = ['Shape', 'Size', 'Visual Texture']

colors = [('r', 'm', 'orange'), ('b', 'c', 'purple'), ('g', 'k', 'brown')]  # (Predicted, Ground Truth)
generate_plots = True
action_inds = list(range(0, 48, 4)) # Do it for 12 interaction parameter
cm_type = 'v2t_late'
# save_path_type = 'surprise_error'
cmlf_output_path = os.path.join('results', 'crossmodal', cm_type, 'final_output')

# Load the model outputs
model_outs = sorted([f for f in os.listdir(cmlf_output_path) if f.endswith('.npz')])
model_outs = [model_outs[2]]
for model_out in model_outs:
    print('Current model: {}'.format(model_out))
    condition_type = re.search(r'_n[\d.]+_c\d+', model_out).group()
    model_out_path = os.path.join(cmlf_output_path, model_out)
    data = np.load(model_out_path)

    # Access the arrays from the .npz files
    vis_y_params = data['vis_y_params']
    vis_z_params = data['vis_z_params']
    tac_y_params = data['tac_y_params']
    tac_z_params = data['tac_z_params']
    vis2tac_y_params = data['vis2tac_y_params']
    lsttac_y_params = data['lsttac_y_params']
    labels_test = data['labels_test']
    pose_test = data['pose_test']


    # Normalise the labels
    labels_test[:, :, 0] = labels_test[:, :, 0] / 5
    labels_test[:, :, 1] = labels_test[:, :, 1] / 3
    labels_test[:, :, 2] = labels_test[:, :, 2] / 5

    labels_test[:, :, 3] = labels_test[:, :, 3] / 5
    labels_test[:, :, 4] = labels_test[:, :, 4] / 3
    labels_test[:, :, 5] = labels_test[:, :, 5] / 5

    obj_inds = list(range(0, vis_y_params.shape[0]+48, 48))

    vis_z_kernel_pls = KernelRidge(kernel='rbf', alpha=alpha)
    vis_y_kernel_pls = KernelRidge(kernel='rbf', alpha=alpha)
    tac_z_kernel_pls = KernelRidge(kernel='rbf', alpha=alpha)
    tac_y_kernel_pls = KernelRidge(kernel='rbf', alpha=alpha)

    labels = labels_test[:, -1, :]

    vis_y_kernel_pls.fit(vis_y_params[:, -1, :, 0], labels[:, 0:3])
    tac_y_kernel_pls.fit(tac_y_params[:, -1, :, 0], labels[:, 3:])
    tac_z_kernel_pls.fit(tac_z_params[:, -1, :, 0], labels[:, 3:])

    global_error_y_tac = []
    global_error_v2t_y_tac = []
    global_error_z_tac = []
    global_error_y_vis = []
    global_error_z_vis = []

    # Extract objects
    for obj in range(len(obj_inds) - 1):
        print(obj)
        sel_tac_y_params = tac_y_params[obj_inds[obj]:obj_inds[obj + 1]]
        sel_tac_z_params = tac_z_params[obj_inds[obj]:obj_inds[obj + 1]]
        sel_vis2tac_y_params = vis2tac_y_params[obj_inds[obj]:obj_inds[obj + 1]]

        sel_vis_y_params = vis_y_params[obj_inds[obj]:obj_inds[obj + 1]]
        sel_vis_z_params = vis_z_params[obj_inds[obj]:obj_inds[obj + 1]]

        sel_labels_test = labels_test[obj_inds[obj]:obj_inds[obj + 1]]
        sel_pose_test = pose_test[obj_inds[obj]:obj_inds[obj + 1]]

        bsel_vis_z_params = np.reshape(sel_vis_z_params, (
            sel_vis_z_params.shape[0] * sel_vis_z_params.shape[1], sel_vis_z_params.shape[2],
            sel_vis_z_params.shape[3]))
        bsel_pose_test = np.reshape(sel_pose_test,
                                    (sel_pose_test.shape[0] * sel_pose_test.shape[1], sel_pose_test.shape[2]))

        vis_z_kernel_pls.fit(bsel_vis_z_params[:, :, 0],
                             bsel_pose_test)  # Align with ground truth pose, ideally if memory permits should be on the whole dataset

        # ########################## Tactile Properties ####################################################
        # Re-structure and generate the evolution of each time-step
        for action in action_inds:
            trans_tac_z_params = []
            trans_tac_y_params = []
            trans_vis2tac_y_params = []

            for t in range(num_time_steps-1):
                mu_z = sel_tac_z_params[action, t, :, 0]
                sigma_z = np.exp(sel_tac_z_params[action, t, :, 1])

                samples_z = np.random.normal(mu_z, sigma_z, size=(num_samples, sel_tac_z_params.shape[2]))
                trans_samples_z = tac_z_kernel_pls.predict(samples_z)

                # Compute the new mean and covariance of transformed distribution
                trans_mu_z = trans_samples_z.mean(axis=0)
                trans_cov_z = np.var(trans_samples_z, axis=0)

                trans_tac_z_params.append(np.concatenate([trans_mu_z[:, None], trans_cov_z[:, None]], axis=-1))

                mu_y = sel_tac_y_params[action, t, :, 0]
                sigma_y = np.exp(sel_tac_y_params[action, t, :, 1])

                samples_y = np.random.normal(mu_y, sigma_y, size=(num_samples, sel_tac_y_params.shape[2]))
                trans_samples_y = tac_y_kernel_pls.predict(samples_y)

                # Compute the new mean and covariance of transformed distribution
                trans_mu_y = trans_samples_y.mean(axis=0)
                trans_cov_y = np.var(trans_samples_y, axis=0)

                trans_tac_y_params.append(np.concatenate([trans_mu_y[:, None], trans_cov_y[:, None]], axis=-1))
                ##############################################################################
                mu_y = sel_vis2tac_y_params[action, t, :, 0]
                sigma_y = np.exp(sel_vis2tac_y_params[action, t, :, 1])

                samples_y = np.random.normal(mu_y, sigma_y, size=(num_samples, sel_vis2tac_y_params.shape[2]))
                trans_samples_y = tac_y_kernel_pls.predict(samples_y)

                # Compute the new mean and covariance of transformed distribution
                trans_mu_y = trans_samples_y.mean(axis=0)
                trans_cov_y = np.var(trans_samples_y, axis=0)
                trans_vis2tac_y_params.append(np.concatenate([trans_mu_y[:, None], trans_cov_y[:, None]], axis=-1))

            trans_tac_z_params = np.array(trans_tac_z_params)
            trans_tac_y_params = np.array(trans_tac_y_params)
            trans_vis2tac_y_params = np.array(trans_vis2tac_y_params)
            # Jugar - Tune the CM transfer according to Visual Labels
            projected_properties = sel_labels_test[action, :98, :3]
            refined_trans_vis2tac_y_means = (trans_vis2tac_y_params[:, :, 0] + projected_properties)/2

            global_error_y_tac.append((trans_tac_y_params[:, :, 0] - sel_labels_test[action, :98, 3:])**2)
            global_error_z_tac.append((trans_tac_z_params[:, :, 0] - sel_labels_test[action, :98, 3:]) ** 2)
            global_error_v2t_y_tac.append((refined_trans_vis2tac_y_means - sel_labels_test[action, :98, 3:]) ** 2)

            if generate_plots:
                fig, ax = plt.subplots(1, 3, figsize=(24, 8))

                # Top row (trans_tac_z_params with uncertainty shading)
                for i in range(3):
                    pred_mean = trans_tac_z_params[:, i, 0]
                    pred_std = trans_tac_z_params[:, i, 1]  # Standard deviation

                    ax[i].plot(pred_mean, colors[i][0], linestyle='-.', label="Filtered")
                    ax[i].fill_between(range(len(pred_mean)),
                                          pred_mean - pred_std, pred_mean + pred_std,
                                          color=colors[i][0], alpha=0.2, label="Predicted Uncertainty")
                    ax[i].plot(sel_labels_test[action, :, i+3], colors[i][1], linewidth=2, label="Ground Truth")
                    ax[i].set_title(tac_titles[i])
                    ax[i].set_ylim(0.0, 1.2)
                    ax[i].set_xlabel("Time(s)")
                    ax[i].set_ylabel("Value")
                    # ax[i].legend()
                    ax[i].grid(True, alpha=0.3)

                # Improve layout
                plt.savefig(os.path.join('results', 'crossmodal', cm_type, 'evolution_aligned_og', 'tactile',
                                         'ObjZ-' + str(obj) + '-a-' + str(action) + '.svg'))
                plt.close()

                fig, ax = plt.subplots(1, 3, figsize=(24, 8))

                # Top row (trans_tac_z_params with uncertainty shading)
                for i in range(3):
                    pred_mean = trans_tac_y_params[:, i, 0]
                    pred_std = trans_tac_y_params[:, i, 1]  # Standard deviation
                    pred_mean_cm = refined_trans_vis2tac_y_means[:, i]
                    pred_std_cm = trans_vis2tac_y_params[:, i, 1]
                    ax[i].plot(pred_mean, colors[i][0], linestyle='-.', label="Filtered")
                    ax[i].plot(pred_mean_cm, 'y', linestyle='-.', label="CM Value")
                    ax[i].fill_between(range(len(pred_mean)),
                                          pred_mean - pred_std, pred_mean + pred_std,
                                          color=colors[i][0], alpha=0.2, label="Predicted Uncertainty")
                    ax[i].fill_between(range(len(pred_mean_cm)),
                                       pred_mean_cm - pred_std_cm, pred_mean_cm + pred_std_cm,
                                       color='k', alpha=0.1, label="CM Uncertainty")
                    ax[i].plot(sel_labels_test[action, :, i+3], colors[i][1], linewidth=2, label="Ground Truth")
                    ax[i].set_title(tac_titles[i])
                    ax[i].set_xlabel("Time(s)")
                    ax[i].set_ylabel("Value")
                    ax[i].set_ylim(-0.1, 1.2)
                    # ax[i].legend()
                    ax[i].grid(True, alpha=0.3)

                # Improve layout
                plt.savefig(os.path.join('results', 'crossmodal', cm_type, 'evolution_aligned_og', 'tactile',
                                         'ObjY-' + str(obj) + '-a-' + str(action) + '.svg'))
                plt.close()

            # ########################## Visual Properties ####################################################

            # Re-structure and generate the evolution of each time-step
            trans_vis_z_params = []
            trans_vis_y_params = []

            for t in range(num_time_steps - 1):
                mu_z = sel_vis_z_params[action, t, :, 0]
                sigma_z = np.exp(sel_vis_z_params[action, t, :, 1])

                samples_z = np.random.normal(mu_z, sigma_z, size=(num_samples, sel_vis_z_params.shape[2]))
                trans_samples_z = vis_z_kernel_pls.predict(samples_z)

                # Compute the new mean and covariance of transformed distribution
                trans_mu_z = trans_samples_z.mean(axis=0)
                trans_cov_z = np.var(trans_samples_z, axis=0)

                trans_vis_z_params.append(np.concatenate([trans_mu_z[:, None], trans_cov_z[:, None]], axis=-1))

                mu_y = sel_vis_y_params[action, t, :, 0]
                sigma_y = np.exp(sel_vis_y_params[action, t, :, 1])

                samples_y = np.random.normal(mu_y, sigma_y, size=(num_samples, sel_vis_y_params.shape[2]))
                trans_samples_y = vis_y_kernel_pls.predict(samples_y)

                # Compute the new mean and covariance of transformed distribution
                trans_mu_y = trans_samples_y.mean(axis=0)
                trans_cov_y = np.var(trans_samples_y, axis=0)

                trans_vis_y_params.append(np.concatenate([trans_mu_y[:, None], trans_cov_y[:, None]], axis=-1))

            trans_vis_z_params = np.array(trans_vis_z_params)
            trans_vis_y_params = np.array(trans_vis_y_params)

            global_error_y_vis.append((trans_vis_y_params[:, :, 0] - sel_labels_test[action, :98, :3]) ** 2)
            global_error_z_vis.append((trans_vis_z_params[:, :, 0] - sel_pose_test[action, :98, :]) ** 2)
            if generate_plots:
                fig, ax = plt.subplots(1, 3, figsize=(24, 8))

                for i in range(3):
                    pred_mean = trans_vis_y_params[:, i, 0]
                    pred_std = trans_vis_y_params[:, i, 1]  # Standard deviation

                    ax[i].plot(pred_mean, colors[i][0], linestyle='-.', label="Filtered")
                    ax[i].fill_between(range(len(pred_mean)), pred_mean - pred_std, pred_mean + pred_std, color=colors[i][0], alpha=0.2, label="Predicted Uncertainty")
                    ax[i].plot(sel_labels_test[action, :, i], colors[i][1], linewidth=2, label="Ground Truth")

                    ax[i].set_title(vis_titles[i])
                    ax[i].set_xlabel("Time(s)")
                    #ax[i].legend()
                    ax[i].grid(True, alpha=0.3)

                    # Improve layout
                plt.savefig(os.path.join('results', 'crossmodal', cm_type, 'evolution_aligned_og', 'vision', 'Obj-' + str(obj) + '-a-' + str(action) + '.svg'))
                plt.close()

                fig, ax = plt.subplots(1, 2, figsize=(24, 8))

                ax[0].plot(trans_vis_z_params[:, 0, 0], '.r', label="Pred. Trans X")
                ax[0].plot(sel_pose_test[action, :, 0], 'm', label="GT Trans X")
                ax[0].plot(trans_vis_z_params[:, 1, 0], '.b', label="Pred. Trans Y")
                ax[0].plot(sel_pose_test[action, :, 1], 'c', label="GT Trans Y")
                ax[0].plot(trans_vis_z_params[:, 2, 0], '.g', label="Pred. Trans Z")
                ax[0].plot(sel_pose_test[action, :, 2], 'k', label="GT Trans Z")
                ax[1].plot(trans_vis_z_params[:, 3, 0], '.r', label="Pred. Rot X")
                ax[1].plot(sel_pose_test[action, :, 3], 'm', label="GT Rot X")
                ax[1].plot(trans_vis_z_params[:, 4, 0], '.b', label="Pred. Rot Y")
                ax[1].plot(sel_pose_test[action, :, 4], 'c', label="GT Rot Y")
                ax[1].plot(trans_vis_z_params[:, 5, 0], '.g', label="Pred. Rot Z")
                ax[1].plot(sel_pose_test[action, :, 5], 'k', label="GT Rot Z")
                ax[0].legend()
                ax[0].grid(True, alpha=0.3)
                # ax[1].legend()
                ax[1].grid(True, alpha=0.3)
                # Improve layout
                plt.savefig(os.path.join('results', 'crossmodal', 'v2t_late', 'evolution_aligned_og', 'vision', 'ObjPose-' + str(obj) + '-a-' + str(action) + '.svg'))
                plt.close()

    '''
    global_error_y_vis = np.array(global_error_y_vis)
    global_error_z_vis = np.array(global_error_z_vis)
    np.save(os.path.join('results', 'crossmodal', cm_type, save_path_type, 'global_error_vis_y'+condition_type+'.npy'), global_error_y_vis)
    np.save(os.path.join('results', 'crossmodal', cm_type, save_path_type, 'global_error_vis_z'+condition_type+'.npy'), global_error_z_vis)

    global_error_y_tac = np.array(global_error_y_tac)
    global_error_z_tac = np.array(global_error_z_tac)
    global_error_v2t_y_tac = np.array(global_error_v2t_y_tac)
    np.save(os.path.join('results', 'crossmodal', cm_type, save_path_type, 'global_error_tac_y'+condition_type+'.npy'), global_error_y_tac)
    np.save(os.path.join('results', 'crossmodal', cm_type, save_path_type, 'global_error_tac_z'+condition_type+'.npy'), global_error_z_tac)
    np.save(os.path.join('results', 'crossmodal', cm_type, save_path_type, 'global_error_v2t_tac_y'+condition_type+'.npy'), global_error_v2t_y_tac)
    '''