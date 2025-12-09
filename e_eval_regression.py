"""
This code performs the Kernel Ridge Regression
"""
import os
import pickle
import numpy as np
import random
from sklearn.kernel_ridge import KernelRidge
import re
import copy
import argparse
from sklearn.model_selection import GridSearchCV


def remap_and_normalize(matrix, new_values, source_values=None):
    new_values = np.array(new_values, dtype=float)

    if source_values is None:
        source_values = np.arange(1.0, len(new_values) + 1.0)
    else:
        source_values = np.array(source_values, dtype=float)

    # Create lookup map
    value_map = dict(zip(source_values, new_values))
    remap_func = np.vectorize(lambda x: value_map.get(x, np.nan))
    remapped = remap_func(matrix)

    if np.isnan(remapped).any():
        raise ValueError("Matrix contains values not in source_values.")

    # Normalize to [0, 1]
    min_val = remapped.min()
    max_val = remapped.max()
    if max_val == min_val:
        normalized = np.zeros_like(remapped)  # Avoid division by zero
    else:
        normalized = (remapped - min_val) / (max_val - min_val)

    return normalized


def safe_fit(model, X, Y):
    # Find rows that have no NaNs in X or Y
    valid_rows = ~np.isnan(X).any(axis=1) & ~np.isnan(Y).any(axis=1)

    # Filter valid rows
    X_clean = X[valid_rows]
    Y_clean = Y[valid_rows]
    diff = len(X_clean) - len(X)
    if diff > 0:
        print('Warning, regression fit unstable')

    if len(X_clean) > 0:
        model.fit(X_clean, Y_clean)
        return model
    else:
        print("No valid data to fit.")
        return None


# Set the seed for reproducibility
random.seed(0)
np.random.seed(0)  # For numpy-bas

# --- Hyperparameters ---
gamma = 0.1  # RBF kernel parameter
alpha = 0.5  # Regularization parameter for Kernel Ridge Regression
# alpha = 1.0 # For surprise set, stronger regularization
num_samples = 100  # Number of samples per time step
num_time_steps = 99
percent = 50

method_name = 'baseline'
condition_type = 'n0_c0'

save_path = 'regression_errors'
test_model_output = 'out_test_'+condition_type+'.npz'
val_model_output = 'out_val_'+condition_type+'.npz'

print('Currently processing - ', method_name)
test_data = np.load(os.path.join('results', method_name, 'model_output', test_model_output))
val_data = np.load(os.path.join('results', method_name, 'model_output', val_model_output))

# Access the arrays from the .npz files
if method_name == 'joint' or method_name == 'baseline':
    y_params = np.concatenate([test_data['y_params'], val_data['y_params']])
    vis_y_params = y_params[:, :, :16, :]
    tac_y_params = y_params[:, :, 16:, :]
else:
    vis_y_params = np.concatenate([test_data['vis_y_params'], val_data['vis_y_params']])
    tac_y_params = np.concatenate([test_data['tac_y_params'], val_data['tac_y_params']])

vis_z_params = np.concatenate([test_data['vis_z_params'], val_data['vis_z_params']])
labels_test = np.concatenate([test_data['labels_test'], val_data['labels_test']])
pose_test = np.concatenate([test_data['pose_test'], val_data['pose_test']])

# Normalise the labels
old_labels = copy.deepcopy(labels_test)
# Shape
labels_test[:, :, 0] = remap_and_normalize(old_labels[:, :, 0], [1.0, 1.5, 0.1, 0.5, 3.0], [1.0, 2.0, 3.0, 4.0, 5.0])
# Size
labels_test[:, :, 1] = remap_and_normalize(old_labels[:, :, 1], [0.12, 0.15, 0.175], [1.0, 2.0, 3.0])
# Vis. Texture
labels_test[:, :, 2] = remap_and_normalize(old_labels[:, :, 2], [1.0, 0.78, 0.59, 0.39, 0.0],[1.0, 2.0, 3.0, 4.0, 5.0])
# Stiffness
labels_test[:, :, 3] = remap_and_normalize(old_labels[:, :, 3], [3.0, 9.0, 15.0, 20.0, 25.0],[1.0, 2.0, 3.0, 4.0, 5.0])
# Mass
labels_test[:, :, 4] = remap_and_normalize(old_labels[:, :, 4], [0.3, 0.5, 0.7], [1.0, 2.0, 3.0])
# Surf. Friction
labels_test[:, :, 5] = remap_and_normalize(old_labels[:, :, 5], [0.1, 0.15, 0.25, 0.45, 0.8],[1.0, 2.0, 3.0, 4.0, 5.0])

labels = labels_test[:, -1, :]
vis_z_kernel_pls = KernelRidge(kernel='rbf', alpha=alpha)
vis_y_kernel_pls = KernelRidge(kernel='rbf', alpha=alpha)
tac_y_kernel_pls = KernelRidge(kernel='rbf', alpha=alpha, gamma=gamma)

tac_X_subset = tac_y_params[:, -1, :, 0]
tac_y_subset = labels[:, 3:]
tac_y_kernel_pls.fit(tac_X_subset, tac_y_subset)

vis_X_subset = vis_y_params[:, -1, :, 0]
vis_y_subset = labels[:, 0:3]
vis_y_kernel_pls.fit(vis_X_subset, vis_y_subset)

# Select a portion of the dataset to fit pose
vis_z_params_res = np.reshape(vis_z_params, (vis_z_params.shape[0] * vis_z_params.shape[1], vis_z_params.shape[2], vis_z_params.shape[3]))
pose_test_res = np.reshape(pose_test, (pose_test.shape[0] * pose_test.shape[1], pose_test.shape[2]))
num_select = int((percent / 100.0) * vis_z_params_res.shape[0])
# Randomly select indices
random_indices = np.random.choice(vis_z_params_res.shape[0], num_select, replace=False)
# Select data subset
X_subset = vis_z_params_res[random_indices, :, 0]  # or reshape if needed
Y_subset = pose_test_res[random_indices]

vis_z_kernel_pls.fit(X_subset, Y_subset)  # Align with ground truth pose, ideally if memory permits should be on the whole dataset

global_error_y_tac = []
global_error_y_vis = []
global_error_z_vis = []

# Select the test set
# Access the arrays from the .npz files
if method_name == 'joint' or method_name == 'baseline':
    y_params = test_data['y_params']
    vis_y_params = y_params[:, :, :16, :]
    tac_y_params = y_params[:, :, 16:, :]
else:
    vis_y_params = test_data['vis_y_params']
    tac_y_params = test_data['tac_y_params']

vis_z_params = test_data['vis_z_params']
labels_test = test_data['labels_test']
pose_test = test_data['pose_test']

# Regress over the entire trajectory
for obj_int in range(0, 10):
    print(obj_int, ' / ' + str(labels_test.shape[0]))
    # Extract object-specific parameters
    tac_mu = tac_y_params[obj_int, :, :, 0]
    tac_sigma = np.exp(tac_y_params[obj_int, :, :, 1])
    vis_mu = vis_y_params[obj_int, :, :, 0]
    vis_sigma = np.exp(vis_y_params[obj_int, :, :, 1])
    z_mu = vis_z_params[obj_int, :-1, :, 0]
    z_sigma = np.exp(vis_z_params[obj_int, :-1, :, 1])

    # Sample for all time steps at once: shape (time, samples, dim)
    tac_samples = np.random.normal(tac_mu[:, None, :], tac_sigma[:, None, :],
                                   size=(num_time_steps - 1, num_samples, tac_mu.shape[1]))
    vis_samples = np.random.normal(vis_mu[:, None, :], vis_sigma[:, None, :],
                                   size=(num_time_steps - 1, num_samples, vis_mu.shape[1]))
    z_samples = np.random.normal(z_mu[:, None, :], z_sigma[:, None, :],
                                 size=(num_time_steps - 1, num_samples, z_mu.shape[1]))

    # Reshape for batch regression
    tac_samples_flat = tac_samples.reshape(-1, tac_mu.shape[1])
    vis_samples_flat = vis_samples.reshape(-1, vis_mu.shape[1])
    z_samples_flat = z_samples.reshape(-1, z_mu.shape[1])

    # Predict
    tac_preds = tac_y_kernel_pls.predict(tac_samples_flat).reshape(num_time_steps - 1, num_samples, -1)
    vis_preds = vis_y_kernel_pls.predict(vis_samples_flat).reshape(num_time_steps - 1, num_samples, -1)
    z_preds = vis_z_kernel_pls.predict(z_samples_flat).reshape(num_time_steps - 1, num_samples, -1)

    # Compute mean and variance
    tac_mean = tac_preds.mean(axis=1)
    tac_var = tac_preds.var(axis=1)
    vis_mean = vis_preds.mean(axis=1)
    vis_var = vis_preds.var(axis=1)
    z_mean = z_preds.mean(axis=1)
    z_var = z_preds.var(axis=1)

    # Combine into parameter arrays
    trans_tac_y_params = np.stack([tac_mean, tac_var], axis=-1)
    trans_vis_y_params = np.stack([vis_mean, vis_var], axis=-1)
    trans_vis_z_params = np.stack([z_mean, z_var], axis=-1)

    # Compute errors
    global_error_y_tac.append((trans_tac_y_params[:, :, 0] - labels_test[obj_int, :98, 3:]) ** 2)
    global_error_y_vis.append((trans_vis_y_params[:, :, 0] - labels_test[obj_int, :98, :3]) ** 2)
    global_error_z_vis.append((trans_vis_z_params[:, :, 0] - pose_test[obj_int, :98, :]) ** 2)

global_error_y_vis = np.array(global_error_y_vis)
global_error_z_vis = np.array(global_error_z_vis)
global_error_y_tac = np.array(global_error_y_tac)

np.save(os.path.join('results', 'regression', method_name+'_reg_error_vis_y_' + condition_type + '.npy'), global_error_y_vis)
np.save(os.path.join('results', 'regression', method_name+'_reg_error_vis_z_' + condition_type + '.npy'), global_error_z_vis)
np.save(os.path.join('results', 'regression', method_name+'_reg_error_tac_y_' + condition_type + '.npy'), global_error_y_tac)
