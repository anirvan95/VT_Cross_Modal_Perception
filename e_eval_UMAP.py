import os
import numpy as np
import random
import distinctipy
import umap
import seaborn as sns
import matplotlib
import re

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
import matplotlib.pyplot as plt
params = {'text.usetex': True,
          'font.size': 25,
          'font.family': 'lmodern',
          }
plt.rcParams.update(params)

# Set the seed for reproducibility
random.seed(0)
np.random.seed(0)  # For numpy-bas

baseline_test = np.load(os.path.join('results', 'baseline', 'model_output', 'out_test_n0_c0.npz'))
baseline_val = np.load(os.path.join('results', 'baseline', 'model_output', 'out_val_n0_c0.npz'))
joint_test = np.load(os.path.join('results', 'joint', 'model_output', 'out_test_n0_c0.npz'))
joint_val = np.load(os.path.join('results', 'joint', 'model_output', 'out_val_n0_c0.npz'))
wcm_test = np.load(os.path.join('results', 'w-cm', 'late', 'model_output', 'out_test_n0_c0.npz'))
wcm_val = np.load(os.path.join('results', 'w-cm', 'late', 'model_output', 'out_val_n0_c0.npz'))
wocm_test = np.load(os.path.join('results', 'wo-cm', 'model_output', 'out_test_n0_c0.npz'))
wocm_val = np.load(os.path.join('results', 'wo-cm', 'model_output', 'out_val_n0_c0.npz'))

# Merge test and val for baseline
baseline_y_params = np.concatenate([baseline_test['y_params'], baseline_val['y_params']])
baseline_labels_test = np.concatenate([baseline_test['labels_test'], baseline_val['labels_test']])
baseline_labels = baseline_labels_test[:, -1, :]
baseline_labels_np = ((baseline_labels[:, 0] - 1) * 15 + (baseline_labels[:, 1] - 1) * 5 + (baseline_labels[:, 2] - 1))
baseline_features_np = baseline_y_params[:, -1, :, 0]

# Merge test and val for joint
joint_y_params = np.concatenate([joint_test['y_params'], joint_val['y_params']])
joint_labels_test = np.concatenate([joint_test['labels_test'], joint_val['labels_test']])
joint_labels = joint_labels_test[:, -1, :]
joint_labels_np = ((joint_labels[:, 0] - 1) * 15 + (joint_labels[:, 1] - 1) * 5 + (joint_labels[:, 2] - 1))
joint_features_np = joint_y_params[:, -1, :, 0]

# Merge test and val for wcm (with cross-modal)
wcm_vis_y_params = np.concatenate([wcm_test['vis_y_params'], wcm_val['vis_y_params']])
wcm_tac_y_params = np.concatenate([wcm_test['tac_y_params'], wcm_val['tac_y_params']])
wcm_labels_test = np.concatenate([wcm_test['labels_test'], wcm_val['labels_test']])
wcm_labels = wcm_labels_test[:, -1, :]
wcm_labels_np = ((wcm_labels[:, 0] - 1) * 15 + (wcm_labels[:, 1] - 1) * 5 + (wcm_labels[:, 2] - 1))
wcm_vis_features_np = wcm_vis_y_params[:, -1, :, 0]
wcm_tac_features_np = wcm_tac_y_params[:, -1, :, 0]
wcm_features_np = np.concatenate([wcm_vis_features_np, wcm_tac_features_np], axis=1)

# Merge test and val for wocm (without cross-modal)
wocm_vis_y_params = np.concatenate([wocm_test['vis_y_params'], wocm_val['vis_y_params']])
wocm_tac_y_params = np.concatenate([wocm_test['tac_y_params'], wocm_val['tac_y_params']])
wocm_labels_test = np.concatenate([wocm_test['labels_test'], wocm_val['labels_test']])
wocm_labels = wocm_labels_test[:, -1, :]
wocm_labels_np = ((wocm_labels[:, 0] - 1) * 15 + (wocm_labels[:, 1] - 1) * 5 + (wocm_labels[:, 2] - 1))
wocm_vis_features_np = wocm_vis_y_params[:, -1, :, 0]
wocm_tac_features_np = wocm_tac_y_params[:, -1, :, 0]
wocm_features_np = np.concatenate([wocm_vis_features_np, wocm_tac_features_np], axis=1)


feature_sets = {"baseline": baseline_features_np,
                "joint": joint_features_np,
                "wcm_vision": wcm_vis_features_np,
                "wcm_tactile": wcm_tac_features_np,
                "wcm_joint": wcm_features_np,
                "wocm_vision": wocm_vis_features_np,
                "wocm_tactile": wocm_tac_features_np,
                "wocm_joint": wocm_features_np}

object_labels = wcm_labels_np

colors = distinctipy.get_colors(75)
# distinctipy.color_swatch(colors)

color_vals = []
for i in range(object_labels.shape[0]):
    object_label = object_labels[i]
    col = colors[int(object_label)]
    col_rgb = np.array([col[0], col[1], col[2]])
    color_vals.append(col_rgb)

color_vals = np.array(color_vals)

for name, features in feature_sets.items():
    print('Computing UMAP projection for ' + name)
    y_embedded = umap.UMAP(n_components=3, min_dist=0.1, n_neighbors=450, random_state=0).fit_transform(features)
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(y_embedded[:, 0], y_embedded[:, 1], y_embedded[:, 2], c=color_vals, s=150)
    ax.view_init(elev=15, azim=45)
    ax.grid(True, alpha=0.01)
    # Compute ticks: round(min, median, max) to integers
    xticks = [int(round(np.min(y_embedded[:, 0]))),
              int(round(np.max(y_embedded[:, 0])))]

    yticks = [int(round(np.min(y_embedded[:, 1]))),
              int(round(np.max(y_embedded[:, 1])))]

    zticks = [int(round(np.min(y_embedded[:, 2]))),
              int(round(np.max(y_embedded[:, 2])))]

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_zticks(zticks)

    ax.tick_params(axis='both', labelsize=90, length=15, width=3)
    ax.tick_params(axis='z', labelsize=90, length=15, width=3)
    # plt.show()
    plt.savefig(os.path.join('results', 'umaps', name + '_umaps.svg'), bbox_inches='tight', pad_inches=0)