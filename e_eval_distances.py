import os
import numpy as np
import random
import distinctipy
from scipy.spatial.distance import pdist, squareform
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


def compute_distances(features, labels, distance_type='euclidean'):
    unique_classes = np.unique(labels)

    # Initialize the distance matrix
    distance_matrix = np.zeros((len(unique_classes), len(unique_classes)))

    # Compute intra-class distances
    for i, cls in enumerate(unique_classes):
        class_indices = np.where(labels == cls)[0]
        class_features = features[class_indices]

        if len(class_features) > 1:  # Ensure there are at least 2 samples
            distances = pdist(class_features, metric=distance_type)  # Pairwise distances
            intra_class_distance = np.mean(distances)  # Average intra-class distance
            distance_matrix[i, i] = intra_class_distance

    # Compute inter-class distances
    for i in range(len(unique_classes)):
        for j in range(i + 1, len(unique_classes)):
            class_i_features = features[labels == unique_classes[i]]
            class_j_features = features[labels == unique_classes[j]]

            # Compute pairwise distances between the two classes
            distances = pdist(np.vstack([class_i_features, class_j_features]), metric=distance_type)
            inter_class_distance = np.mean(distances)  # Average inter-class distance
            distance_matrix[i, j] = inter_class_distance
            distance_matrix[j, i] = inter_class_distance  # Symmetric matrix

    return distance_matrix, unique_classes

# Set the seed for reproducibility
random.seed(0)
np.random.seed(0)  # For numpy-bas

baseline_test = np.load(os.path.join('results', 'baseline', 'model_output', 'out_test_n0_c0.npz'))
baseline_val = np.load(os.path.join('results', 'baseline', 'model_output', 'out_val_n0_c0.npz'))
joint_test = np.load(os.path.join('results', 'joint', 'model_output', 'out_test_n0_c0.npz'))
joint_val = np.load(os.path.join('results', 'joint', 'model_output', 'out_val_n0_c0.npz'))
wcm_test = np.load(os.path.join('results', 'w-cm', 'late', 'model_output', 'out_test_n0_c0.npz'))
wcm_val = np.load(os.path.join('results', 'w-cm', 'late', 'model_output', 'out_test_n0_c0.npz'))
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

for name, features in feature_sets.items():
    print('Computing distances for ' + name)
    distance_matrix, unique_classes = compute_distances(features, object_labels, distance_type='euclidean')

    distance_matrix = distance_matrix / np.max(distance_matrix)

    unique_classes = unique_classes.astype(np.int32)
    plt.figure(figsize=(18, 15))
    # heatmap = sns.heatmap(distance_matrix, annot=False, cmap='coolwarm', xticklabels=unique_classes, yticklabels=unique_classes,  vmin=0, vmax=1.0, cbar=True)
    heatmap = sns.heatmap(distance_matrix, annot=False, cmap='coolwarm', xticklabels=unique_classes,
                          yticklabels=unique_classes, cbar=True)
    # Get number of classes and 8 evenly spaced indices
    num_classes = len(unique_classes)
    tick_indices = np.linspace(0, num_classes - 1, 2, dtype=int)
    tick_labels = unique_classes[tick_indices]

    # Shift tick positions to center of cells by adding 0.5
    centered_ticks = tick_indices + 0.5

    # Set ticks at center
    plt.xticks(ticks=centered_ticks, labels=tick_labels, fontsize=50)
    plt.yticks(ticks=centered_ticks, labels=tick_labels, fontsize=50)

    plt.gca().invert_yaxis()

    # Increase colorbar tick label size
    heatmap.collections[0].colorbar.ax.tick_params(labelsize=50)  # Adjust size as needed
    plt.savefig(os.path.join('results', 'euc_distances', name + '_distances.svg'), bbox_inches='tight', pad_inches=0)
