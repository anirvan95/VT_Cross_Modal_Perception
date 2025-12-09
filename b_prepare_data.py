import pickle
import os
import re
import numpy as np
import random

# Seeding for reproducibility
random.seed(0)
np.random.seed(0)


def normalize(matrix, min_val, max_val):
    denominator = max_val - min_val
    denominator[denominator == 0] = 1  # Prevent division by zero

    # Normalize the matrix between 0 and 1
    norm_matrix = (matrix - min_val) / denominator
    return norm_matrix


def process_interactions(object_db, interaction_indices, object_name, char_map):
    action_np = []
    vis_obs_np = []
    tac_obs_np = []
    gt_obs_np = []

    match = re.match(r'([a-z])-([0-9]+)-([0-9]+)\.pickle', object_name)
    if not match:
        raise ValueError(f"Filename {object_name} does not match expected pattern.")
    obj_shape, obj_size, obj_friction = match.groups()
    mapped_obj_shape = char_map.get(obj_shape)

    if mapped_obj_shape is None:
        raise ValueError(f"Object shape '{obj_shape}' not found in char_map.")

    for idx in interaction_indices:
        interaction_db = object_db[idx]

        # Extract and append data
        action_np.append(interaction_db['action'])  # Shape: [99, 9]
        vis_obs_np.append(interaction_db['vis_obs'])  # Shape: [99, 128, 128, 2]
        tac_obs_np.append(interaction_db['tactile_obs'])  # Shape: [99, 80, 80]

        pose = interaction_db['pose']  # Shape: [99, 6]
        physical_params = np.hstack([
            np.ones((pose.shape[0], 1)) * int(mapped_obj_shape),
            np.ones((pose.shape[0], 1)) * int(obj_size),
            np.ones((pose.shape[0], 1)) * int(obj_friction),
            np.ones((pose.shape[0], 1)) * int(mapped_obj_shape),
            np.ones((pose.shape[0], 1)) * int(obj_size),
            np.ones((pose.shape[0], 1)) * int(obj_friction),
        ])
        gt_obs = np.hstack([pose, physical_params])  # Final shape: [99, 12]
        gt_obs_np.append(gt_obs)

    # Convert to numpy arrays
    action_np = np.array(action_np)
    vis_obs_np = np.array(vis_obs_np)
    tac_obs_np = np.array(tac_obs_np)
    gt_obs_np = np.array(gt_obs_np)

    return action_np, vis_obs_np, tac_obs_np, gt_obs_np

char_map = {'c': 1, 'e': 2, 's': 3, 'r': 4, 'h': 5}
db_path = os.path.join('dataset', 'cm_dataset', 'processed')
save_path = os.path.join('dataset', 'cm_dataset')

os.makedirs(os.path.join(save_path, 'train_set'), exist_ok=True)
os.makedirs(os.path.join(save_path, 'val_set'), exist_ok=True)
os.makedirs(os.path.join(save_path, 'test_set'), exist_ok=True)

# Load the normalization data
vis_norm_data = np.load(os.path.join('results', 'vis_normalized.npz'))
min_viz = vis_norm_data['min_vis']
max_viz = vis_norm_data['max_vis']

# Load the normalization data
tac_norm_data = np.load(os.path.join('results', 'tac_normalized.npz'))
min_tac = tac_norm_data['min_tac']
max_tac = tac_norm_data['max_tac']

objects = [name for name in sorted(os.listdir(db_path))]

print('Num of objects found: ', len(objects))
overall_test_indices = []
base_counter = 0
for i, object_name in enumerate(objects):
    print('Processing', object_name)
    file_name = os.path.join(db_path, object_name)

    with open(file_name, 'rb') as f:
        object_db = pickle.load(f)

    indices = list(range(len(object_db)))  # Select 5 interactions out of 16
    random.shuffle(indices)
    testing_indices = indices[:5]
    remaining_indices = indices[5:]
    random.shuffle(remaining_indices)
    validation_indices = remaining_indices[:5]
    training_indices = remaining_indices
    random.shuffle(training_indices)

    # Train set
    action_np, vis_obs_np, tac_obs_np, gt_obs_np = process_interactions(object_db, training_indices, object_name, char_map)
    vis_obs_np = normalize(vis_obs_np, min_viz, max_viz)
    tac_obs_np = normalize(tac_obs_np, min_tac, max_tac)
    print('Saving data')
    train_file_path = os.path.join(save_path, 'train_set', 'training_' + str(i) + '.npz')
    np.savez(train_file_path,
             vis_obs=vis_obs_np.astype(np.float16),
             tac_obs=tac_obs_np.astype(np.float16),
             action=action_np.astype(np.float16),
             gt_obs=gt_obs_np.astype(np.float16))

    # Validation set
    action_np, vis_obs_np, tac_obs_np, gt_obs_np = process_interactions(object_db, validation_indices, object_name, char_map)
    vis_obs_np = normalize(vis_obs_np, min_viz, max_viz)
    tac_obs_np = normalize(tac_obs_np, min_tac, max_tac)
    val_file_path = os.path.join(save_path, 'val_set', 'validation_' + str(i) + '.npz')
    np.savez(val_file_path,
             vis_obs=vis_obs_np.astype(np.float16),
             tac_obs=tac_obs_np.astype(np.float16),
             action=action_np.astype(np.float16),
             gt_obs=gt_obs_np.astype(np.float16))

    # Test set
    action_np, vis_obs_np, tac_obs_np, gt_obs_np = process_interactions(object_db, testing_indices, object_name, char_map)
    vis_obs_np = normalize(vis_obs_np, min_viz, max_viz)
    tac_obs_np = normalize(tac_obs_np, min_tac, max_tac)
    test_file_path = os.path.join(save_path, 'test_set', 'testing_' + str(i) + '.npz')
    np.savez(test_file_path,
             vis_obs=vis_obs_np.astype(np.float16),
             tac_obs=tac_obs_np.astype(np.float16),
             action=action_np.astype(np.float16),
             gt_obs=gt_obs_np.astype(np.float16))
