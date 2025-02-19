# Code to find the normalization values on the whole dataset

import pickle
import os
import re
import numpy as np

def normalize(matrix):
    shape = matrix.shape
    axes_to_normalize = tuple(range(len(shape) - 1))

    # Compute min and max across these axes
    min_val = np.min(matrix, axis=axes_to_normalize, keepdims=True)
    max_val = np.max(matrix, axis=axes_to_normalize, keepdims=True)

    # Avoid division by zero
    denominator = max_val - min_val
    denominator[denominator == 0] = 1  # Prevent division by zero

    # Normalize the matrix between 0 and 1
    # norm_matrix = (matrix - min_val) / denominator
    return min_val, max_val

char_map = {'c': 1, 'e': 2, 's': 3, 'r': 4, 'h': 5}
db_path = 'dataset/cm_dataset/processed'
training_path = 'dataset/cm_dataset/training'

objects = [name for name in sorted(os.listdir(db_path))]

print('Num of objects found: ', len(objects))

vis_obs_np = []
tac_obs_np = []
action_np = []
gt_obs_np = []

for object_name in objects:
    print('Processing', object_name)
    file_name = os.path.join(db_path, object_name)
    with open(file_name, 'rb') as f:
        object_db = pickle.load(f)

    for interaction in range(len(object_db)):
        interaction_db = object_db[interaction]
        # action parameters - x, y, z, roll, pitch, yaw, gripper_distance, target_force, target, angular_velocity
        # action parameters - [99, 9]
        action_np.append(interaction_db['action'])
        # vis obs - [99, 128, 128, 2]
        vis_obs_np.append(interaction_db['vis_obs'])
        # tac obs - [99, 80, 80]
        tac_obs_np.append(interaction_db['tactile_obs'])
        # gt obs - [x,y,z,roll,pitch,yaw,shape,size,color,stiffness, mass, friction] # Time invariant and time varying parameters
        pose = interaction_db['pose']
        # gt_obs_np.append(interaction_db['pose'])
        match = re.match(r'([a-z])-([0-9]+)-([0-9]+)\.pickle', object_name)
        obj_shape, obj_size, obj_friction = match.groups()
        mapped_obj_shape = char_map.get(obj_shape)
        gt_obs = np.hstack([pose, np.ones((pose.shape[0], 1))*int(mapped_obj_shape),
                            np.ones((pose.shape[0], 1))*int(obj_size),
                            np.ones((pose.shape[0], 1))*int(obj_friction),
                            np.ones((pose.shape[0], 1))*int(mapped_obj_shape), # Note in the initial set there is a direct correlation between visual and haptic properties
                            np.ones((pose.shape[0], 1))*int(obj_size),
                            np.ones((pose.shape[0], 1))*int(obj_friction)])

        gt_obs_np.append(gt_obs)

vis_obs_np = np.array(vis_obs_np)
tac_obs_np = np.array(tac_obs_np)
action_np = np.array(action_np)
gt_obs_np = np.array(gt_obs_np)

# Compute min max values
# min_vis, max_vis = normalize(vis_obs_np)
# np.savez_compressed('vis_normalized.npz', min_vis=min_vis, max_vis=max_vis)

min_tac, max_tac = normalize(tac_obs_np)
np.savez_compressed('tac_normalized.npz', min_tac=min_tac, max_tac=max_tac)

