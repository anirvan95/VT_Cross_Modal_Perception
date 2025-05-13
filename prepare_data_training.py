import pickle
import os
import re
import numpy as np

def normalize(matrix, min_val, max_val):
    denominator = max_val - min_val
    denominator[denominator == 0] = 1  # Prevent division by zero

    # Normalize the matrix between 0 and 1
    norm_matrix = (matrix - min_val) / denominator
    return norm_matrix

char_map = {'c': 1, 'e': 2, 's': 3, 'r': 4, 'h': 5}
db_path = 'dataset/cm_dataset/processed'
training_path = 'dataset/cm_dataset/training'
action_skip = 3 # check this

# Load the normalization data
vis_norm_data = np.load('vis_normalized.npz')
min_viz = vis_norm_data['min_vis']
max_viz = vis_norm_data['max_vis']

# Load the normalization data
tac_norm_data = np.load('tac_normalized.npz')
min_tac = tac_norm_data['min_tac']
max_tac = tac_norm_data['max_tac']

objects = [name for name in sorted(os.listdir(db_path))]

print('Num of objects found: ', len(objects))

for i, object_name in enumerate(objects):
    print('Processing', object_name)
    file_name = os.path.join(db_path, object_name)
    vis_obs_np = []
    tac_obs_np = []
    action_np = []
    gt_obs_np = []

    with open(file_name, 'rb') as f:
        object_db = pickle.load(f)

    for interaction in range(0, len(object_db), action_skip):
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
    action_torch = np.array(action_np)
    gt_obs_torch = np.array(gt_obs_np)

    vis_obs_torch = normalize(vis_obs_np, min_viz, max_viz)
    tac_obs_torch = normalize(tac_obs_np, min_tac, max_tac)
    print('Saving data')
    train_file_path = os.path.join(save_path, 'training_'+str(i)+'.npz')
    np.savez(train_file_path,
                            vis_obs=vis_obs_torch.astype(np.float16),
                            tac_obs=tac_obs_torch.astype(np.float16),
                            action=action_torch.astype(np.float16),
                            gt_obs=gt_obs_torch.astype(np.float16))
