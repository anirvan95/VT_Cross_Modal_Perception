import pickle
import os
import numpy as np
from scipy.interpolate import interp1d
import imageio
import matplotlib.cm as cm

def compute_min_max(data):
    return np.min(data), np.max(data)


def apply_colormap(image, cmap='viridis'):
    colormap = cm.get_cmap(cmap)  # Get the colormap
    colored_image = colormap(image)[:, :, :3]*255  # Apply colormap and drop alpha channel
    colored_image = np.clip(colored_image, 0, 255).astype(np.uint8)
    return colored_image

def normalize_tactile_data(data, values, scale=1.0):
    min_val, max_val = values
    min_val*=scale
    max_val*=scale
    data = np.asarray(data, dtype=np.float32)
    return (data - min_val) / (max_val - min_val + 1e-8) # Avoid division by zero

def resample_filt(data, timestamps, new_timestamps):
    resampled_data = []
    resampled_data.append(data[0])
    # Iterate over each custom timestamp interval
    for i in range(1, len(new_timestamps)):
        # Define the start and end of the current interval
        start_time = new_timestamps[i - 1]
        end_time = new_timestamps[i]

        # Find indices where timestamps fall within the interval
        mask = (timestamps >= start_time) & (timestamps < end_time)

        # Extract values within the interval
        interval_values = data[mask]
        if len(interval_values) > 0:
            mean_value = np.mean(interval_values, axis=0)
            prev_mean = mean_value
        else:
            mean_value = prev_mean
        # Store the result
        resampled_data.append(mean_value)
    return np.array(resampled_data)


def process_tactile_data(tactile_data):
    tactile_ts = []
    fx, fy, fz = [], [], []
    tx, ty, tz = [], [], []
    pfx, pfy, pfz = [], [], []

    for i in range(len(tactile_data)):
        tactile_ts.append(tactile_data[i]['time'])
        fx.append(tactile_data[i]['force_fx'])
        fy.append(tactile_data[i]['force_fy'])
        fz.append(tactile_data[i]['force_fz'])
        tx.append(tactile_data[i]['torque_tx'])
        ty.append(tactile_data[i]['torque_ty'])
        tz.append(tactile_data[i]['torque_tz'])
        pfx.append(tactile_data[i]['pillar_forces_fx'])
        pfy.append(tactile_data[i]['pillar_forces_fy'])
        pfz.append(tactile_data[i]['pillar_forces_fz'])

    tactile_ts = np.array(tactile_ts) - tactile_ts[0]
    fx, fy, fz = np.array(fx), np.array(fy), np.array(fz)
    tx, ty, tz = np.array(tx), np.array(ty), np.array(tz)
    pfx, pfy, pfz = np.array(pfx), np.array(pfy), np.array(pfz)

    return tactile_ts, fx, fy, fz, tx, ty, tz, pfx, pfy, pfz

def resample(data, timestamps, new_timestamps):
    # interpolation
    f = interp1d(timestamps, data, axis=0, kind='cubic')
    resampled_data = f(new_timestamps)
    # print('resampled_data:', resampled_data.shape)
    return resampled_data

# Load the normalization file
with open('tactile_min_max_values.pickle', 'rb') as f:
    min_max_values = pickle.load(f)
f.close()
overall_min_max = {key: (np.min([x[0] for x in min_max_values[key]]), np.max([x[1] for x in min_max_values[key]])) for key in min_max_values}
interaction_ind = list(range(0, 48, 3))
keys = ['gt_object_pose', 'action', 'tactile_0', 'tactile_1', 'rgb', 'depth']
object_name = 'c-1-5'
object_path = os.path.join('/media/robotac_ws0/Elements/Cross_Modal_Dataset/'+object_name)
interactions = [name for name in sorted(os.listdir(object_path)) if os.path.isdir(os.path.join(object_path, name))]
for ind in interaction_ind:
    print('Processing {}'.format(interactions[ind]))
    file_name = os.path.join(object_path, interactions[ind], 'test.pickle')
    with open(file_name, 'rb') as f:
        db = pickle.load(f)
    f.close()

    robot_pose = db['action']
    tactile_0 = db['tactile_0']
    tactile_1 = db['tactile_1']

    tactile_0_ts, fx_0, fy_0, fz_0, tx_0, ty_0, tz_0, pfx_0, pfy_0, pfz_0 = process_tactile_data(tactile_0)
    tactile_1_ts, fx_1, fy_1, fz_1, tx_1, ty_1, tz_1, pfx_1, pfy_1, pfz_1 = process_tactile_data(tactile_1)

    # Normalise per-dimension
    fx_0 = normalize_tactile_data(fx_0, overall_min_max['fx_0'], 1.5)
    fx_1 = normalize_tactile_data(fx_1, overall_min_max['fx_1'], 1.5)
    fy_0 = normalize_tactile_data(fy_0, overall_min_max['fy_0'], 1.5)
    fy_1 = normalize_tactile_data(fy_1, overall_min_max['fy_1'], 1.5)
    fz_0 = normalize_tactile_data(fz_0, overall_min_max['fz_0'], 1.5)
    fz_1 = normalize_tactile_data(fz_1, overall_min_max['fz_1'], 1.5)
    tx_0 = normalize_tactile_data(tx_0, overall_min_max['tx_0'], 1.5)
    tx_1 = normalize_tactile_data(tx_1, overall_min_max['tx_1'], 1.5)
    ty_0 = normalize_tactile_data(ty_0, overall_min_max['ty_0'], 1.5)
    ty_1 = normalize_tactile_data(ty_1, overall_min_max['ty_1'], 1.5)

    pfx_0 = normalize_tactile_data(pfx_0, overall_min_max['pfx_0'], 1.5)
    pfx_1 = normalize_tactile_data(pfx_1, overall_min_max['pfx_1'], 1.5)
    pfy_0 = normalize_tactile_data(pfy_0, overall_min_max['pfy_0'], 1.5)
    pfy_1 = normalize_tactile_data(pfy_1, overall_min_max['pfy_1'], 1.5)
    pfz_0 = normalize_tactile_data(pfz_0, overall_min_max['pfz_0'], 1.5)
    pfz_1 = normalize_tactile_data(pfz_1, overall_min_max['pfz_1'], 1.5)


    # 3 channel tactile data
    tac_0 = len(fx_0)
    tac_1 = len(fx_1)
    if tac_0 > tac_1:
        tac = tac_1
        tactile_ts = tactile_1_ts
    else:
        tac = tac_0
        tactile_ts = tactile_0_ts
    orig_fs = len(tactile_ts)/(tactile_ts[-1]-tactile_ts[0])

    tactile_data_fx = np.concatenate([fx_0[:tac, None], fx_1[:tac, None], tx_0[:tac, None], tx_1[:tac, None], pfx_0[:tac, :], pfx_1[:tac, :]], axis=-1)
    tactile_data_fy = np.concatenate([fy_0[:tac, None], fy_1[:tac, None], ty_0[:tac, None], ty_1[:tac, None], pfy_0[:tac, :], pfy_1[:tac, :]], axis=-1)
    tactile_data_fz = np.concatenate([fz_0[:tac, None], fz_1[:tac, None], tz_0[:tac, None], tz_1[:tac, None], pfz_0[:tac, :], pfz_1[:tac, :]], axis=-1)

    # Sub-sample slightly to re-structure the tactile data
    target_fs = 240
    # assert orig_fs > target_fs
    target_fs_sub = 3
    num_samples = int(target_fs_sub * 33)

    target_ts = np.linspace(0, 33, target_fs*33)
    tac_fx_rs = resample_filt(tactile_data_fx, tactile_ts, target_ts)
    tac_fy_rs = resample_filt(tactile_data_fy, tactile_ts, target_ts)
    tac_fz_rs = resample_filt(tactile_data_fz, tactile_ts, target_ts)
    pad_dim = int(target_fs/target_fs_sub - tac_fx_rs.shape[1]*3)
    tac_obs = np.concatenate([tac_fx_rs, tac_fy_rs, tac_fz_rs, np.zeros([target_fs*33, pad_dim])], axis=-1)
    tac_obs = np.reshape(tac_obs, [target_fs_sub*33, 80, 80])

    tac_obs_gif_path = os.path.join('dump/tactile', object_name+'_'+str(ind)+'_tac.gif')
    # rescaled = (tac_obs + 1) * 127.5
    colored_frames = [apply_colormap(frame, 'viridis') for frame in tac_obs]
    imageio.mimsave(tac_obs_gif_path, colored_frames, fps=15)

'''
# Compute min max of each dimension of tactile
base_dir = '/media/robotac_ws0/Elements/Cross_Modal_Dataset'
objects = [name for name in sorted(os.listdir(base_dir)) if os.path.isdir(os.path.join(base_dir, name))]

# Initialize a dictionary to store min/max values for all objects
all_objects_min_max = {}
min_max_values = {
    'fx_0': [], 'fy_0': [], 'fz_0': [], 'tx_0': [], 'ty_0': [], 'tz_0': [],
    'pfx_0': [], 'pfy_0': [], 'pfz_0': [],
    'fx_1': [], 'fy_1': [], 'fz_1': [], 'tx_1': [], 'ty_1': [], 'tz_1': [],
    'pfx_1': [], 'pfy_1': [], 'pfz_1': []
}

print('Num of objects found: ', len(objects))
for object_name in objects:
    # Confirm if 75 objects are there
    print('Processing: ', object_name)
    object_path = os.path.join(base_dir, object_name)
    interactions = [name for name in sorted(os.listdir(object_path)) if os.path.isdir(os.path.join(object_path, name))]
    interaction = interactions[-1]
    file_name = os.path.join(object_path, interaction, 'test.pickle')
    with open(file_name, 'rb') as f:
        db = pickle.load(f)
    f.close()

    # Load object keys
    keys = ['gt_object_pose', 'action', 'tactile_0', 'tactile_1', 'rgb', 'depth']
    tactile_0 = db['tactile_0']
    tactile_1 = db['tactile_1']
    rgb = db['rgb']
    depth = db['depth']

    tactile_0_ts, fx_0, fy_0, fz_0, tx_0, ty_0, tz_0, pfx_0, pfy_0, pfz_0 = process_tactile_data(tactile_0)
    tactile_1_ts, fx_1, fy_1, fz_1, tx_1, ty_1, tz_1, pfx_1, pfy_1, pfz_1 = process_tactile_data(tactile_1)

    # Compute min/max for each variable and store them
    min_max_values['fx_0'].append(compute_min_max(fx_0))
    min_max_values['fy_0'].append(compute_min_max(fy_0))
    min_max_values['fz_0'].append(compute_min_max(fz_0))
    min_max_values['tx_0'].append(compute_min_max(tx_0))
    min_max_values['ty_0'].append(compute_min_max(ty_0))
    min_max_values['tz_0'].append(compute_min_max(tz_0))
    min_max_values['pfx_0'].append(compute_min_max(pfx_0))
    min_max_values['pfy_0'].append(compute_min_max(pfy_0))
    min_max_values['pfz_0'].append(compute_min_max(pfz_0))

    min_max_values['fx_1'].append(compute_min_max(fx_1))
    min_max_values['fy_1'].append(compute_min_max(fy_1))
    min_max_values['fz_1'].append(compute_min_max(fz_1))
    min_max_values['tx_1'].append(compute_min_max(tx_1))
    min_max_values['ty_1'].append(compute_min_max(ty_1))
    min_max_values['tz_1'].append(compute_min_max(tz_1))
    min_max_values['pfx_1'].append(compute_min_max(pfx_1))
    min_max_values['pfy_1'].append(compute_min_max(pfy_1))
    min_max_values['pfz_1'].append(compute_min_max(pfz_1))


# Save the min/max values for all objects as a pickle file
min_max_file = 'all_objects_min_max_values.pickle'
# Compute overall min and max for each variable for the current object
overall_min_max = {
    key: (np.min([x[0] for x in min_max_values[key]]), np.max([x[1] for x in min_max_values[key]]))
    for key in min_max_values}
print(f"Overall min/max values for {object_name}:", overall_min_max)

with open(min_max_file, 'wb') as f:
    pickle.dump(min_max_values, f)
'''