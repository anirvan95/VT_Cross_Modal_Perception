import pickle
import os
import sys
import cv2
import matplotlib
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import copy
import time
import imageio

params = {'text.usetex': True,
          'font.family': 'lmodern',
          'font.size': 18,
          }
plt.rcParams.update(params)

def parse_string(s):
    """
    Parse a formatted string into a dictionary.
    Parameters:
        s (str): Input string with key-value pairs separated by '=' and '-'.
    Returns:
        dict: Parsed key-value pairs.
    """
    pairs = s.split('-')  # Split by '-'
    result = {}
    for pair in pairs:
        key, value = pair.split('=')  # Split by '='
        # Try to convert value to float if possible, else keep as string
        try:
            value = float(value)
        except ValueError:
            pass  # Keep as string if conversion fails
        result[key] = value
    return result

def apply_colormap(image, cmap='viridis'):
    colormap = matplotlib.colormaps[cmap]  # Get the colormap
    colored_image = colormap(image)[:, :, :3]*255  # Apply colormap and drop alpha channel
    colored_image = np.clip(colored_image, 0, 255).astype(np.uint8)
    return colored_image

def normalize_tactile_data(data, values, scale=1.0):
    """
    Function to normalize the tactile data by passing a global min max along each dimension
    Note, this is only used for visualization and verification
    :param data:
    :param method:
    :return:
    """
    min_val, max_val = values
    min_val*=scale
    max_val*=scale
    data = np.asarray(data, dtype=np.float32)
    return (data - min_val) / (max_val - min_val + 1e-8) # Avoid division by zero

def fix_depth(data):
    """
    Function to fix depth map, remove nans to 0.0
    :param data:
    :return:
    """
    data_no_nan = np.nan_to_num(data, nan=0.0)
    data_clipped = np.clip(data_no_nan, 0, 1)  # 1 meters max based on the ZED camera configuration
    return data_clipped*255


def resample_filt(data, timestamps, new_timestamps):
    """
    Function to resampling the data using mean filtering approach, only works for downsampling
    :param data:
    :param timestamps:
    :param new_timestamps:
    :return:
    """
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

def resample(data, timestamps, new_timestamps):
    """
    General function to resample data via interpolation
    :param data:
    :param timestamps:
    :param new_timestamps:
    :return:
    """
    # interpolation
    f = interp1d(timestamps, data, axis=0, kind='cubic')
    resampled_data = f(new_timestamps)
    return resampled_data


def process_tactile_data(tactile_data):
    """
    Function to process tactile data
    :param tactile_data:
    :return:
    """
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


def pose_to_se3(pose):
    """
    Function to convert pose - x,y,z and euler angles to SE3 Matrix
    :param pose:
    :return:
    """
    translation = pose[:3]
    euler_angles = pose[3:]
    rotation_matrix = R.from_euler('xyz', euler_angles).as_matrix()
    se3_matrix = np.eye(4)
    se3_matrix[:3, :3] = rotation_matrix
    se3_matrix[:3, 3] = translation
    return se3_matrix
########################################################################################################################
base_dir = '/media/robotac_ws0/Elements/Cross_Modal_Dataset'
vis_dump_dir = 'dataset/cm_dataset/dump/vision'
tactile_dump_dir = 'dataset/cm_dataset/dump/tactile'
stats_dump_dir = 'dataset/cm_dataset/dump/stats'
if not os.path.exists(vis_dump_dir):
    os.makedirs(vis_dump_dir)
if not os.path.exists(tactile_dump_dir):
    os.makedirs(tactile_dump_dir)
if not os.path.exists(stats_dump_dir):
    os.makedirs(stats_dump_dir)

save_dir = 'dataset/cm_dataset/preprocessed'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

objects = [name for name in sorted(os.listdir(base_dir)) if os.path.isdir(os.path.join(base_dir, name))]
# Confirm if 75 objects are there
print('Num of objects found: ', len(objects))

######################################## Parameters of the Processed Dataset #########################################
target_fs = 3  # Hz sampling frequency
target_fs_tactile = 240 # Hz Tactile information kept at higher frequency to retain the spatial and vibrational component
start_time = 0 # s
end_time = 33 # s
horizon = end_time - start_time
target_ts = np.linspace(start_time, end_time, target_fs*horizon)
num_samples = int(target_fs * (end_time - start_time)) # 90 samples for each sequence
dim_tactile = 80 # Fixed, cannot be changed easily
dim_vis = 128
generate_plots = True

intrinsic_np = np.array([[173.9099, 0.0, 214.7523],
                         [0.0, 173.7742156982422, 119.35607147216797],
                         [0.0, 0.0, 1.0]]) # Camera parameters
intrinsic = o3d.core.Tensor(intrinsic_np)

# Load the normalization file
with open('tactile_min_max_values.pickle', 'rb') as f:
    min_max_values = pickle.load(f)

tactile_min_max = {key: (np.min([x[0] for x in min_max_values[key]]), np.max([x[1] for x in min_max_values[key]])) for key in min_max_values}

for object_name in objects:
    object_db = []
    print('Processing: ', object_name)
    object_path = os.path.join(base_dir, object_name)
    interactions = [name for name in sorted(os.listdir(object_path)) if os.path.isdir(os.path.join(object_path, name))]
    print('Num of interactions found: ', len(interactions))
    for ind, interaction in enumerate(interactions):
        process_start = time.time()
        file_name = os.path.join(object_path, interaction, 'test.pickle')
        with open(file_name, 'rb') as f:
            db = pickle.load(f)
        # Load object keys
        keys = ['gt_object_pose', 'action', 'tactile_0', 'tactile_1', 'rgb', 'depth']
        init_pound_cloud = db['init_point_cloud']
        camera_trans = init_pound_cloud['trans']
        camera_rot = init_pound_cloud['rot']
        object_pose = db['gt_object_pose']
        robot_pose = db['action']
        tactile_0 = db['tactile_0']
        tactile_1 = db['tactile_1']
        rgb = db['rgb']
        depth = db['depth']

        object_label = parse_string(interaction)

        ############################################### Process Tactile ###############################################
        # Dimension - [Seq_length, 80, 80]
        tactile_0_ts, fx_0, fy_0, fz_0, tx_0, ty_0, tz_0, pfx_0, pfy_0, pfz_0 = process_tactile_data(tactile_0)
        tactile_1_ts, fx_1, fy_1, fz_1, tx_1, ty_1, tz_1, pfx_1, pfy_1, pfz_1 = process_tactile_data(tactile_1)

        # Normalise per-dimension
        fx_0_norm = normalize_tactile_data(fx_0, tactile_min_max['fx_0'], 1.5)
        fx_1_norm = normalize_tactile_data(fx_1, tactile_min_max['fx_1'], 1.5)
        fy_0_norm = normalize_tactile_data(fy_0, tactile_min_max['fy_0'], 1.5)
        fy_1_norm = normalize_tactile_data(fy_1, tactile_min_max['fy_1'], 1.5)
        fz_0_norm = normalize_tactile_data(fz_0, tactile_min_max['fz_0'], 1.5)
        fz_1_norm = normalize_tactile_data(fz_1, tactile_min_max['fz_1'], 1.5)
        tx_0_norm = normalize_tactile_data(tx_0, tactile_min_max['tx_0'], 1.5)
        tx_1_norm = normalize_tactile_data(tx_1, tactile_min_max['tx_1'], 1.5)
        ty_0_norm = normalize_tactile_data(ty_0, tactile_min_max['ty_0'], 1.5)
        ty_1_norm = normalize_tactile_data(ty_1, tactile_min_max['ty_1'], 1.5)
        tz_0_norm = normalize_tactile_data(tz_0, tactile_min_max['tz_0'], 1.5)
        tz_1_norm = normalize_tactile_data(tz_1, tactile_min_max['tz_1'], 1.5)

        pfx_0_norm = normalize_tactile_data(pfx_0, tactile_min_max['pfx_0'], 1.5)
        pfx_1_norm = normalize_tactile_data(pfx_1, tactile_min_max['pfx_1'], 1.5)
        pfy_0_norm = normalize_tactile_data(pfy_0, tactile_min_max['pfy_0'], 1.5)
        pfy_1_norm = normalize_tactile_data(pfy_1, tactile_min_max['pfy_1'], 1.5)
        pfz_0_norm = normalize_tactile_data(pfz_0, tactile_min_max['pfz_0'], 1.5)
        pfz_1_norm = normalize_tactile_data(pfz_1, tactile_min_max['pfz_1'], 1.5)

        # Check which tactile was shorter
        if len(fx_0) > len(fx_1):
            dim_f = len(fx_1)
            tactile_ts = tactile_1_ts - tactile_1_ts[0]
        else:
            dim_f = len(fx_0)
            tactile_ts = tactile_0_ts - tactile_0_ts[0]

        orig_fs = len(tactile_ts) / (tactile_ts[-1] - tactile_ts[0])

        if orig_fs < 240:
            print('Warning !! orig_fs is less than 240, repeated data in tactile-  ', 240-orig_fs)

        tactile_data_fx = np.concatenate(
            [fx_0[:dim_f, None], fx_1[:dim_f, None],
             pfx_0[:dim_f, :], pfx_1[:dim_f, :], tx_0[:dim_f, None], tx_1[:dim_f, None]], axis=-1)
        tactile_data_fy = np.concatenate(
            [fy_0[:dim_f, None], fy_1[:dim_f, None],
             pfy_0[:dim_f, :], pfy_1[:dim_f, :], ty_0[:dim_f, None], ty_1[:dim_f, None]], axis=-1)
        tactile_data_fz = np.concatenate(
            [fz_0[:dim_f, None], fz_1[:dim_f, None],
             pfz_0[:dim_f, :], pfz_1[:dim_f, :], tz_0[:dim_f, None], tz_1[:dim_f, None]], axis=-1)

        tactile_data_fx_norm = np.concatenate(
            [fx_0_norm[:dim_f, None], fx_1_norm[:dim_f, None],
             pfx_0_norm[:dim_f, :], pfx_1_norm[:dim_f, :], tx_0_norm[:dim_f, None], tx_1_norm[:dim_f, None]], axis=-1)
        tactile_data_fy_norm = np.concatenate(
            [fy_0_norm[:dim_f, None], fy_1_norm[:dim_f, None],
             pfy_0_norm[:dim_f, :], pfy_1_norm[:dim_f, :], ty_0_norm[:dim_f, None], ty_1_norm[:dim_f, None]], axis=-1)
        tactile_data_fz_norm = np.concatenate(
            [fz_0_norm[:dim_f, None], fz_1_norm[:dim_f, None],
             pfz_0_norm[:dim_f, :], pfz_1_norm[:dim_f, :], tz_0_norm[:dim_f, None], tz_1_norm[:dim_f, None]], axis=-1)


        # Resample tactile data
        target_tactile_ts = np.linspace(start_time, end_time, target_fs_tactile * horizon)
        tac_fx_rs = resample_filt(tactile_data_fx, tactile_ts, target_tactile_ts)
        tac_fy_rs = resample_filt(tactile_data_fy, tactile_ts, target_tactile_ts)
        tac_fz_rs = resample_filt(tactile_data_fz, tactile_ts, target_tactile_ts)

        tac_fx_norm_rs = resample_filt(tactile_data_fx_norm, tactile_ts, target_tactile_ts)
        tac_fy_norm_rs = resample_filt(tactile_data_fy_norm, tactile_ts, target_tactile_ts)
        tac_fz_norm_rs = resample_filt(tactile_data_fz_norm, tactile_ts, target_tactile_ts)


        pad_dim = int(target_fs_tactile / target_fs - tac_fx_rs.shape[1]*3) # 3 is for the fx, fy, fz combination
        tac_obs = np.concatenate([tac_fx_rs, tac_fy_rs, tac_fz_rs, np.zeros([target_fs_tactile * horizon, pad_dim])], axis=-1)
        tac_obs = np.reshape(tac_obs, [target_fs * horizon, dim_tactile, dim_tactile])

        tac_obs_norm = np.concatenate([tac_fx_norm_rs, tac_fy_norm_rs, tac_fz_norm_rs, np.zeros([target_fs_tactile * horizon, pad_dim])], axis=-1)
        tac_obs_norm = np.reshape(tac_obs_norm, [target_fs * horizon, dim_tactile, dim_tactile])


        ############################################### Process Vision ###############################################
        # Load GT object
        object_name_gt = object_label['object']
        gt_object_path = os.path.join('dataset', 'gt_objects', str(object_name_gt + '.stl'))
        try:
            mesh = o3d.io.read_triangle_mesh(gt_object_path)
            if mesh.is_empty():
                raise ValueError(f"Failed to load mesh: {gt_object_path}")
        except Exception as e:
            print('Error: ', e)
            sys.exit(1)

        mesh.scale(0.00105, center=(0, 0, 0)) # meters # scale slightly higher to account for errors
        gt_pcd = mesh.sample_points_poisson_disk(100000)
        gt_pcd = gt_pcd.paint_uniform_color([1, 0, 0])

        rgb_data = np.array([entry['data'][0] for entry in rgb if 'data' in entry and len(entry['data']) > 0])
        rgb_data[..., [0, 2]] = rgb_data[..., [2, 0]]  # Flip the RGB channels based on ROS convention
        rgb_ts = np.array([entry['time'] for entry in rgb])
        rgb_ts = rgb_ts - rgb_ts[0]
        depth_data = np.array([entry['data'][0] for entry in depth if 'data' in entry and len(entry['data']) > 0])
        depth_ts = np.array([entry['time'] for entry in depth])
        depth_ts = depth_ts - depth_ts[0]
        object_pose_data = np.array([entry['data'] for entry in object_pose if 'data' in entry and len(entry['data']) > 0])
        object_pose_ts = np.array([entry['time'] for entry in object_pose])
        object_pose_ts = object_pose_ts - object_pose_ts[0]

        # Resample RGB-D data and Object pose data
        object_pose_rs = resample_filt(object_pose_data, object_pose_ts, target_ts)
        rgb_arr_rs = resample_filt(rgb_data, rgb_ts, target_ts)
        depth_arr_rs = resample_filt(fix_depth(depth_data), depth_ts, target_ts)

        # Perform segmentation
        camera_r = R.from_quat(camera_rot)
        camera_rot_mat = camera_r.as_matrix()
        optical_transform = np.array(
            [[camera_rot_mat[0, 0], camera_rot_mat[0, 1], camera_rot_mat[0, 2], camera_trans[0]],
             [camera_rot_mat[1, 0], camera_rot_mat[1, 1], camera_rot_mat[1, 2], camera_trans[1]],
             [camera_rot_mat[2, 0], camera_rot_mat[2, 1], camera_rot_mat[2, 2], camera_trans[2]],
             [0.0, 0.0, 0.0, 1.0]])
        extrinsic = o3d.core.Tensor(optical_transform)

        vis_obs = []
        rgb_obs = []
        depth_obs = []

        pcd_copy = copy.deepcopy(gt_pcd)
        for i in range(0, len(object_pose_rs)):
            H_transform = pose_to_se3(object_pose_rs[i])
            transformed_points = np.asarray(gt_pcd.points) @ H_transform[:3, :3].T + H_transform[:3, 3]
            pcd_copy.points = o3d.utility.Vector3dVector(transformed_points)
            pcd = o3d.t.geometry.PointCloud(o3d.core.Device('CPU:0'))
            pcd.point["positions"] = o3d.core.Tensor(np.asarray(pcd_copy.points), o3d.core.float32, o3d.core.Device('CPU:0'))
            pcd.point["colors"] = o3d.core.Tensor(np.asarray(pcd_copy.colors), o3d.core.float32, o3d.core.Device('CPU:0'))
            rgbd_reproj = pcd.project_to_rgbd_image(427,
                                                    240,
                                                    intrinsic,
                                                    extrinsic,
                                                    depth_scale=1,
                                                    depth_max=2.0)
            projected_rgb = np.asarray(rgbd_reproj.color)
            # Create segmentation mask
            mask = np.any(projected_rgb > 0, axis=-1)  # Non-zero pixels
            mask = mask[0:224, 90:314]  # Fixed cropping
            rgb_frame = copy.deepcopy(rgb_arr_rs[i])
            depth_frame = copy.deepcopy(depth_arr_rs[i])
            rgb_frame = rgb_frame[0:224, 90:314, 0:3].astype(np.uint8)
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
            depth_frame = depth_frame[0:224, 90:314]

            segmented_gray = np.zeros([224, 224], dtype=np.uint8)  # Black background
            segmented_depth = np.zeros([224, 224], dtype=np.uint8)

            # Apply mask
            segmented_gray[mask] = gray[mask]
            segmented_depth[mask] = depth_frame[mask]

            # Resize
            resized_gray_arr = cv2.resize(segmented_gray, (dim_vis, dim_vis), interpolation=cv2.INTER_CUBIC)
            resized_depth_arr = cv2.resize(segmented_depth, (dim_vis, dim_vis), interpolation=cv2.INTER_CUBIC)

            # Compute the combined visual observation
            vis_obs_frame = np.concatenate([resized_gray_arr[:, :, None], resized_depth_arr[:, :, None]], axis=-1)

            # Store results
            vis_obs.append(vis_obs_frame)
            rgb_obs.append(rgb_frame)
            depth_obs.append(depth_frame)

        vis_obs = np.array(vis_obs)
        rgb_obs = np.array(rgb_obs)
        depth_obs = np.array(depth_obs)

        # ########################################### Process Robot Action  ############################################
        # action parameters - x, y, z, roll, pitch, yaw, gripper_distance, target_force, target, angular_velocity
        robot_pose_data = np.array([entry['data'] for entry in robot_pose if 'data' in entry and len(entry['data']) > 0])
        action_ts = np.array([entry['time'] for entry in robot_pose])
        planning_frame = np.hstack([robot_pose_data[:, 0:3], robot_pose_data[:, 9:12]])
        left_finger = np.hstack([robot_pose_data[:, 3:6], robot_pose_data[:, 12:15]])
        right_finger = np.hstack([robot_pose_data[:, 6:9], robot_pose_data[:, 15:18]])

        action_obs_arr = []

        for i in range(len(planning_frame)):
            current_planning_frame_se3 = pose_to_se3(planning_frame[i])
            current_left_finger_se3 = pose_to_se3(left_finger[i])
            current_right_finger_se3 = pose_to_se3(right_finger[i])
            relative_left_finger_se3 = np.linalg.inv(current_planning_frame_se3) @ current_left_finger_se3
            relative_right_finger_se3 = np.linalg.inv(current_planning_frame_se3) @ current_right_finger_se3
            current_gripper_width = abs(relative_left_finger_se3[0, 3] - relative_right_finger_se3[0, 3])
            action_data = np.hstack([planning_frame[i], current_gripper_width, object_label['grasp_dist'], object_label['angular_vel']])
            action_obs_arr.append(action_data)

        # Resample
        action_ts = action_ts - action_ts[0]
        action_obs_arr = np.array(action_obs_arr)
        action_obs = resample_filt(action_obs_arr, action_ts, target_ts)

        # ########################################## Generate Plots & GIFs  ############################################
        if generate_plots:
            tac_obs_gif_path = os.path.join(tactile_dump_dir, object_name+'_'+str(ind)+'.gif')
            tactile_colored_frames = [apply_colormap(frame, 'viridis') for frame in tac_obs_norm]
            imageio.mimsave(tac_obs_gif_path, tactile_colored_frames, fps=9)

            rgb_obs_gif_path = os.path.join(vis_dump_dir, object_name+'_'+str(ind)+ '_rgb.gif')
            depth_obs_gif_path = os.path.join(vis_dump_dir, object_name+'_'+str(ind) + '_depth.gif')
            vis_obs_gif_path = os.path.join(vis_dump_dir, object_name+'_'+str(ind) + '_vis.gif')
            rgb_obs_gif = rgb_obs.astype(np.uint8)
            depth_obs_gif = depth_obs.astype(np.uint8)
            vis_obs_gif = vis_obs.astype(np.uint8)
            imageio.mimsave(rgb_obs_gif_path, rgb_obs_gif, fps=15)
            imageio.mimsave(depth_obs_gif_path, depth_obs_gif, fps=15)
            imageio.mimsave(vis_obs_gif_path, vis_obs_gif, fps=15)

            fig, ax = plt.subplots(2, 3, figsize=(12, 7))
            ax[0, 0].plot(target_ts, object_pose_rs[:, 0], 'r', label="x")  # x
            ax[0, 0].plot(target_ts, object_pose_rs[:, 1], 'g', label="y")  # y
            ax[0, 0].plot(target_ts, object_pose_rs[:, 2], 'b', label="z")  # z
            ax[0, 0].set_xlabel(r"Time ($\mathrm{s}$)")
            ax[0, 0].set_ylabel(r"($\mathrm{m}$)")
            ax[0, 0].set_title("Object Position")

            ax[0, 1].plot(target_ts, object_pose_rs[:, 3], 'r', label="x")  # roll
            ax[0, 1].plot(target_ts, object_pose_rs[:, 4], 'g', label="y")  # pitch
            ax[0, 1].plot(target_ts, object_pose_rs[:, 5], 'b', label="z")  # yaw
            ax[0, 1].set_xlabel(r"Time ($\mathrm{s}$)")
            ax[0, 1].set_ylabel(r"($\mathrm{rad}$)")
            ax[0, 1].set_title("Object Angle")

            ax[0, 2].plot(target_ts, action_obs[:, 0], 'r', label="x")  # x
            ax[0, 2].plot(target_ts, action_obs[:, 1], 'g', label="y")  # y
            ax[0, 2].plot(target_ts, action_obs[:, 2], 'b', label="z")  # z
            ax[0, 2].plot(target_ts, action_obs[:, 6], 'k', label="gw") # gripper width
            ax[0, 2].set_xlabel(r"Time ($\mathrm{s}$)")
            ax[0, 2].set_ylabel(r"($\mathrm{m}$)")
            ax[0, 2].set_title("Robot Position")

            ax[1, 0].plot(target_ts, action_obs[:, 3], 'r', label="x")  # roll
            ax[1, 0].plot(target_ts, action_obs[:, 4], 'g', label="y")  # pitch
            ax[1, 0].plot(target_ts, action_obs[:, 5], 'b', label="z")  # yaw
            ax[1, 0].set_xlabel(r"Time ($\mathrm{s}$)")
            ax[1, 0].set_ylabel(r"($\mathrm{rad}$)")
            ax[1, 0].set_title("Robot Angle")

            temp_fx = (abs(tac_obs[:, :, 0]) + abs(tac_obs[:, :, 1]))/2
            temp_fx = np.mean(temp_fx, axis=-1)
            temp_fy = (abs(tac_obs[:, :, 22]) + abs(tac_obs[:, :, 23]))/2
            temp_fy = np.mean(temp_fy, axis=-1)
            temp_fz = (abs(tac_obs[:, :, 44]) + abs(tac_obs[:, :, 45]))/2
            temp_fz = np.mean(temp_fz, axis=-1)

            temp_tx = (abs(tac_obs[:, :, 2]) + abs(tac_obs[:, :, 3]))
            temp_tx = np.mean(temp_tx, axis=-1)
            temp_ty = abs(tac_obs[:, :, 24]) + abs(tac_obs[:, :, 25])
            temp_ty = np.mean(temp_ty, axis=-1)
            temp_tz = abs(tac_obs[:, :, 46]) + abs(tac_obs[:, :, 47])
            temp_tz = np.mean(temp_tz, axis=-1)

            ax[1, 1].plot(target_ts, temp_fx, '.r', label="x")  # fx
            ax[1, 1].plot(target_ts, temp_fy, '.g', label="y")  # fy
            ax[1, 1].plot(target_ts, temp_fz, '.b', label="z")  # fz
            ax[1, 1].set_xlabel(r"Time ($\mathrm{s}$)")
            ax[1, 1].set_ylabel(r"($\mathrm{m}$)")
            ax[1, 1].set_title("Mean Force")

            ax[1, 2].plot(target_ts, temp_tx, '.r', label="x")  # fx
            ax[1, 2].plot(target_ts, temp_ty, '.g', label="y")  # fy
            ax[1, 2].plot(target_ts, temp_tz, '.b', label="z")  # fz
            ax[1, 2].set_xlabel(r"Time ($\mathrm{s}$)")
            ax[1, 2].set_ylabel(r"($\mathrm{m}$)")
            ax[1, 2].set_title("Mean Torque")
            plt.tight_layout()
            fig.savefig(os.path.join(stats_dump_dir, object_name+'_'+str(ind)+'.png'))
            plt.close()

        process_end = time.time()
        print('Done: ', interaction)
        print('Time taken: ', process_end - process_start)
        interaction_db = {'vis_obs': vis_obs, 'tactile_obs': tac_obs, 'action': action_obs, 'pose': object_pose_rs}
        object_db.append(interaction_db)

    # Initiate Saving, add labels for shape-stiffness, color-friction, size-weight
    save_path = os.path.join(save_dir, object_name+'.pickle')
    with open(save_path, 'wb') as f:
        pickle.dump(object_db, f)