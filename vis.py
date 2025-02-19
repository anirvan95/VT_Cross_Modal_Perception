import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import pickle
from scipy.spatial.transform import Rotation as R


def transform(data):
    """
    Remove nan. Clip data within (0,1). Scale the data in the range of (0,255).
    :param data: image np.ndarray
    :return: image np.ndarray
    """
    data_no_nan = np.nan_to_num(data, nan=0.0)
    data_clipped = np.clip(data_no_nan, 0, 1)
    min_val = np.min(data_clipped)
    max_val = np.max(data_clipped)
    scaled = np.array([(frame - min_val) / (max_val - min_val) * 255 for _, frame in enumerate(data)])
    return scaled


def pose_to_se3(pose):
    """
    Convert a 6-DOF pose (translation + Euler angles) to a 4x4 SE(3) transformation matrix.
    :param pose: A 6-DOF vector [tx, ty, tz, roll, pitch, yaw].
    :return: A 4x4 transformation matrix.
    """
    # Extract translation and rotation components
    translation = pose[:3]
    euler_angles = pose[3:]

    # Create rotation matrix from Euler angles
    rotation_matrix = R.from_euler('xyz', euler_angles).as_matrix()

    # Create SE(3) homogeneous transformation matrix
    se3_matrix = np.eye(4)
    se3_matrix[:3, :3] = rotation_matrix
    se3_matrix[:3, 3] = translation
    return se3_matrix


def se3_to_pose(se3_matrix):
    """
    Convert a 4x4 SE(3) matrix back to a 6-DOF pose (translation + Euler angles).
    :param se3_matrix: A 4x4 SE(3) transformation matrix.
    :return: A 6-DOF vector [tx, ty, tz, roll, pitch, yaw].
    """
    # Extract translation and rotation components
    translation = se3_matrix[:3, 3]
    rotation_matrix = se3_matrix[:3, :3]

    # Convert rotation matrix to Euler angles
    euler_angles = R.from_matrix(rotation_matrix).as_euler('xyz')

    return np.hstack((translation, euler_angles))


class DataProcessor:
    def __init__(self, rgb_, depth_, object_pose_, robot_pose_, tactile_0_, tactile_1_, folder_path_):
        self.rgb = rgb_
        self.depth = depth_
        self.object_pose = object_pose_
        self.robot_pose = robot_pose_
        self.tactile_0 = tactile_0_
        self.tactile_1 = tactile_1_
        self.folder_path = folder_path_

        self.rgb_gif = rgb
        self.depth_gif = depth
        self.object_pose_ts = []
        self.object_pose_data = []
        self.robot_pose_ts = []
        self.robot_pose_data = []
        self.tactile_0_ts = []
        self.tactile_1_ts = []
        self.fx_0, self.fy_0, self.fz_0 = [], [], []
        self.fx_1, self.fy_1, self.fz_1 = [], [], []
        self.tx_0, self.ty_0, self.tz_0 = [], [], []
        self.tx_1, self.ty_1, self.tz_1 = [], [], []
        self.pfx_0, self.pfx_1 = [], []
        self.pfy_0, self.pfy_1 = [], []
        self.pfz_0, self.pfz_1 = [], []

    def process_rgb_depth(self):
        self.rgb_gif = np.array([entry['data'][0] for entry in self.rgb if 'data' in entry and len(entry['data']) > 0])
        self.rgb_time = np.array([entry['time'] for entry in self.rgb])
        self.rgb_time = self.rgb_time - self.rgb_time[0]
        self.rgb_gif[..., [0, 2]] = self.rgb_gif[..., [2, 0]]  # Flip the RGB channels
        self.depth_gif = np.array([entry['data'][0] for entry in self.depth if 'data' in entry and len(entry['data']) > 0])

    def process_object_pose(self):
        initial_pose = np.array(self.object_pose[0]['data'])
        initial_se3 = pose_to_se3(initial_pose)

        for i in range(len(self.object_pose)):
            current_pose = np.array(self.object_pose[i]['data'])
            current_se3 = pose_to_se3(current_pose)

            # Compute relative transformation: T_relative = T_initial^-1 * T_current
            relative_se3 = np.linalg.inv(initial_se3) @ current_se3
            relative_pose = se3_to_pose(relative_se3)

            self.object_pose_ts.append(self.object_pose[i]['time'])
            self.object_pose_data.append(relative_pose)

        self.object_pose_ts = np.array(self.object_pose_ts) - self.object_pose_ts[0]
        self.object_pose_data = np.array(self.object_pose_data)

    def process_robot_pose(self):
        """
        Process robot poses by computing relative transformations with respect to the initial pose.
        The pose contains three systems: planning frame, left finger, and right finger.
        Each is compared independently and stacked back together.
        """
        def extract_pose_segment(pose, trans_idx, rot_idx):
            """ Helper function to extract a 6-DOF segment from the 18-DOF pose. """
            translation = pose[trans_idx:trans_idx + 3]
            rotation = pose[rot_idx:rot_idx + 3]
            return np.hstack([translation, rotation])

        def stack_pose_segments(planning_frame_pose, left_finger_pose, right_finger_pose):
            """ Helper function to stack three 6-DOF poses into the final 18-DOF pose. """
            return np.hstack([planning_frame_pose[:3], left_finger_pose[:3], right_finger_pose[:3],
                              planning_frame_pose[3:], left_finger_pose[3:], right_finger_pose[3:]])

        initial_pose = np.array(self.robot_pose[0]['data'])

        initial_planning_frame_pose = extract_pose_segment(initial_pose, 0, 9)
        initial_left_finger_pose = extract_pose_segment(initial_pose, 3, 12)
        initial_right_finger_pose = extract_pose_segment(initial_pose, 6, 15)

        initial_planning_frame_se3 = pose_to_se3(initial_planning_frame_pose)
        initial_left_finger_se3 = pose_to_se3(initial_left_finger_pose)
        initial_right_finger_se3 = pose_to_se3(initial_right_finger_pose)

        for i in range(len(self.robot_pose)):
            current_pose = np.array(self.robot_pose[i]['data'])

            # Split the current pose into planning frame, left finger, and right finger
            current_planning_frame_pose = extract_pose_segment(current_pose, 0, 9)
            current_left_finger_pose = extract_pose_segment(current_pose, 3, 12)
            current_right_finger_pose = extract_pose_segment(current_pose, 6, 15)

            current_planning_frame_se3 = pose_to_se3(current_planning_frame_pose)
            current_left_finger_se3 = pose_to_se3(current_left_finger_pose)
            current_right_finger_se3 = pose_to_se3(current_right_finger_pose)

            relative_planning_frame_se3 = np.linalg.inv(initial_planning_frame_se3) @ current_planning_frame_se3
            relative_left_finger_se3 = np.linalg.inv(initial_left_finger_se3) @ current_left_finger_se3
            relative_right_finger_se3 = np.linalg.inv(initial_right_finger_se3) @ current_right_finger_se3

            relative_planning_frame_pose = se3_to_pose(relative_planning_frame_se3)
            relative_left_finger_pose = se3_to_pose(relative_left_finger_se3)
            relative_right_finger_pose = se3_to_pose(relative_right_finger_se3)

            # Stack the results back into 18-DOF pose

            relative_robot_pose = stack_pose_segments(
                relative_planning_frame_pose,
                relative_left_finger_pose,
                relative_right_finger_pose
            )
            '''
            relative_robot_pose = stack_pose_segments(
                current_planning_frame_pose,
                current_left_finger_pose,
                current_right_finger_pose
            )
            '''
            self.robot_pose_ts.append(self.robot_pose[i]['time'])
            self.robot_pose_data.append(relative_robot_pose)

        # Convert to numpy arrays for further processing
        self.robot_pose_ts = np.array(self.robot_pose_ts) - self.robot_pose_ts[0]
        self.robot_pose_data = np.array(self.robot_pose_data)

    def process_tactile_data(self):
        for i in range(len(self.tactile_0)):
            self.tactile_0_ts.append(self.tactile_0[i]['time'])
            self.fx_0.append(self.tactile_0[i]['force_fx'])
            self.fy_0.append(self.tactile_0[i]['force_fy'])
            self.fz_0.append(self.tactile_0[i]['force_fz'])
            self.tx_0.append(self.tactile_0[i]['torque_tx'])
            self.ty_0.append(self.tactile_0[i]['torque_ty'])
            self.tz_0.append(self.tactile_0[i]['torque_tz'])
            self.pfx_0.append(self.tactile_0[i]['pillar_forces_fx'])
            self.pfy_0.append(self.tactile_0[i]['pillar_forces_fy'])
            self.pfz_0.append(self.tactile_0[i]['pillar_forces_fz'])

        for i in range(len(self.tactile_1)):
            self.tactile_1_ts.append(self.tactile_1[i]['time'])
            self.fx_1.append(self.tactile_1[i]['force_fx'])
            self.fy_1.append(self.tactile_1[i]['force_fy'])
            self.fz_1.append(self.tactile_1[i]['force_fz'])
            self.tx_1.append(self.tactile_1[i]['torque_tx'])
            self.ty_1.append(self.tactile_1[i]['torque_ty'])
            self.tz_1.append(self.tactile_1[i]['torque_tz'])
            self.pfx_1.append(self.tactile_1[i]['pillar_forces_fx'])
            self.pfy_1.append(self.tactile_1[i]['pillar_forces_fy'])
            self.pfz_1.append(self.tactile_1[i]['pillar_forces_fz'])

        self.tactile_0_ts = np.array(self.tactile_0_ts) - self.tactile_0_ts[0]
        self.tactile_1_ts = np.array(self.tactile_1_ts) - self.tactile_1_ts[0]
        self.fx_0, self.fy_0, self.fz_0 = np.array(self.fx_0), np.array(self.fy_0), np.array(self.fz_0)
        self.fx_1, self.fy_1, self.fz_1 = np.array(self.fx_1), np.array(self.fy_1), np.array(self.fz_1)
        self.tx_0, self.ty_0, self.tz_0 = np.array(self.tx_0), np.array(self.ty_0), np.array(self.tz_0)
        self.tx_1, self.ty_1, self.tz_1 = np.array(self.tx_1), np.array(self.ty_1), np.array(self.tz_1)
        self.pfx_0, self.pfy_0, self.pfz_0 = np.array(self.pfx_0), np.array(self.pfy_0), np.array(self.pfz_0)
        self.pfx_1, self.pfy_1, self.pfz_1 = np.array(self.pfx_1), np.array(self.pfy_1), np.array(self.pfz_1)

    def save_stats(self, param):
        fig, ax = plt.subplots(2, 4, figsize=(30, 20))

        ax[0, 0].plot(self.object_pose_ts, self.object_pose_data[:, 0], 'r', label="x")  # x
        ax[0, 0].plot(self.object_pose_ts, self.object_pose_data[:, 1], 'g', label="y")  # y
        ax[0, 0].plot(self.object_pose_ts, self.object_pose_data[:, 2], 'b', label="z")  # z
        ax[0, 0].set_xlabel(r"Time ($\mathrm{s}$)")
        ax[0, 0].set_ylabel(r"($\mathrm{m}$)")
        ax[0, 0].set_title("object pose position")

        ax[0, 1].plot(self.object_pose_ts, self.object_pose_data[:, 3], 'r', label="x")  # roll
        ax[0, 1].plot(self.object_pose_ts, self.object_pose_data[:, 4], 'g', label="y")  # pitch
        ax[0, 1].plot(self.object_pose_ts, self.object_pose_data[:, 5], 'b', label="z")  # yaw
        ax[0, 1].set_xlabel(r"Time ($\mathrm{s}$)")
        ax[0, 1].set_ylabel(r"($\mathrm{rad}$)")
        ax[0, 1].set_title("object pose angle")

        # Labels for the robot position
        position_labels = [
            'trans_x', 'trans_y', 'trans_z',
            'trans_lf_x', 'trans_lf_y', 'trans_lf_z',
            'trans_rf_x', 'trans_rf_y', 'trans_rf_z'
        ]
        # Labels for the robot rotation (Euler angles)
        angle_labels = [
            'euler_x', 'euler_y', 'euler_z',
            'euler_lf_x', 'euler_lf_y', 'euler_lf_z',
            'euler_rf_x', 'euler_rf_y', 'euler_rf_z'
        ]
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'orange']
        for i in range(3):  # 9: consider right/left gripper, 3: only EE
            # position
            ax[0, 2].plot(self.robot_pose_ts, self.robot_pose_data[:, i+3], colors[i], label=position_labels[i])
            # angle
            ax[0, 3].plot(self.robot_pose_ts, self.robot_pose_data[:, i+9+3], colors[i], label=angle_labels[i])

        ax[0, 2].set_title("robot position")
        ax[0, 2].set_xlabel(r"Time ($\mathrm{s}$)")
        ax[0, 2].set_ylabel(r"($\mathrm{m}$)")
        ax[0, 2].legend(loc='upper right')
        ax[0, 3].set_title("robot pose angle")
        ax[0, 3].set_xlabel(r"Time ($\mathrm{s}$)")
        ax[0, 3].set_ylabel(r"($\mathrm{rad}$)")
        ax[0, 3].legend(loc='upper right')

        ax[1, 0].plot(self.tactile_0_ts, self.fx_0, 'r')
        ax[1, 0].plot(self.tactile_0_ts, self.fy_0, 'g')
        ax[1, 0].plot(self.tactile_0_ts, self.fz_0, 'b')
        ax[1, 0].set_xlabel(r"Time ($\mathrm{s}$)")
        ax[1, 0].set_ylabel(r"($\mathrm{N}$)")
        # ax[1, 0].text(10, -2.5, 'Blue is consistent with gripper movement direction. Green is vertical.', fontsize=12, color='blue')
        ax[1, 0].set_title("force in tactile sensor 0")

        ax[1, 1].plot(self.tactile_1_ts, self.fx_1, 'r')
        ax[1, 1].plot(self.tactile_1_ts, self.fy_1, 'g')
        ax[1, 1].plot(self.tactile_1_ts, self.fz_1, 'b')
        ax[1, 1].set_xlabel(r"Time ($\mathrm{s}$)")
        ax[1, 1].set_ylabel(r"($\mathrm{N}$)")
        ax[1, 1].set_title("force in tactile sensor 1")

        # torque
        ax[1, 2].plot(self.tactile_0_ts, self.tx_0, 'r')
        ax[1, 2].plot(self.tactile_0_ts, self.ty_0, 'g')
        ax[1, 2].plot(self.tactile_0_ts, self.tz_0, 'b')
        ax[1, 2].plot(self.tactile_1_ts, self.tx_1, 'c')
        ax[1, 2].plot(self.tactile_1_ts, self.ty_1, 'y')
        ax[1, 2].plot(self.tactile_1_ts, self.tz_1, 'm')
        ax[1, 2].set_xlabel(r"Time ($\mathrm{s}$)")
        ax[1, 2].set_ylabel(r"($\mathrm{N·m}$)")
        ax[1, 2].set_title("torques in tactile 0 and 1")

        ax[1, 3].plot(self.object_pose_ts, '*k')
        ax[1, 3].plot(self.robot_pose_ts, '.r')
        ax[1, 3].plot(self.rgb_time, '+m')
        ax[1, 3].plot(self.tactile_0_ts, '*g')
        ax[1, 3].plot(self.tactile_1_ts, '.b')
        ax[1, 3].set_ylabel(r"Time ($\mathrm{s}$)")
        ax[1, 3].set_title("number of samples")

        stats_path = os.path.join(self.folder_path, 'stats.png')
        fig.savefig(stats_path)
        plt.close()

    def save_gif(self):
        rgb_file = os.path.join(self.folder_path, 'rgb.gif')
        depth_file = os.path.join(self.folder_path, 'depth.gif')
        imageio.mimsave(rgb_file, self.rgb_gif, fps=15)
        imageio.mimsave(depth_file, transform(self.depth_gif), fps=15)


if __name__ == '__main__':
    # Change the date and parameters!!
    base_dir = os.path.join('dataset/cm_dataset/s-1-1')
    folders = [name for name in sorted(os.listdir(base_dir)) if os.path.isdir(os.path.join(base_dir, name))]
    for folder_name in folders:
        print('Processing - ', folder_name)
        folder_path = os.path.join(base_dir, folder_name)
        file_name = os.path.join(folder_path, 'test.pickle')
        with open(file_name, 'rb') as f:
            db = pickle.load(f)

        keys = ['gt_object_pose', 'action', 'tactile_0', 'tactile_1', 'rgb', 'depth']
        object_pose = db['gt_object_pose']
        robot_pose = db['action']
        tactile_0 = db['tactile_0']
        tactile_1 = db['tactile_1']
        rgb = db['rgb']
        depth = db['depth']

        processor = DataProcessor(rgb, depth, object_pose, robot_pose, tactile_0, tactile_1, folder_path)

        processor.process_rgb_depth()
        processor.process_object_pose()
        processor.process_robot_pose()
        processor.process_tactile_data()

        print("RGB/D sampling rate: ", len(processor.rgb_gif) / processor.object_pose_ts[-1])
        print("object sampling rate: ", len(processor.object_pose_ts) / processor.object_pose_ts[-1])
        print("robot sampling rate: ", len(processor.robot_pose_ts) / processor.robot_pose_ts[-1])
        print("tactile sensor 0 sampling rate: ", len(processor.tactile_0_ts) / processor.tactile_0_ts[-1])
        print("tactile sensor 1 sampling rate: ", len(processor.tactile_1_ts) / processor.tactile_1_ts[-1])

        processor.save_gif()
        processor.save_stats(folder_path)

