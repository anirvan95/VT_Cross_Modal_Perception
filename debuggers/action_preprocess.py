import pickle
import os
import numpy as np
from scipy.interpolate import interp1d
import imageio
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


def pose_to_se3(pose):
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


# action parameters - x,y,z,roll, pitch, yaw, gripper_distance, target force, target, angular_velocity

base_dir = os.path.join('dataset/test/s-1-1')
folder_name = 'object=square_1-angular_vel=0.4-grasp_dist=2.75-iteration=1'
folder_path = os.path.join(base_dir, folder_name)
file_name = os.path.join(folder_path, 'test.pickle')

with open(file_name, 'rb') as f:
    db = pickle.load(f)

keys = ['gt_object_pose', 'action', 'tactile_0', 'tactile_1', 'rgb', 'depth']
robot_pose = db['action']

robot_pose_data = np.array([entry['data'] for entry in robot_pose if 'data' in entry and len(entry['data']) > 0])
planning_frame = np.hstack([robot_pose_data[:, 0:3], robot_pose_data[:, 9:12]])
left_finger = np.hstack([robot_pose_data[:, 3:6], robot_pose_data[:, 12:15]])
right_finger = np.hstack([robot_pose_data[:, 6:9], robot_pose_data[:, 15:18]])
gripper_width_arr = []
for i in range(len(planning_frame)):
    current_planning_frame_se3 = pose_to_se3(planning_frame[i])
    current_left_finger_se3 = pose_to_se3(left_finger[i])
    current_right_finger_se3 = pose_to_se3(right_finger[i])
    relative_left_finger_se3 = np.linalg.inv(current_planning_frame_se3) @ current_left_finger_se3
    relative_right_finger_se3 = np.linalg.inv(current_planning_frame_se3) @ current_right_finger_se3
    current_gripper_width = abs(relative_left_finger_se3[0, 3] - relative_right_finger_se3[0, 3])
    gripper_width_arr.append(current_gripper_width)

plt.plot(gripper_width_arr)
plt.show()