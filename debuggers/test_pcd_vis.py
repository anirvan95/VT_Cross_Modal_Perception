import os
import open3d as o3d
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


parameters = 'object=rect_3-angular_vel=1.2-grasp_dist=0.088-iteration=2'
# folder_path = os.path.join('dataset/test/exp', parameters)
file_name = os.path.join(parameters, 'test.pickle')
with open(file_name, 'rb') as f:
    db = pickle.load(f)

keys = ['gt_object_pose', 'action', 'tactile_0', 'tactile_1',  'rgb', 'depth']
pcd_ros = db['init_point_cloud']
points_np = pcd_ros['points'][0]


point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(pcd_ros['points'][0])
point_cloud.paint_uniform_color([1, 0, 0])
# point_cloud.colors = o3d.utility.Vector3dVector(pcd_ros['colors'][0] / 255)

o3d.visualization.draw_geometries([point_cloud])
