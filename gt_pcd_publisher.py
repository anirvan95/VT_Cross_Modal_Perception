import open3d as o3d
import os
import copy
import numpy as np
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
import rospy
import time
import tf


def get_gt_pose(object_name):
    global tf_listener
    global gt_pose_received
    object_frame = "/" + object_name
    tf_listener.waitForTransform("/world", object_frame, rospy.Time(0), rospy.Duration(1.0))
    trans, rot = tf_listener.lookupTransform('/world', object_frame, rospy.Time(0))
    gt_pose_received = True
    return trans, rot


def publish_pointcloud(points_pcd):
    global pub

    # Create a PointCloud2 message
    pc_msg = PointCloud2()
    pc_msg.header.stamp = rospy.Time.now()
    pc_msg.header.frame_id = "world"  # Set the frame ID for the point cloud

    # Fill in the point cloud data
    pc_msg.height = 1
    pc_msg.width = len(points_pcd)
    pc_msg.fields.append(PointField(name="x", offset=0, datatype=7, count=1))
    pc_msg.fields.append(PointField(name="y", offset=4, datatype=7, count=1))
    pc_msg.fields.append(PointField(name="z", offset=8, datatype=7, count=1))
    pc_msg.is_bigendian = False
    pc_msg.point_step = 12  # Size of each point in bytes (3 floats)
    pc_msg.row_step = pc_msg.point_step * len(points_pcd)
    pc_msg.is_dense = True
    pc_msg.data = points_pcd.tobytes()
    pub.publish(pc_msg)

object_name = 'hexagon_1'
sampling_freq = 50
gt_object_path = os.path.join('dataset', 'gt_objects', str(object_name+'.stl'))
mesh = o3d.io.read_triangle_mesh(gt_object_path)
mesh.scale(0.001, center=(0, 0, 0))
gt_pcd = mesh.sample_points_poisson_disk(100000)
gt_pcd = gt_pcd.paint_uniform_color([1, 0, 0])
o3d.visualization.draw(gt_pcd)

rospy.init_node('gt_pcd_publisher')
gt_pose_received = False
pub = rospy.Publisher('gt_pointcloud', PointCloud2, queue_size=10)
tf_listener = tf.TransformListener()

while not rospy.is_shutdown():
    trans, rot = get_gt_pose(object_name)
    H_transform = tf_listener.fromTranslationRotation(trans, rot)
    gt_pcd_mod = copy.deepcopy(gt_pcd).transform(H_transform)
    publish_pointcloud(np.asarray(gt_pcd_mod.points, dtype=np.float32))
    time.sleep(1/sampling_freq)
