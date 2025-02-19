from robot_interface_utils import log_helper, all_close, load_yaml_file, Plan, to_pose_msg, yaml_to_pose_msg
from robotiq_2f_gripper_msgs.msg import CommandRobotiqGripperAction, CommandRobotiqGripperGoal
import actionlib
from interfaces import RobotInterface, ContactileInterface_0, ContactileInterface_1
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest
from geometry_msgs.msg import Twist
import numpy as np
import time
import pickle
from cv_bridge import CvBridge
import rospy
import tf
from tf.transformations import euler_from_quaternion
from collections import deque
from sensor_msgs.msg import Image
import os
import ros_numpy
from sensor_msgs.msg import PointCloud2


class GenerateDB(object):
    def __init__(self, object_name):
        self.__name__ = "DataGrabber"
        rospy.init_node('cm_grabber')
        rospy.on_shutdown(self.handle_exit)

        self.object_name = object_name
        self.object_frame = "/" + self.object_name
        self.properties = np.array(['c', 3, 1, 3, 0.508, 0.5])  # TODO: GT parameters - shape index, size, color, stiffness, weight, friction

        # Database parameters
        self.keys = ['init_point_cloud', 'gt_object_pose', 'action', 'tactile_0', 'tactile_1', 'rgb', 'depth']
        self.dataset = {}
        for key in self.keys:
            self.dataset[key] = []

        # Trajectory parameters
        self.iteration = 2  # repeat times
        self.base_velocity = 0.05  # should be around 0.05
        self.lift_time = 5  # seconds
        self.rotation_time = 8  # Total rotation time - x+2x+x
        self.base_omega = [1.2]  # TODO: param
        self.base_position = [-0.0075, 0, 0.0075]
        self.grasp_force = [2.5, 2.75, 3.0, 3.25, 3.5]  # 1. [2.75, 3.0, 3.25, 3.5]  # TODO: param
        self.sampling_freq = 250.0  # in Hz
        self.pal = 0.07
        self.waypoints = [self.pal, self.pal, 0.14, 0.14, self.pal, self.pal, 0.14, 0.14]

        self.tf_listener = tf.TransformListener()
        self.bridge = CvBridge()
        self.process_depth = False
        self.process_rgb = False
        self.process_pointcloud = False
        self.gt_pose_received = False
        self.robot_pose_received = False
        self.rgb_img_received = False
        self.depth_img_received = False
        self.pointcloud_received = False
        self.record_data = False
        self.depth_image_buffer = deque(maxlen=5)
        self.rgb_image_buffer = deque(maxlen=5)
        self.pointcloud_point_buffer = deque(maxlen=1)
        self.pointcloud_color_buffer = deque(maxlen=1)

        # Set up config
        self.poses = load_yaml_file('config/positions.yaml')
        self.pos_frame_ur = self.poses['ur5']['frame']
        self.pos_frame_panda = self.poses['panda']['frame']
        self.poses = yaml_to_pose_msg(self.poses)
        config = load_yaml_file('config/config.yaml')
        self.config = config['data_grabber']

        # Set up ros topics
        self.rgb_subscriber = rospy.Subscriber('/zed2i/zed_node/rgb/image_rect_color', Image, self.rgb_img_callback,
                                               queue_size=1)
        self.depth_subscriber = rospy.Subscriber('/zed2i/zed_node/depth/depth_registered', Image,
                                                 self.depth_img_callback, queue_size=1)
        self.pointcloud_subscriber = rospy.Subscriber("/crop_box_filter/output", PointCloud2, self.point_cloud_callback)

        # Set up interfaces
        self.ur_interface = RobotInterface(config, 'ur5', 'UR_INTERFACE')
        self.panda_interface = RobotInterface(config, 'panda', 'PANDA_INTERFACE')
        self.robotiq_client = actionlib.SimpleActionClient('command_robotiq_action', CommandRobotiqGripperAction)
        self.robotiq_client.wait_for_server()
        log_helper("Connected to Robotiq Action Server", self.__name__)

        self.contactile_0_interface = ContactileInterface_0(config)
        self.contactile_1_interface = ContactileInterface_1(config)

        self.active_robot = 'ur5'

        log_helper('Initialisation completed !', self.__name__)

    @staticmethod
    def handle_exit():
        """
        Handles the termination of the texture data collection node
        :return:
        """
        print("Shutting down CM data collection !!")

    def depth_img_callback(self, depth_data):
        if self.process_depth:
            img = self.bridge.imgmsg_to_cv2(depth_data)
            # scale_percent = 75  # percent of original size
            # width = int(img.shape[1] * scale_percent / 100)
            # height = int(img.shape[0] * scale_percent / 100)
            # dim = (width, height)
            # resize image
            # resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            self.depth_image_buffer.append(img)
            self.depth_img_received = True

    def rgb_img_callback(self, rgb_data):
        if self.process_rgb:
            img = self.bridge.imgmsg_to_cv2(rgb_data)
            # scale_percent = 75  # percent of original size
            # width = int(img.shape[1] * scale_percent / 100)
            # height = int(img.shape[0] * scale_percent / 100)
            # dim = (width, height)
            # resize image
            # resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            self.rgb_image_buffer.append(img)
            self.rgb_img_received = True

    def point_cloud_callback(self, data):
        if self.process_pointcloud:
            pc = ros_numpy.point_cloud2.pointcloud2_to_array(data)

            # If RGB is packed as a single field, split it:
            if 'rgb' in pc.dtype.names:  # Check if 'rgb' field exists
                pc = ros_numpy.point_cloud2.split_rgb_field(pc)

            points = np.zeros((pc.shape[0], 3))
            rgb = np.zeros((pc.shape[0], 3))
            x_points = pc['x']
            y_points = pc['y']
            z_points = pc['z']
            r = pc['r']
            g = pc['g']
            b = pc['b']
            points[:, 0] = x_points
            points[:, 1] = y_points
            points[:, 2] = z_points
            rgb[:, 0] = r
            rgb[:, 1] = g
            rgb[:, 2] = b
            # pcd_np = ros_numpy.point_cloud2.pointcloud2_to_array(data)
            self.pointcloud_point_buffer.append([points])
            self.pointcloud_color_buffer.append([rgb])
            print('Captured pointcloud with shape', np.array(points).shape)
            self.pointcloud_received = True

    def switch_controllers(self, start_controllers, stop_controllers):
        # Wait for the switch_controller service to be available
        rospy.wait_for_service('/controller_manager/switch_controller')

        try:
            # Create a proxy for the service
            switch_controller = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)

            # Create a request message
            req = SwitchControllerRequest()
            req.start_controllers = start_controllers
            req.stop_controllers = stop_controllers
            req.strictness = SwitchControllerRequest.STRICT

            # Call the service with the request
            result = switch_controller(req)

            # Check if the switch was successful
            if result.ok:
                log_helper("Switched controllers successfully", self.__name__)
            else:
                log_helper("Failed to switch controllers", self.__name__)

        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)

    def transform_twist(self, twist_cmd, from_frame, to_frame):
        # Initialize the node
        try:
            # Get the transformation (position and orientation)
            (trans, rot) = self.tf_listener.lookupTransform(to_frame, from_frame, rospy.Time(0))
            h_transform = self.tf_listener.fromTranslationRotation(trans, rot)
            rot_matrix = h_transform[:3, :3]
            linear_velocity = np.array([twist_cmd.linear.x, twist_cmd.linear.y, twist_cmd.linear.z])
            transformed_linear_velocity = rot_matrix.dot(linear_velocity)
            angular_velocity = np.array([twist_cmd.angular.x, twist_cmd.angular.y, twist_cmd.angular.z])
            transformed_angular_velocity = rot_matrix.dot(angular_velocity)

            # Create the transformed twist
            transformed_twist = Twist()
            transformed_twist.linear.x = transformed_linear_velocity[0]
            transformed_twist.linear.y = transformed_linear_velocity[1]
            transformed_twist.linear.z = transformed_linear_velocity[2]
            transformed_twist.angular.x = transformed_angular_velocity[0]
            transformed_twist.angular.y = transformed_angular_velocity[1]
            transformed_twist.angular.z = transformed_angular_velocity[2]
            return transformed_twist

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr('Transform lookup failed: %s', e)

    def send_twist_command(self, time_exec, linear, angular):
        # Publisher to send the twist command to the UR5
        twist_publisher = rospy.Publisher('/twist_controller/command', Twist, queue_size=10)

        # Create a Twist message
        twist_msg = Twist()
        twist_msg.linear.x = linear[0]
        twist_msg.linear.y = linear[1]
        twist_msg.linear.z = linear[2]
        twist_msg.angular.x = angular[0]
        twist_msg.angular.y = angular[1]
        twist_msg.angular.z = angular[2]

        # Transform the message
        base_frame_twist = self.transform_twist(twist_msg, "tool0_controller", "base_link")
        # print('Computed Base Frame Twist - ', base_frame_twist)
        # Publish the twist message
        log_helper("Sending twist command", self.__name__)
        start_time = time.time()
        time_elapsed = time.time() - start_time
        while time_elapsed < time_exec:
            time_elapsed = time.time() - start_time
            twist_publisher.publish(base_frame_twist)
            self.collect_data()

    def collect_data(self):
        if self.record_data:
            while not self.contactile_0_interface.data_received or not self.contactile_1_interface.data_received or not self.rgb_img_received or not self.depth_img_received:
                current_time = time.time()
                robot_pos = self.get_robot_pose()
                self.robot_traj.append({'time': current_time, 'data': robot_pos, 'mode': self.action_mode})
                trans_obj, eul_obj, quat_obj = self.get_gt_pose()
                self.object_traj.append({'time': current_time, 'data': np.array([trans_obj[0], trans_obj[1], trans_obj[2], eul_obj[0], eul_obj[1], eul_obj[2]])})
                time_taken = time.time() - current_time
                time_left = (1/self.sampling_freq) - time_taken
                rospy.sleep(time_left)

            robot_pos = self.get_robot_pose()
            self.robot_traj.append({'time': time.time(), 'data': robot_pos, 'mode': self.action_mode})
            trans_obj, eul_obj, quat_obj = self.get_gt_pose()
            self.object_traj.append({'time': time.time(), 'data': np.array([trans_obj[0], trans_obj[1], trans_obj[2], eul_obj[0], eul_obj[1], eul_obj[2]])})

            self.contactile_0_interface.process_contactile = False
            self.contactile_0_interface.data_received = False
            self.contactile_1_interface.process_contactile = False
            self.contactile_1_interface.data_received = False

            # Store the tactile data
            contactile_0_data_len = len(self.contactile_0_interface.sensor_data_buffer)
            contactile_1_data_len = len(self.contactile_1_interface.sensor_data_buffer)
            self.tactile_0_traj.extend(list(self.contactile_0_interface.sensor_data_buffer)[:contactile_0_data_len])
            self.tactile_1_traj.extend(list(self.contactile_1_interface.sensor_data_buffer)[:contactile_1_data_len])
            self.contactile_0_interface.sensor_data_buffer.clear()
            self.contactile_0_interface.process_contactile = True
            self.contactile_1_interface.sensor_data_buffer.clear()
            self.contactile_1_interface.process_contactile = True

            # Store the rgb and depth data
            self.process_rgb = False
            self.process_depth = False
            rgb_len = len(self.rgb_image_buffer)
            depth_len = len(self.depth_image_buffer)
            self.rgb_traj.append({'time': time.time(), 'data': list(self.rgb_image_buffer)[:rgb_len]})
            self.depth_traj.append({'time': time.time(), 'data': list(self.depth_image_buffer)[:depth_len]})
            self.rgb_image_buffer.clear()
            self.depth_image_buffer.clear()
            self.rgb_img_received = False
            self.depth_img_received = False
            self.process_rgb = True
            self.process_depth = True

    def get_gt_pose(self):
        # self.tf_listener.waitForTransform("/world", object_frame, rospy.Time(0), rospy.Duration(1))
        trans, quat = self.tf_listener.lookupTransform('/world', self.object_frame, rospy.Time(0))
        euler = euler_from_quaternion(quat)
        self.gt_pose_received = True

        return trans, euler, quat

    def get_robot_pose(self):
        # self.tf_listener.waitForTransform("/world", '/tool0_controller', rospy.Time(0), rospy.Duration(1))
        trans, rot = self.tf_listener.lookupTransform('/world', '/tool0_controller', rospy.Time(0))
        euler = euler_from_quaternion(rot)
        trans_lf, rot_lf = self.tf_listener.lookupTransform('/world', '/left_inner_finger_pad', rospy.Time(0))
        euler_lf = euler_from_quaternion(rot_lf)
        trans_rf, rot_rf = self.tf_listener.lookupTransform('/world', '/right_inner_finger_pad', rospy.Time(0))
        euler_rf = euler_from_quaternion(rot_rf)

        robot_pose = np.array(
            [trans[0], trans[1], trans[2], trans_lf[0], trans_lf[1], trans_lf[2], trans_rf[0], trans_rf[1], trans_rf[2],
             euler[0], euler[1], euler[2], euler_lf[0], euler_lf[1], euler_lf[2], euler_rf[0], euler_rf[1], euler_rf[2]]
        )
        self.robot_pose_received = True
        return robot_pose

    def interact(self, base_omega, grasp_force, base_distance, iteration):
        log_helper('Currently running  - ' + self.object_name + ' with angular velocity: ' + str(base_omega)
                   + ' grasp distance: ' + str(grasp_force) + ' base distance: ' + str(base_distance) + ' iteration: ' + str(iteration),
                   self.__name__)

        # Set baseline
        self.contactile_0_interface.baseline_set_flag = False
        self.contactile_0_interface.set_baseline()
        self.contactile_1_interface.baseline_set_flag = False
        self.contactile_1_interface.set_baseline()

        # Initialize database
        self.object_traj = []
        self.robot_traj = []
        self.tactile_0_traj = []
        self.tactile_1_traj = []
        self.rgb_traj = []
        self.depth_traj = []
        self.action_mode = 0

        # Check if robots are at home position, if not move to home
        if not self.ur_interface.robot_at_pose(self.poses['ur5']['home'], self.pos_frame_ur):
            self.ur_interface.move_to_pose(self.poses['ur5']['home'], self.pos_frame_ur, confirm=False)

        if not self.panda_interface.robot_at_pose(self.poses['panda']['home'], self.pos_frame_panda):
            self.panda_interface.move_to_pose(self.poses['panda']['home'], self.pos_frame_panda, confirm=False)

        # Capture initial pointcloud
        self.process_pointcloud = True
        while not self.pointcloud_received:
            # print('Waiting for point cloud data !! ')
            time.sleep(1/self.sampling_freq)
        self.process_pointcloud = False
        self.tf_listener.waitForTransform("/zed2i_left_camera_optical_frame", "/world", rospy.Time(0), rospy.Duration(1.0))
        trans_cam, rot_cam = self.tf_listener.lookupTransform("/zed2i_left_camera_optical_frame", "/world", rospy.Time(0))
        self.dataset['init_point_cloud'] = {'points': self.pointcloud_point_buffer[0], 'colors': self.pointcloud_color_buffer[0], 'trans': trans_cam, 'rot': rot_cam}

        # Open gripper for safe grasping
        self.move_gripper(0.14)

        # ################################### Move to Grasp Pose ###################################
        start_pose_pf = self.ur_interface.compute_planning_pose(y_offset=0.016, vertical_loc=base_distance, object_frame=self.object_frame)
        res, plan = self.ur_interface.move_above_down_to_pose(start_pose_pf, 'planning_frame',
                                                              self.poses['ur5']['home'], self.pos_frame_ur,
                                                              async_command=True, confirm=False)
        log_helper("Moving down to start pose", self.__name__)
        while not plan.move_group_succeeded:
            time.sleep(1/self.sampling_freq)
        plan.delete()

        # ################################### Perform Palpation Motion ###################################
        self.record_data = True
        self.process_rgb = True
        self.process_depth = True
        self.contactile_0_interface.process_contactile = True
        self.contactile_1_interface.process_contactile = True
        start_time_int = time.time()
        log_helper("palpation starts", self.__name__)
        self.palpation_gripper()
        log_helper("palpation ends", self.__name__)
        end_time_int = time.time()
        # print('Palpation time - ', end_time_int - start_time_int)

        # ################################### Grasp Object ###################################
        self.action_mode += 1
        self.move_gripper(0.09, force_goal=grasp_force)
        log_helper("The gripper has grasped the object.", self.__name__)
        time.sleep(1)

        # Switch controller to velocity control
        self.switch_controllers(['twist_controller'], ['scaled_pos_joint_traj_controller'])

        # ################################### Pick up Object ###################################
        self.action_mode += 1
        self.send_twist_command(self.lift_time, [0.0, self.base_velocity, 0.0], [0.0, 0.0, 0.0])
        # ################################### Perform in-hand Rotation ###################################
        self.send_twist_command(int(self.rotation_time / 4), [0.0, 0.0, 0.0], [0.0, 0.0, base_omega])
        self.send_twist_command(int(self.rotation_time / 2), [0.0, 0.0, 0.0], [0.0, 0.0, -base_omega])
        self.send_twist_command(int(self.rotation_time / 4), [0.0, 0.0, 0.0], [0.0, 0.0, base_omega])

        # Stop Twist
        self.send_twist_command(0.1, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        # Switch controller to position control
        self.switch_controllers(['scaled_pos_joint_traj_controller'], ['twist_controller'])

        # ################################### Place object back ###################################
        self.action_mode += 1
        res, plan = self.ur_interface.move_to_pose(start_pose_pf, 'planning_frame', async_command=True, confirm=False)
        log_helper("Moving down to start pose", self.__name__)
        while not plan.move_group_succeeded:
            self.collect_data()
        plan.delete()
        self.record_data = False

        self.move_gripper(0.14)
        self.dataset['gt_object_pose'] = self.object_traj
        self.dataset['action'] = self.robot_traj
        self.dataset['tactile_0'] = self.tactile_0_traj
        self.dataset['tactile_1'] = self.tactile_1_traj
        self.dataset['rgb'] = self.rgb_traj
        self.dataset['depth'] = self.depth_traj

        self.ur_interface.move_up(self.pos_frame_ur, self.poses['ur5']['home'], confirm=False)

    def move_gripper(self, position_goal, force_goal=None):  # target_force in [%]
        goal = CommandRobotiqGripperGoal()
        goal.emergency_release = False
        goal.stop = False
        goal.speed = 0.01
        goal.force = 100
        goal.position = position_goal
        if force_goal is not None:
            force_error = 1
            while abs(force_error) > 0.5:
                self.robotiq_client.send_goal(goal, feedback_cb=self.feedback_callback)
                self.robotiq_client.wait_for_result()
                # refine goal position
                force_error = (self.contactile_0_interface.force[2] + self.contactile_1_interface.force[2]) / 2 - force_goal
                goal.position = goal.position + 0.0001*force_error  # p controller
        else:
            self.robotiq_client.send_goal(goal, feedback_cb=self.feedback_callback)
            self.robotiq_client.wait_for_result()

        return self.robotiq_client.get_state()

    def palpation_gripper(self):
        for position in self.waypoints:
            state = self.move_gripper(position)
            if state != actionlib.GoalStatus.SUCCEEDED:
                break

    def feedback_callback(self, feedback):
        self.collect_data()

    def run_experiments(self, name):
        for base_omega in self.base_omega:
            for grasp_force in self.grasp_force:
                for base_dist in self.base_position:
                    iteration = 0
                    while iteration < self.iteration:
                        self.interact(base_omega, grasp_force, base_dist, iteration)
                        user_choice = input('Accept trajectory and reset object done ? ')
                        # user_choice = 'y'
                        if user_choice == 'y':
                            # Save the database for each experiment
                            folder_path = os.path.join(f'dataset_slip/{name}',
                                                       f'object={self.object_name}-angular_vel={base_omega}-grasp_dist={grasp_force}-base_dist={base_dist}-iteration={iteration}')
                            iteration += 1
                            if not os.path.exists(folder_path):
                                os.makedirs(folder_path)

                            pickle_path = os.path.join(folder_path, 'test.pickle')
                            with open(pickle_path, 'wb') as f:
                                pickle.dump(self.dataset, f)

                            log_helper('finished - ' + self.object_name + ' with angular velocity: ' + str(base_omega)
                                       + ' grasp force: ' + str(grasp_force) + ' base distance: ' + str(base_dist) + ' iteration: ' + str(iteration),
                                       self.__name__)
                            log_helper("##################################################################", self.__name__)
                        else:
                            # iteration -= 1
                            log_helper('iteration: ' + str(iteration) + ' failed and not accepted!', self.__name__)


def main(argv=None):
    object_name = 'circle_3'  # Input object name here
    folder_name = 'c-3-1'
    db_runner = GenerateDB(object_name)
    db_runner.run_experiments(folder_name)


if __name__ == "__main__":
    main()
