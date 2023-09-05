#! /usr/bin/env python3
import rospy
import gp_odometer
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
from scipy.optimize._optimize import MemoizeJac
import scipy.optimize as opt
import numpy as np
import time


import gp_dist_field as gpdf
import utils



kDefaultDownsampleNb = 5000
kDefaultMaxRange = 1.5
kDefaultVoxelSize = 0.02
kDefaultUseKeops = True

class CoarseOdometer:
    
    def __init__(self):

        self.counter = 0
        self.rot = np.zeros(3)
        self.pos = np.zeros(3)
        
        rospy.init_node('coarse_odometry')
        rospy.loginfo("Starting coarse_odometry node")

        self.pub = rospy.Publisher('/coarse_odometry/points', PointCloud2, queue_size=10)
        self.pub_odom = rospy.Publisher('/coarse_odometry/relative_pose', PoseStamped, queue_size=10)


        self.max_range = rospy.get_param('~max_range', kDefaultMaxRange)
        self.use_keops = rospy.get_param('~use_keops', kDefaultUseKeops)
        self.voxel_size = rospy.get_param('~voxel_size', kDefaultVoxelSize)
        self.downsample_nb = rospy.get_param('~downsample_nb', kDefaultDownsampleNb)




    def pcCallback(self, msg):

        print('\n\nProcessing frame ', self.counter, '...')

        t0 = time.perf_counter()

        # Check if there is colour information
        if self.counter == 0:
            self.receiverd_seq = msg.header.seq
            self.msg_reader = utils.RosNpConverter(msg, self.downsample_nb, self.voxel_size, self.max_range)
        else:
            if msg.header.seq != self.receiverd_seq + 1:
                rospy.logerr('At least one missed frame!')
                rospy.logerr('Please run the data at lower frequency for optimal performance')
            self.receiverd_seq = msg.header.seq

        pts, colours = self.msg_reader.rosMsgToNp(msg, get_colours = True, store_raw_cropped = True)


        prev_rot = self.rot
        prev_pos = self.pos


        self.src_pts, self.src_colours = utils.voxelDownsample(pts, colours, self.voxel_size)



        # Register the pointcloud
        if self.counter > 0:

            self.dist_field = gpdf.gpDistFieldSE(self.target_pts, 2*self.voxel_size, use_keops=self.use_keops, lumped_matrix=True)

            # Optimize
            fun = MemoizeJac(self.smoothMinCostFunction)
            jac = fun.derivative
            x0 = np.concatenate((prev_rot, prev_pos))
            results = opt.least_squares(fun, x0, jac=jac, method='lm', verbose=0, ftol=1e-6, xtol=1e-6, gtol=1e-6, max_nfev=5)

            self.rot = results.x[0:3]
            self.pos = results.x[3:6]

            results = opt.least_squares(fun, results.x, jac=jac, method='dogbox', verbose=0, ftol=1e-3, xtol=1e-5, loss='cauchy', f_scale=self.voxel_size)

            self.rot = results.x[0:3]
            self.pos = results.x[3:6]
    


        self.target_pts = utils.transformPoints3D(self.src_pts, np.concatenate((self.rot, self.pos)))



        # Publish the pointcloud
        pts_temp, colours_temp = self.msg_reader.getLastRawCropped(True)
        pc_msg = self.msg_reader.npToRosMsg(pts_temp, colours_temp, msg.header.stamp, 'cam', self.counter)

        self.pub.publish(pc_msg)


        # Compute the relative pose
        R0 = R.from_rotvec(prev_rot).as_matrix()
        R1 = R.from_rotvec(self.rot).as_matrix()
        T0 = np.eye(4)
        T0[0:3,0:3] = R0
        T0[0:3,3] = prev_pos
        T1 = np.eye(4)
        T1[0:3,0:3] = R1
        T1[0:3,3] = self.pos
        T = np.matmul(np.linalg.inv(T0), T1)
        q_vec = R.from_matrix(T[0:3,0:3]).as_quat()
        t_vec = T[0:3,3]

        # Publish the relative pose
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = msg.header.frame_id
        pose_msg.header.stamp = msg.header.stamp
        pose_msg.header.seq = self.counter
        pose_msg.pose.position.x = t_vec[0]
        pose_msg.pose.position.y = t_vec[1]
        pose_msg.pose.position.z = t_vec[2]
        pose_msg.pose.orientation.x = q_vec[0]
        pose_msg.pose.orientation.y = q_vec[1]
        pose_msg.pose.orientation.z = q_vec[2]
        pose_msg.pose.orientation.w = q_vec[3]
        self.pub_odom.publish(pose_msg)

        t3 = time.perf_counter()
        self.counter += 1
        
        # Print some info
        rospy.loginfo('Coarse odometry time: '+ str(round(t3-t0,6)))



    def smoothMinCostFunction(self, x):
        epsilon = 1e-10
        nb_pts = self.src_pts.shape[0]
        all_pts = np.empty((7*nb_pts,3))
        all_pts[0:nb_pts,:] = utils.transformPoints3D(self.src_pts, x)
        for i in range(6):
            x_eps = x.copy()
            x_eps[i] += epsilon
            all_pts[(i+1)*nb_pts:(i+2)*nb_pts,:] = utils.transformPoints3D(self.src_pts, x_eps)
        
        all_dists = self.dist_field.query(all_pts).squeeze()
        res = all_dists[0:nb_pts]
        jac = np.empty((nb_pts,6))
        for i in range(6):
            jac[:,i] = ((all_dists[(i+1)*nb_pts:(i+2)*nb_pts] - res)/epsilon).squeeze()

        return res, jac


if __name__ == '__main__':

    gp_odometer = CoarseOdometer()
    rospy.Subscriber("/points", PointCloud2, gp_odometer.pcCallback, queue_size=1000)
    #rospy.Subscriber("/camera/rgb/points", PointCloud2, gp_odometer.pcCallback, queue_size=1000)
    rospy.spin()