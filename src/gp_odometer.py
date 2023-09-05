import rospy
import numpy as np
import time
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseArray, Pose, PoseStamped


import scipy.optimize as opt
from scipy.optimize._optimize import MemoizeJac
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree

from utils import Block
import utils

import gp_dist_field as gpdf

kDefaultVoxelSize = 0.02
kDefaultBlockMultiplier = 10
kDefaultDownsampleNb = 5000
kDefaultMaxRange = 1.5
kDefaultUseKeops = True





class GPOdometer:

    def __init__(self):


        self.blocks = {}
        self.counter = 0
        self.rot = np.zeros(3)
        self.pos = np.zeros(3)

        self.rot_est = []
        self.pos_est = []


        rospy.init_node('gp_odometry')
        rospy.loginfo("Starting gp_odometry node")

        self.pub = rospy.Publisher('/gp_odometry/map', PointCloud2, queue_size=10)

        # Publisher for the odometry trajectory estimates
        self.pub_odom = rospy.Publisher('/gp_odometry/odometry', PoseStamped, queue_size=10)
        self.pub_odom_all = rospy.Publisher('/gp_odometry/trajectory', PoseArray, queue_size=10)

        # Read parameters

        self.max_range = rospy.get_param('~max_range', kDefaultMaxRange)
        self.use_keops = rospy.get_param('~use_keops', kDefaultUseKeops)
        self.voxel_size = rospy.get_param('~voxel_size', kDefaultVoxelSize)
        self.downsample_nb = rospy.get_param('~downsample_nb', kDefaultDownsampleNb)


        self.block_size = self.voxel_size*kDefaultBlockMultiplier


        
        print('--Params-- Downsample nb: ', self.downsample_nb)
        print('--Params-- Voxel size: ', self.voxel_size)
        print('--Params-- Max range: ', self.max_range)
        print('--Params-- Use keops: ', self.use_keops)



    # Callback for the pointcloud and odometry messages (Pointcloud2 and PoseStamped)
    def pcCallback(self, msg, trans=None):

        print('\n\nProcessing frame ', self.counter, '...')
        if trans is not None:
            current_T = np.eye(4)
            current_T[0:3,0:3] = R.from_rotvec(self.rot).as_matrix()
            current_T[0:3,3] = self.pos

            delta_T = np.eye(4)
            quat = np.array([trans.pose.orientation.x, trans.pose.orientation.y ,trans.pose.orientation.z ,trans.pose.orientation.w])
            delta_T[0:3, 0:3] = R.from_quat(quat).as_matrix()
            delta_T[0,3] = trans.pose.position.x
            delta_T[1,3] = trans.pose.position.y
            delta_T[2,3] = trans.pose.position.z

            new_T = np.matmul(current_T, delta_T)
            self.rot = R.from_matrix(new_T[0:3,0:3]).as_rotvec()
            self.pos = new_T[0:3,3]




        t0 = time.perf_counter()

        # If first msg, create msg reader
        if self.counter == 0:
            self.receiverd_seq = msg.header.seq
            self.msg_reader = utils.RosNpConverter(msg, self.downsample_nb, self.voxel_size, self.max_range)
        # Check if the sequence is correct (no frame skip)
        else:
            if msg.header.seq != self.receiverd_seq + 1:
                rospy.logerr('At least one missed frame!')
                rospy.logerr('Please run the data at lower frequency for optimal performance')
            self.receiverd_seq = msg.header.seq


        # Read the pointcloud message
        pts, colours = self.msg_reader.rosMsgToNp(msg, get_colours = True, store_raw_cropped = False)

        t1 = time.perf_counter()

        # Register the pointcloud
        if self.counter > 0:
            pts = self.odometryRegistration(pts)

        t2 = time.perf_counter()

        # Get block ids for each point
        block_ids = np.floor_divide(pts, self.block_size).astype(int)
        #Get unique ids of blocks
        unique_block_ids = np.unique(block_ids, axis=0)

        # For each observed blocks insert the new points
        for i in range(unique_block_ids.shape[0]):
            block_id = unique_block_ids[i,:]
            block_mask = np.all(block_ids == block_id, axis=1)
            block_pts = pts[block_mask,:]
            block_id = tuple(block_id)
            if not(tuple(block_id) in self.blocks):
                self.blocks[block_id] = Block(block_id, self.voxel_size)
            if colours is not None:
                block_colours = colours[block_mask,:]
            else:
                block_colours = None
            
            nb_new_voxels = self.blocks[block_id].addPoints(block_pts, block_colours)
            if nb_new_voxels > 2:
                self.blocks[block_id].prune()


        t3 = time.perf_counter()
        self.counter += 1
        
        # Print some info
        print('Time to receive / downsample pointcloud: ', round(t1-t0,6))
        print('Time to register pointcloud: ', round(t2-t1,6))
        print('Time to insert points in blocks: ', round(t3-t2,6))
        print('Time total : ', round(t3-t0,6))


        self.rot_est.append(self.rot)
        self.pos_est.append(self.pos)

        # Publish the map
        self.publishAll(msg.header.stamp)

        # Publish the odometry
        self.publishOdometry(msg.header.stamp)

        # Publish all the odometry estimates
        if self.pub_odom_all.get_num_connections() > 0:
            self.publishOdometryAll(msg.header.stamp)


    def getAllPoints(self):
        pts = []
        for block in self.blocks:
            temp = self.blocks[block].getPoints()
            if temp.shape[0] > 0:
                pts.append(temp)
        return np.concatenate(pts, axis=0)
    
    def getAllColours(self):
        colours = []
        for block in self.blocks:
            temp = self.blocks[block].getColours()
            if temp.shape[0] > 0:
                colours.append(temp)
        return np.concatenate(colours, axis=0)
    
    def getAllCounts(self):
        counts = []
        for block in self.blocks:
            temp = self.blocks[block].getCounts()
            if temp.shape[0] > 0:
                counts.append(temp)
        return np.concatenate(counts, axis=0)




    # Publish the current map as a PointCloud2
    def publishAll(self, stamp):
        pts = self.getAllPoints().astype(np.float32)
        if self.msg_reader.contains_colour:
            colours = self.getAllColours()
            pc_msg = self.msg_reader.npToRosMsg(pts, colours, stamp, seq=self.counter)
        else:
            pc_msg = self.msg_reader.npToRosMsg(pts, None, stamp, seq=self.counter)
        self.pub.publish(pc_msg)

        print('Nb points in global map: ', pts.shape[0])


    # Publish the current odometry estimate as a PoseStamped
    def publishOdometry(self, stamp):
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "map"
        pose_msg.header.stamp = stamp
        pose_msg.header.seq = self.counter
        pose_msg.pose.position.x = self.pos[0]
        pose_msg.pose.position.y = self.pos[1]
        pose_msg.pose.position.z = self.pos[2]
        rot = R.from_rotvec(self.rot).as_quat()
        pose_msg.pose.orientation.x = rot[0]
        pose_msg.pose.orientation.y = rot[1]
        pose_msg.pose.orientation.z = rot[2]
        pose_msg.pose.orientation.w = rot[3]
        self.pub_odom.publish(pose_msg)


    # Publish all the odometry estimates as a PoseArray
    def publishOdometryAll(self, stamp):
        pose_msg = PoseArray()
        pose_msg.header.frame_id = "map"
        pose_msg.header.stamp = stamp
        pose_msg.header.seq = self.counter
        for i in range(len(self.rot_est)):
            pose = Pose()
            pose.position.x = self.pos_est[i][0]
            pose.position.y = self.pos_est[i][1]
            pose.position.z = self.pos_est[i][2]
            rot = R.from_rotvec(self.rot_est[i]).as_quat()
            pose.orientation.x = rot[0]
            pose.orientation.y = rot[1]
            pose.orientation.z = rot[2]
            pose.orientation.w = rot[3]
            pose_msg.poses.append(pose)
        self.pub_odom_all.publish(pose_msg)


    # Perform the odometry registration
    def odometryRegistration(self, pts):
        print('Frame registration:')
        t0 = time.perf_counter()

        # Get the points in the blocks close to the points to register
        # Get the blocks close to the points
        block_tree = KDTree(np.asarray(tuple(self.blocks.keys()))*self.block_size + self.block_size/2.0)

        idx = block_tree.query(utils.transformPoints3D(pts,np.concatenate((self.rot, self.pos))), k = 1)
        # Get all the ids in a single array
        ids = idx[1]
        # Get the unique ids
        unique_ids = np.unique(ids)
        # Get the corresponding blocks ids
        blocks_ids = np.asarray(tuple(self.blocks.keys()))[unique_ids,:]
        # For each blocks gets the neighbouring block ids as well
        dim = np.size(blocks_ids, axis=1)
        neighbours = []
        #for b in range(np.size(blocks_ids, axis=0)):
        #    for x in range(3):
        #        for y in range(3):
        #            if dim == 2:
        #                neighbours.append((blocks_ids[b,0]+x-1, blocks_ids[b,1]+y-1))
        #            else:
        #                for z in range(3):
        #                    neighbours.append((blocks_ids[b,0]+x-1, blocks_ids[b,1]+y-1, blocks_ids[b,2]+z-1))

        if neighbours == []:
            all_block = blocks_ids
        else:
            neighbours = np.asarray(neighbours)
            all_block = np.concatenate((blocks_ids, neighbours), axis=0)

        all_block = np.unique(all_block, axis=0)

        # Get the points of the corresponding blocks
        pts_block = []
        for i in range(all_block.shape[0]):
            block_id = tuple(all_block[i,:])
            if block_id in self.blocks:
                temp = self.blocks[block_id].getPoints()
                if temp.shape[0] > 0:
                    pts_block.append(temp)

        pts_block = np.concatenate(pts_block, axis=0)

        # Voxel downsample the points to register
        self.curr_pts = utils.voxelDownsample(pts, voxel_size=self.voxel_size)




        # Create the distance field object (perform the alpha computation)
        self.dist_field = gpdf.gpDistFieldSE(pts_block, 2*self.voxel_size, sz = 0.01, use_keops = self.use_keops)
        t1 = time.perf_counter()
        print('\tNb local map points: ', pts_block.shape[0])
        print('\tTime to build distance field: ', round(t1-t0,6))
        print('\tNb points to register: ', self.curr_pts.shape[0])

        ## Perform the registration without loss function first
        #t0 = time.perf_counter()
        #if self.use_keops:
        #    fun = MemoizeJac(self.odometryCostNumerical)
        #else:
        #    fun = MemoizeJac(self.odometryCostAndJac)
        #jac = fun.derivative
        #result = opt.least_squares(fun, np.concatenate((self.rot, self.pos)), jac=jac, method='lm', verbose=0, max_nfev=3, ftol=1e-6, xtol=1e-6)
        #self.rot = result.x[0:3]
        #self.pos = result.x[3:6]
        #t1 = time.perf_counter()
        #print('\tTime to optimise the first (no loss function): ', round(t1-t0, 6))


        # Perform the registration with loss function
        t0 = time.perf_counter()
        if self.use_keops:
            fun = MemoizeJac(self.odometryCostNumerical)
            jac = fun.derivative
            result = opt.least_squares(fun, np.concatenate((self.rot, self.pos)), jac=jac, method='dogbox', verbose=0, max_nfev=100, loss='cauchy', f_scale=self.voxel_size, ftol=1e-3, xtol=1e-5)
        else:
            result = opt.least_squares(self.odometryCostFunction, np.concatenate((self.rot, self.pos)) , method='dogbox', verbose=0, max_nfev=100, loss='cauchy', f_scale=self.voxel_size, ftol=1e-3, xtol=1e-5)
        self.rot = result.x[0:3]
        self.pos = result.x[3:6]
        t1 = time.perf_counter()
        print('\tTime to optimise the second step (cauchy loss function): ', round(t1-t0, 6))


        return utils.transformPoints3D(pts, result.x)
    

    def odometryCostAndJac(self, x):
        pts_trans, pts_jac = utils.transformPoints3D(self.curr_pts, x, with_grad = True)
        dist, grad = self.dist_field.queryWithGrad(pts_trans)
        ## Normalize the gradient
        #grad_norm = np.linalg.norm(grad, axis=1)
        #mask = grad_norm > 0.001
        #grad[mask,:] = grad[mask,:]/grad_norm[mask,None]
        #grad[~mask,:] = 0.0
        jac = np.matmul(grad[:,None,:], pts_jac).squeeze()
        return dist.squeeze(), jac
        

    def odometryCostNumerical(self, x):
        epsilon = 1e-10
        nb_pts = self.curr_pts.shape[0]
        all_pts = np.empty((7*nb_pts,3))
        all_pts[0:nb_pts,:] = utils.transformPoints3D(self.curr_pts, x)
        for i in range(6):
            x_eps = x.copy()
            x_eps[i] += epsilon
            all_pts[(i+1)*nb_pts:(i+2)*nb_pts,:] = utils.transformPoints3D(self.curr_pts, x_eps)
        dist = self.dist_field.query(all_pts)

        jac = np.empty((nb_pts,6))
        for i in range(6):
            jac[:,i] = ((dist[(i+1)*nb_pts:(i+2)*nb_pts] - dist[0:nb_pts])/epsilon).squeeze()
        dist = dist[0:nb_pts]
        return dist.squeeze(), jac



    def odometryCostFunction(self, x):
        pts = utils.transformPoints3D(self.curr_pts, x)
        dist = self.dist_field.query(pts)
        return dist.squeeze()


    # Unit test the jacobian of the odometry cost function
    def testOdometryCostFunction(self):
        print("Testing odometry cost function")
        x = np.random.rand(6)
        eps = 0.0001
        cost, jac = self.odometryCostAndJac(x)
        jac_num = np.empty_like(jac)
        for i in range(6):
            x_eps = x.copy()
            x_eps[i] += eps
            cost_eps = self.odometryCostFunction(x_eps)
            jac_num[:,i] = (cost_eps-cost)/eps
        print("Max error ", np.max(np.abs(jac-jac_num)))
        print(np.stack((jac.ravel(), jac_num.ravel()), axis = 1))
