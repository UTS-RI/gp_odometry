import numpy as np
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import PointCloud2
from pypcd import pypcd
import open3d as o3d
import rospy

def skewSymmetric(v):
    if len(v.shape) == 1:
        return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    else:
        return np.array([[np.zeros(v.shape[0]), -v[:,2], v[:,1]],
                     [v[:,2], np.zeros(v.shape[0]), -v[:,0]],
                     [-v[:,1], v[:,0], np.zeros(v.shape[0])]]).transpose(2,0,1)

def leftSO3Jacobian(rot):
    theta = np.linalg.norm(rot)
    if theta < 1e-8:
        return np.eye(3)
    else:
        return np.sin(theta)/theta*np.eye(3) + (1-np.sin(theta)/theta)*rot[:,None]*rot[None,:]/theta + (1-np.cos(theta))/theta*skewSymmetric(rot)

def transformPoints3D(pts, state, with_grad = False):
    rot = state[0:3]
    pos = state[3:6]
    Rot = R.from_rotvec(rot).as_matrix()
    rot_pts = np.matmul(pts, Rot.T)
    pts_out = rot_pts + pos
    if with_grad:
        jac = np.zeros((pts.shape[0],pts.shape[1], 6))
        jac[:,:,0:3] = np.matmul(skewSymmetric(rot_pts), -leftSO3Jacobian(rot))
        jac[:,:,3:6] = np.eye(3)
        return pts_out, jac
    else:
        return pts_out


# Unit test for the gradients the transformPoints function
def testTransformPoints():
    print("Testing transformPoints")
    pts = np.random.rand(10,3)
    state = np.random.rand(6)
    pts_trans, jac = transformPoints3D(pts, state, with_grad = True)
    jac_num = np.empty_like(jac)
    eps = 0.0001
    for i in range(6):
        state_eps = state.copy()
        state_eps[i] += eps
        pts_eps = transformPoints3D(pts, state_eps)
        jac_num[:,:,i] = (pts_eps - pts_trans)/eps
    print(np.stack((jac.ravel(), jac_num.ravel()), axis = 1))
    print("Max error ", np.max(np.abs(jac-jac_num)))


# Data structure to store the data in sparse voxels
class Block:
    def __init__(self, id, voxel_size=0.1):
        self.id = id
        self.voxel_size = voxel_size
        self.point_sums = {}
        self.point_counts = {}
        self.colour_sums = {}
        self.dim = 0


    # Insert points in the block and return the number of new voxels
    def addPoints(self, pt, colours=None):
        
        thr = 0.2*self.voxel_size
        compute_alpha = False

        if self.dim == 0:
            self.dim = pt.shape[1]

        # Get the ids of the points
        ids = np.floor_divide(pt, self.voxel_size).astype(int)
        unique_ids = np.unique(ids, axis=0)

        # Add the points to each voxels
        new_count = 0
        for i in range(unique_ids.shape[0]):
            id = unique_ids[i,:]
            mask = np.all(ids == id, axis=1)
            pts = pt[mask,:]
            id = tuple(id)
            if id in self.point_sums:
                self.point_sums[id] += np.sum(pts, axis=0)
                self.point_counts[id] += pts.shape[0]
            else:
                self.point_sums[id] = np.sum(pts, axis=0)
                self.point_counts[id] = pts.shape[0]
                new_count += 1

            # Same with colours
            if colours is not None:
                colours_block = colours[mask,:]
                if id in self.colour_sums:
                    self.colour_sums[id] += np.sum(colours_block, axis=0)
                else:
                    self.colour_sums[id] = np.sum(colours_block, axis=0)
        

        return new_count

    # Return the points in the block
    def getPoints(self):
        return np.asarray(tuple(self.point_sums.values()))/np.asarray(tuple(self.point_counts.values()))[:,None]


    # Return the colours in the block
    def getColours(self):
        return np.asarray(tuple(self.colour_sums.values()))/np.asarray(tuple(self.point_counts.values()))[:,None]

    # Return the number of points in each voxel
    def getCounts(self):
        return np.asarray(tuple(self.point_counts.values()))

    # Prune the block according to simple/naive rules
    def prune(self):
        # For each voxel check how many neighbours are exist
        # If there are more than a certain amount of points in the neighbourhood, keep the voxel if the centroid of the neighbourhood is not too far from the voxel's centroid
        min_voxel_threshold = 4 if self.dim == 2 else 8
        new_point_sums = {}
        new_point_counts = {}
        new_colour_sums = {}
        for voxel in self.point_sums:
            
            voxel_count = 0
            pt_sum = np.zeros(self.dim)
            pt_count = 0
            for x in range(3):
                for y in range(3):
                    if self.dim == 2:
                        neighbour_id = (voxel[0]+x-1, voxel[1]+y-1)
                        if neighbour_id in self.point_sums:
                            voxel_count += 1
                            pt_sum += self.point_sums[neighbour_id]
                            pt_count += self.point_counts[neighbour_id]
                    else:
                        for z in range(3):
                            neighbour_id = (voxel[0]+x-1, voxel[1]+y-1, voxel[2]+z-1)
                            if neighbour_id in self.point_sums:
                                voxel_count += 1
                                pt_sum += self.point_sums[neighbour_id]
                                pt_count += self.point_counts[neighbour_id]

            if voxel_count > min_voxel_threshold:
                nn_centroid = pt_sum/pt_count
                voxel_centroid = self.point_sums[voxel]/self.point_counts[voxel]
                if np.linalg.norm(nn_centroid-voxel_centroid) < (1.0*self.voxel_size):
                    new_point_sums[voxel] = self.point_sums[voxel]
                    new_point_counts[voxel] = self.point_counts[voxel]
                    if voxel in self.colour_sums:
                        new_colour_sums[voxel] = self.colour_sums[voxel]
            elif voxel_count > 2:
                new_point_sums[voxel] = self.point_sums[voxel]
                new_point_counts[voxel] = self.point_counts[voxel]
                if voxel in self.colour_sums:
                    new_colour_sums[voxel] = self.colour_sums[voxel]
            
        self.point_sums = new_point_sums
        self.point_counts = new_point_counts
        self.colour_sums = new_colour_sums


class RosNpConverter:
    
    def __init__(self, msg, downsample_nb, voxel_size, max_range):
        
        self.downsample_nb = downsample_nb
        self.voxel_size = voxel_size
        self.max_range = max_range

        temp = pypcd.PointCloud.from_msg(msg)
        if 'rgb' in temp.fields:
            self.contains_colour = True



    def rosMsgToNp(self, msg, get_colours = False, store_raw_cropped = False):

        pc = pypcd.PointCloud.from_msg(msg)
        pts = np.array([pc.pc_data['x'], pc.pc_data['y'], pc.pc_data['z']]).T

        treat_colour = get_colours and self.contains_colour

        # Decode the colour information
        if treat_colour:
            colours = np.array([pc.pc_data['rgb']]).T.view(np.uint32)
            colours = np.array([(colours & 0xff0000) >> 16, (colours & 0x00ff00) >> 8, colours & 0x0000ff]).squeeze().T.astype(np.float32)

        # Remove NaNs
        if len(pts.shape) == 3:
            mask = np.isnan(pts[:,:,0])
            if treat_colour:
                colours = colours.transpose(1,0,2)
        else:
            mask = np.isnan(pts[:,0])
        pts = pts[~mask,:]
        if treat_colour:
            colours = colours[~mask,:]


        # Convert into 2D array if needed
        if len(pts.shape) == 3:
            pts = pts.reshape(-1,3)
            if treat_colour:
                colours = colours.reshape(-1,3)

        # Crop to max range
        if self.max_range > 0:
            mask = np.linalg.norm(pts, axis=1) < self.max_range
            pts = pts[mask,:]
            if treat_colour:
                colours = colours[mask,:]

        if store_raw_cropped:
            self.raw_cropped_pts = pts.copy()
            if treat_colour:
                self.raw_cropped_colours = colours.copy()
            else:
                self.raw_cropped_colours = None


        # Downsample
        if self.downsample_nb > 0 and pts.shape[0] > self.downsample_nb:
            idx = np.random.choice(pts.shape[0], self.downsample_nb, replace=False)
            pts = pts[idx,:]
            if treat_colour:
                colours = colours[idx,:]


        if get_colours:
            if self.contains_colour:
                return pts, colours
            else:
                return pts, None
        else:
            return pts

    def getLastRawCropped(self, get_colours = False):
        if get_colours:
            return self.raw_cropped_pts, self.raw_cropped_colours
        else:
            return self.raw_cropped_pts
        

    def npToRosMsg(self, pts, colours = None, stamp = None, frame_id = 'map', seq = 0):
        pts_temp = pts.astype(np.float32)
        if colours is not None:
            colours_temp = colours.astype(np.uint32)
            # Convert the colours (RGB 0-255)  to the format used by pypcd
            colours_pcd = np.zeros((colours_temp.shape[0]), dtype=np.uint32)
            colours_pcd = np.bitwise_or(colours_pcd, colours_temp[:,0] << 16)
            colours_pcd = np.bitwise_or(colours_pcd, colours_temp[:,1] << 8)
            colours_pcd = np.bitwise_or(colours_pcd, colours_temp[:,2])
            colours_pcd = colours_pcd.view(np.float32)
            pc_out = pypcd.PointCloud.from_array( np.hstack((pts_temp, colours_pcd[:,None])).view(np.dtype([('x', pts_temp.dtype), ('y', pts_temp.dtype), ('z', pts_temp.dtype), ('rgb',colours_pcd.dtype)])).squeeze())
        else:
            pc_out = pypcd.PointCloud.from_array(pts_temp.view(np.dtype([('x', pts_temp.dtype), ('y', pts_temp.dtype), ('z', pts_temp.dtype)])).squeeze())
        pc_msg = pc_out.to_msg()
        pc_msg.header.frame_id = frame_id
        pc_msg.header.stamp = stamp if stamp is not None else rospy.Time.now()
        pc_msg.header.seq = seq
        return pc_msg
    
def voxelDownsample(pts, colours = None, voxel_size = 0.02):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if colours is not None:
        pcd.colors = o3d.utility.Vector3dVector(colours/255)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    if colours is not None:
        return np.asarray(pcd.points), (np.asarray(pcd.colors)*255)
    else:
        return np.asarray(pcd.points)