#! /usr/bin/env python3
import rospy
import numpy as np
from pypcd import pypcd
import time
import open3d as o3d
from sensor_msgs.msg import PointCloud2


class pcFreqDownsampler:
    def __init__(self):
        rospy.init_node('pc_freq_downsample', anonymous=True)
        self.pc_pub = rospy.Publisher('/points_low_freq', PointCloud2, queue_size=10)
        self.pc_sub = rospy.Subscriber('/points', PointCloud2, self.pcCallback, queue_size=10)

        max_freq = rospy.get_param('~max_freq', 2)
        self.min_delta_t = 1.0/max_freq

        print('--Params-- Max frequency : ', max_freq)

        self.last_pc_time = rospy.Time(0,0)
        self.counter = 0



    def pcCallback(self, pc_msg):
        if (pc_msg.header.stamp - self.last_pc_time).to_sec() > self.min_delta_t:
            self.last_pc_time = pc_msg.header.stamp
            pc_msg.header.seq = self.counter
            self.pc_pub.publish(pc_msg)
            self.counter += 1


            
        

if __name__ == '__main__':

    pc_freq_downsampler = pcFreqDownsampler()
    rospy.spin()