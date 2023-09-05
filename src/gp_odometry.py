#! /usr/bin/env python3
import rospy
import gp_odometer
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
import message_filters



if __name__ == '__main__':

    gp_odometer = gp_odometer.GPOdometer()
    pc_sub = message_filters.Subscriber("/points", PointCloud2)
    trans_sub = message_filters.Subscriber("/relative_pose", PoseStamped)
    ts = message_filters.TimeSynchronizer([pc_sub, trans_sub], 100)
    ts.registerCallback(gp_odometer.pcCallback)
    #rospy.Subscriber("/points", PointCloud2, gp_odometer.pcCallback, queue_size=1000)
    #rospy.Subscriber("/camera/rgb/points", PointCloud2, gp_odometer.pcCallback, queue_size=1000)
    rospy.spin()