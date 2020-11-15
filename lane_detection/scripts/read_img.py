#!/usr/bin/env python
from __future__ import print_function

import roslib
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError


import cv2
import sys
import numpy as np

def read_img():
    image_pub = rospy.Publisher('lane_image_topic', CompressedImage, queue_size=5)
    rospy.init_node('read_img', anonymous=True)
    rate = rospy.Rate(0.5) # 1hz
    rd_cunt = 0
    while not rospy.is_shutdown():
        lane_str = "lane_detection %s" % rospy.get_time()
        rospy.loginfo(lane_str)

        img_idx = rd_cunt % 7
        img_path = "/home/zhangjun/catkin_ws/src/lane_detection/img/" + "{}.jpg".format(img_idx)
        print('img_path: ', img_path)
        cv_image = cv2.imread(img_path)
        cv2.imshow("Image window", cv_image)
        cv2.waitKey(3)
        rd_cunt += 1

        # create compressedImage
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', cv_image)[1]).tostring()
 
        # publish new image
        image_pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        read_img()
    except rospy.ROSInterruptException:
        pass
