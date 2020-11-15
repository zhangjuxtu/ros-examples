#!/usr/bin/env python

"""OpenCV feature detectors with ros CompressedImage Topics in python.
This example subscribes to a ros topic containing sensor_msgs CompressedImage. 
It converts the CompressedImage into a numpy.ndarray,
then detects and marks features in that image. 
It finally displays and publishs the new image - again as CompressedImage topic.
"""

import sys, time

#numpy and scipy
import numpy as np
from scipy.ndimage import filters
import cv2
import torch

#ROS libraries
import roslib
import rospy

#Ros messages
from sensor_msgs.msg import CompressedImage
# we do not use cv_bridge. It does not support CompressedImage in python
# from cv_bridge import CvBridge, CvBridgeError

import agent
import test
from parameters import Parameters

VERBOSE = False

class image_feature:
    def __init__(self):
        '''Initialize ros publisher, ros subscriber'''
        # topic where we publish
        self.image_pub = rospy.Publisher("/output/image_raw/compressed", CompressedImage, queue_size=5)
        #self.bridge = CvBridge()

        #subscribed topic 
        self.subscriber = rospy.Subscriber("lane_image_topic", CompressedImage, self.callback, queue_size = 1)
        if VERBOSE :
            print("subscribed to /camera/image/compressed")

        self.init_lanenet()

    def init_lanenet(self):
        self.lane_agent = agent.Agent()
        self.lane_agent.load_weights(640, "tensor(0.2298)")
        self.lane_agent.evaluate_mode()

        if torch.cuda.is_available():
            self.lane_agent.cuda()

    def callback(self, ros_data):
        '''Callback function of subscribed topic.
        Here image get converted and feature detected
        '''
        if VERBOSE :
            print("received image of type: %s"% ros_data.format)
        #direct conversion to CV2#
        np_arr = np.fromstring(ros_data.data, np.uint8)
        #image_np = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        test_image = cv2.resize(image_np, (512,256))/255.0
        test_image = np.rollaxis(test_image, axis=2, start=0)
        _, _, ti = test.test(self.lane_agent, np.array([test_image]))
        cv2.imshow('lane detection', ti[0])
        cv2.waitKey(2)
        
        # create compressedImage
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', ti[0])[1]).tostring()
        # publish new image
        self.image_pub.publish(msg)

def main(args):
    '''Initialize and cleanup ros node '''
    ic = image_feature()
    rospy.init_node('lane_detector', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("shutting dow ROS image feature detector module")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
