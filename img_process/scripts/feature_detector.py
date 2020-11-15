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

#OpenCV
import cv2

#ROS libraries
import roslib
import rospy

#Ros messages
from sensor_msgs.msg import CompressedImage
# we do not use cv_bridge. It does not support CompressedImage in python
# from cv_bridge import CvBridge, CvBridgeError

VERBOSE = False

class image_feature:
    def __init__(self):
        '''Initialize ros publisher, ros subscriber'''
        # topic where we publish
        self.image_pub = rospy.Publisher("/output/image_raw/compressed", CompressedImage, queue_size=5)
        #self.bridge = CvBridge()

        #subscribed topic 
        self.subscriber = rospy.Subscriber("image_topic_2", CompressedImage, self.callback, queue_size = 1)
        if VERBOSE :
            print("subscribed to /camera/image/compressed")

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

        #feature detection using CV2
        method = "GridFAST"
        fast = cv2.FastFeatureDetector_create()
        fast.setNonmaxSuppression(0)
        fast.setThreshold(10)
        time1 = time.time()

        #convert np image to grayscale
        #kp =  fast.detect(cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY))
        kp =  fast.detect(image_np, None)
        cv2.drawKeypoints(image_np, kp, image_np, color=(255,0,0))
        time2 = time.time()
        if VERBOSE :
            print("%s detector found : %s points in : % sec."%(method, len(featPoints), time2-time1))

        '''
        for featpoint in featpoints:
            x, y = featpoint.pt
            cv2.circle(image_np, (int(x)), (int(y)), 3, (0, 0, 255), -1)
        '''

        cv2.imshow('cv_img', image_np)
        cv2.waitKey(2)
        
        # create compressedImage
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', image_np)[1]).tostring()
        # publish new image
        self.image_pub.publish(msg)

def main(args):
    '''Initialize and cleanup ros node '''
    ic = image_feature()
    rospy.init_node('image_feature', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("shutting dow ROS image feature detector module")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
