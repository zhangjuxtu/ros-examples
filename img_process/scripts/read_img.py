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

class read_img:
    def __init__(self):
        self.image_pub = rospy.Publisher('image_topic_2', CompressedImage, queue_size=5)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.callback)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        (rows, cols, channels) = cv_image.shape
        if cols > 60 and rows >60:
            cv2.circle(cv_image, (50, 50), 10, 255)

        cv2.imshow("Image window", cv_image)
        cv2.waitKey(3)

        # create compressedImage
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', cv_image)[1]).tostring()
        # publish new image
        self.image_pub.publish(msg)

def main(args):
    ic = read_img()
    rospy.init_node('image_read', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("shutting dow")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
