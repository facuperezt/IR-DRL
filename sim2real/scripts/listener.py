#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import JointState


def callback(data):
    #rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
    print(data.transforms)
    #print(data.position)
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)
    print("subscribed")
    rospy.Subscriber("/tf", TFMessage, callback)
    

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()



#rewritten into a function
#import rospy
"""
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

def get_joint_position(hz):
    def callback(data):
        print(data.transforms[0].transform.translation)

    def listener():
        rospy.init_node('listener', anonymous=True)
        print("subscribed")
        rospy.Subscriber("/tf", TFMessage, callback)
        rate = rospy.Rate(hz)
        while not rospy.is_shutdown():
            rate.sleep()

    listener()
"""