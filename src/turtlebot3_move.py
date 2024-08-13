#!/usr/bin/env python3

import rospy
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

import math
import os

def read_file_path(filename):
    if not os.path.exists(filename):
        raise Exception(f"File {filename} does not exist")
    with open(filename, 'r') as file_name:
        lines = file_name.readlines()
    coordinates = [[round(float(x), 2), round(float(y), 2)] for x, y in [line.strip().split('\t') for line in lines]]
    return coordinates

#Defining the initial coordinates of the robot as defined in the launch file
x_coord = 0.0
y_coord = 0.0
theta = 0.0

def move_turtlebot_to_goal(goal_x_coord, goal_y_coord):
    publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=100)
    vel_msg = Twist()
    goal_reached = False
    rate = rospy.Rate(50)

    while not goal_reached and not rospy.is_shutdown():
        delta_y = goal_y_coord - y_coord
        delta_x = goal_x_coord - x_coord

        rotation = math.atan2(delta_y, delta_x)
        dist = math.sqrt(delta_x**2 + delta_y**2)

        # Proportional controller
        angular_error = rotation - theta
        print("error :  " , angular_error)
        print("dist: ", dist)
        vel_msg.angular.z = 2 * max(min(angular_error, 1), -1)
        vel_msg.linear.x = 0.6 * max(min(dist, 1), 0)

        if abs(angular_error) < 0.05 and dist < 0.05:
            goal_reached = True
            vel_msg.linear.x = 0.05
            vel_msg.angular.z = 0.0

        publisher.publish(vel_msg)
        rate.sleep()

def orientation_callback(vel_msg):
    global x_coord, y_coord, theta
    x_coord = vel_msg.pose.pose.position.x
    y_coord = vel_msg.pose.pose.position.y
    quaternion = (
        vel_msg.pose.pose.orientation.x,
        vel_msg.pose.pose.orientation.y,
        vel_msg.pose.pose.orientation.z,
        vel_msg.pose.pose.orientation.w
    )
    _, _, theta = euler_from_quaternion(quaternion)

if __name__ == "__main__":
    rospy.init_node('move_turtlebot')
    rospy.Subscriber('/odom', Odometry, orientation_callback)

    file_path = rospy.get_param('~file_path', '/home/sarin/catkin_ws/src/enpm661_final_project/project_path.txt') #change the file path as per your system
    poses = read_file_path(file_path)

    for pose in poses:
        goal_x, goal_y = pose
        move_turtlebot_to_goal(goal_x/100, goal_y/100)
    rospy.spin()
