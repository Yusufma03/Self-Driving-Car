#!/usr/bin/env python
import rospy 
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from nav_msgs.msg import *
import numpy as np
import json

class SpeedController:
    def __init__(self):
        self.path2config = rospy.get_param('~path2config', None)
        self.hz = rospy.get_param('~hz', 10)
        with open(self.path2config, 'r') as f:
            self.config = json.load(f)
        
        self.initPubs()
        self.rate = rospy.Rate(self.hz) 
        np.random.seed(self.config["random_seed"])
        self.simu_started = False

    def initPubs(self):
        self.robots = [] 
        robot_id = 0
        for i in range(self.config["num_lanes"]):
            num_car_lane = self.config['num_cars_per_lane'][i]
            for j in range(num_car_lane):
                key = 'robot_' + str(robot_id + 1)
                self.robots.append({"lane":i, "pub":rospy.Publisher(key + '/cmd_vel', Twist, queue_size=1)})
                robot_id += 1

    def update_vel(self, i):
        mean_vel = self.config["mean_speed_per_lane"][i]
        stdrr_vel = self.config["stdrr_speed_per_lane"][i]
        return np.random.normal(mean_vel, stdrr_vel)
    
    def start_simu(self, msg):
        self.simu_started = msg
        if self.simu_started:
            print("Starting simulation")

    def run(self):
        while not rospy.is_shutdown():
            if self.simu_started:
                # publish the speed for each robot
                for robot in self.robots:
                    vel = self.update_vel(robot["lane"])
                    self.send_control(robot["pub"], vel)
                self.rate.sleep()
            
    def send_control(self, robot_pub, vel):
        msg = Twist()
        msg.linear.x = vel
        msg.angular.z = 0
        robot_pub.publish(msg)

if __name__=='__main__':

    rospy.init_node('lane_controller')
    controller = SpeedController()
    rospy.Subscriber('/start_simu', Bool, controller.start_simu, queue_size=1)
    
    
    controller.run()
