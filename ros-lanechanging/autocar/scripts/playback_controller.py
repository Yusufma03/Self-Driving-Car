#!/usr/bin/env python
import rospy 
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from nav_msgs.msg import *
import numpy as np
import json
import os

class PlaybackController:
    
    def __init__(self, cmds):
        self.cmds = cmds
        self.path2config = rospy.get_param('~path2config', None)
        self.hz = rospy.get_param('~hz', 10)
        self.rate = rospy.Rate(self.hz)
        with open(self.path2config, 'r') as f:
            self.config = json.load(f)
        self.pub = rospy.Publisher('robot_0/cmd_vel', Twist, queue_size=1)
        self.simu_started = False
        self.t_i = 0
        self.end = False
        np.random.seed(self.config["random_seed"])
        
            
    def start_simu(self, msg):
        self.simu_started = msg
    
    def send_control(self, cmd):
        msg = Twist()
        msg.linear.x = cmd[0]
        msg.linear.y = cmd[1]
        msg.angular.z = 0
        self.pub.publish(msg)
            
    def run(self):
        while not rospy.is_shutdown() and not self.end:
            if self.simu_started:
                cmd = self.cmds[self.t_i]
                self.send_control(cmd)
                self.t_i += 1
                if self.t_i == len(self.cmds):
                    self.end = True
                
                self.rate.sleep()
        
    

if __name__=='__main__':
    
    rospy.init_node('playback_controller')
    path = rospy.get_param('~path2playback', None)

    with open(path,'r') as f:
        cmds = json.load(f)
            
        controller = PlaybackController(cmds)
        rospy.Subscriber('/start_simu', Bool, controller.start_simu, queue_size=1)
        controller.run()
