#!/usr/bin/env python
import sys
from std_msgs.msg import Int32
import rospy 
import pygame

pygame.init()
pygame.display.set_mode((250, 250))

rospy.init_node('start_simu')
pub = rospy.Publisher('/start_simu', Int32, queue_size=1)
started = False

while not rospy.is_shutdown() and not started:
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            started = True
    
    keys = pygame.key.get_pressed()

    if(keys[pygame.K_s]):
        started = True
        t = rospy.get_rostime().secs
        pub.publish(t+1)
        
        
