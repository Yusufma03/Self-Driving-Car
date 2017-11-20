#!/usr/bin/env python
import json
import numpy as np

def update_vel(i, config):
    mean_vel = config["mean_speed_per_lane"][i]
    stdrr_vel = config["stdrr_speed_per_lane"][i]
    return np.random.normal(mean_vel, stdrr_vel)


with open('lane_config.json', 'r') as f:
    config = json.load(f)

DT = config["dt"]

nb_steps = config["nb_timesteps"]
np.random.seed(config["random_seed"])

y_lane = 0
poses = {}
lane_ids = {}

count = 1
for i in range(config["num_lanes"]):
    num_car_lane = config['num_cars_per_lane'][i]
    for j in range(num_car_lane):
        key = 'robot_' + str(count)
        x = config["car_start_poses"][i][j]
        poses[key] = [[x, y_lane, 0]]
        lane_ids[key] = i
        count += 1

    y_lane -= config["lane_width"]

for t_i in range(0,nb_steps-1):

    for key in poses.keys():
        vel = update_vel(lane_ids[key], config)
        x,y,t = poses[key][t_i]
        x += vel * DT
        t = round(DT + t, 2)
        poses[key].append([x,y,t])

with open('poses.json', 'w') as f:
    json.dump(poses, f)


