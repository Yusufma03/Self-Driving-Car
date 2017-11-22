# config_utils.py
# Aggregates configuration from lane_config.json and pomdp_config.json into a global config dictionary.

import json
import numpy as np

def load_configs():
    lane_config = json.load(open("lane_config.json", "r"))
    pomdp_config = json.load(open("pomdp_config.json", "r"))

    config = {}

    num_sublanes_per_lane = pomdp_config["num_sublanes_per_lane"]
    num_lanes = lane_config["num_lanes"]
    side_sublanes_crop = int(num_sublanes_per_lane/2)
    config["ny"] = num_sublanes_per_lane * num_lanes - 2 * side_sublanes_crop

    config["ndx"] = pomdp_config["num_cells_obs"]

    num_cars_per_lane = lane_config["num_cars_per_lane"]
    config["ncars"] = sum(num_cars_per_lane) + 1

    config["num_lanes"] = num_lanes
    config["num_sublanes_per_lane"] = num_sublanes_per_lane

    cars_lanes = [0]
    for lane_id in range(len(num_cars_per_lane)):
        nb_cars_lane = num_cars_per_lane[lane_id]
        for i in range(nb_cars_lane):
            cars_lanes.append(lane_id)
    config["cars_sublanes"] = (np.array(cars_lanes,np.int32)*num_sublanes_per_lane).tolist()


    lanes_speeds = lane_config["mean_speed_per_lane"]
    lanes_speeds_repeated = np.repeat(lanes_speeds, num_sublanes_per_lane)
    config["sublanes_speeds"] = lanes_speeds_repeated[side_sublanes_crop:-side_sublanes_crop].tolist()
    config["lanes_speeds"] = lanes_speeds

    dt = lane_config["dt"]
    cell_length = pomdp_config["cell_length"]

    config["lanes_speeds_cells"] = [int(round(speed * dt / cell_length)) for speed in config["lanes_speeds"]]
    config["sublanes_speeds_cells"] = [int(round(speed * dt / cell_length)) for speed in config["sublanes_speeds"]]

    lane_width = lane_config["lane_width"]
    config["sublane_width"] =  lane_width / num_sublanes_per_lane 

    trans_probas = lane_config["transition_probas"]
    trans_probas_repeated = np.repeat(trans_probas, num_sublanes_per_lane, axis=0)
    config["transition_probas"] = trans_probas_repeated[side_sublanes_crop:-side_sublanes_crop]

    config["dt"] = dt
    config["cell_length"] = cell_length
    config["obs_probas"] = pomdp_config["obs_probas"]
    config["rewards"] = pomdp_config["rewards"]
    config["discount"] = pomdp_config["discount"]
    config["nb_timesteps"] = lane_config["nb_timesteps"]
    config["lane_width"] = lane_width
    config["autonomous_car_start_pos"] = lane_config["autonomous_car_start_pos"]

    return config
