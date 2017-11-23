import json
import numpy as np
import random

def gen_config(file_name):
    CAR_NUM = [[0, 5], [0, 4]]

    folder_path = './configs/'

    cnt = 0

    config = {
        "nb_timesteps": 300,
        "dt": 0.1,
        "ros_dt_mult": 5,
        "num_lanes": 2,
        "lane_width": 3.0,
        "num_cars_per_lane": [2, 5],
        "car_start_poses": [[120, 140], [100, 95, 90, 85, 80]],
        "autonomous_car_start_pos": 102,
        "mean_speed_per_lane": [20, 40],
        "stdrr_speed_per_lane": [0, 0],
        "transition_probas": [[1.0], [0.4, 0.6]],
        "random_seed": 123
    }

    car_num = random.randint(0, 1)

    num_1, num_2 = CAR_NUM[car_num]
    locs_1 = random.sample(range(15), num_1)
    for i in range(len(locs_1)):
        locs_1[i] *= 2
    locs_2 = random.sample(range(15), num_2)
    for i in range(len(locs_2)):
        locs_2[i] *= 2
    loc = [locs_1, locs_2]
    auto_pos = random.randint(5, 15)

    config['num_cars_per_lane'] = CAR_NUM[car_num][0]
    config['car_start_poses'] = loc
    config['autonomous_car_start_pos'] = int(auto_pos)
    with open(folder_path + file_name + '.json', 'w') as fout:
        json.dump(config, fout)


for i in range(10):
    gen_config(str(i))
