import json
import numpy as np

CAR_NUM = [[0, 5], [0, 4], [0, 6]]

folder_path='./configs/'

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
    "transition_probas": [[1.0], [0.2, 0.8]],
    "random_seed": 123
}

car_num = np.random.choice(np.arange(len(CAR_NUM)), 1)[0]

num_1, num_2 = CAR_NUM[car_num]
locs_1 = np.random.choice(np.arange(15).astype(np.int32), size=num_1, replace=False) * 2
locs_1 = list(locs_1.astype(int))
locs_2 = np.random.choice(np.arange(15).astype(np.int32), size=num_2, replace=False) * 2
locs_2 = list(locs_2.astype(int))
loc = [locs_1, locs_2]
auto_pos = np.random.randint(low=5, high=15)

config['num_cars_per_lane'] = CAR_NUM[car_num][0]
config['car_start_poses'] = loc
config['autonomous_car_start_pos'] = int(auto_pos)
import pdb; pdb.set_trace()
with open('lane_config.json', 'w') as fout:
    json.dump(config, fout)



