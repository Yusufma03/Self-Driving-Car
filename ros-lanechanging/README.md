# ROS Simulator

## How to run the Simulator

```
cd autocar/scripts/
./start_simulator
```

Place the pygame window on top and press "S" to start the simulation.

## Configuration

Configuration for the simulator is in *autocar/scripts/lane_config.json* :

```
{
    "nb_timesteps":300,
    "num_lanes": 2,
    "num_cars_per_lane": [1, 1],
    "car_start_poses": [[20], [11]],
    "autonomous_car_start_pos":10,
    "mean_speed_per_lane": [20, 40],
    "stdrr_speed_per_lane": [0, 0],
    "random_seed":123
}
```

# Future position files

To generate future poses :

```
cd autocar/scripts/
./generate_poses.py
```

It will generate *poses.json* in *autocar/scripts/*. Format is same as in Assignment 3:

```
{
    ... ,
    "robot_i":[..., [x_pos, y_pos, t], ...],
    "robot_i+1":[..., [x_pos, y_pos, t], ...],
    ...
}
```

# Policy playback

To test your policy, put your velocity commands in *autocar/scripts/cmds.json*. Format is :

```
[ ..., [vel_i, t_i], [vel_i+1, t_i+1], ... ]
```

If the file exists, it will automatically loaded by the simulator.

# Coordinate systems

The middle of the left-most lane has X coordinate 0. Then the middles of other lanes from left to right have coordinates -3, -6, -9, etc.
The upstream edge of the road has Y coordinate 0.

The dimensions of the car are 2x1, with its origin at the center.


