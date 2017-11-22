from autocar_model import AutoCarModel
from pomcp import POMCP
from agent import Agent
import json
import numpy as np
import sys


def load_data():
    with open('poses.json', 'r') as fin:
        parsed = json.load(fin)
    return parsed

def get_agent_poses(dic, index):
    ret = [
        v[index][:2]
        for k, v in dic.items()
    ]
    return ret




if __name__=='__main__':
    data = load_data()
    index = 0
    autocar = AutoCarModel()
    autocar.load_config()
    with open('../ros-lanechanging/autocar/scripts/lane_config.json', 'r') as fin:
        parsed = json.load(fin)
        robot_start_x = parsed['autonomous_car_start_pos']
    robot_pos = [robot_start_x, 1]
    dump = []

    num_cars = 3
    road_len = 200
    num_lanes = 6


    autocar.start_state = [[[0 for x in range(num_lanes)] for y in range(road_len)] for z in range(num_cars)]
    autocar.start_state[0][100][3] = 1
    autocar.start_state[1][110][4] = 1
    autocar.start_state[2][120][5] = 1
    robot_pos = [100, 1]


    solver = POMCP


    agent = Agent(autocar, solver)
    action_seq = agent.run_pomcp(robot_pos)

    print(action_seq)


    pos_x = 100
    pos_y = 0


    for index in range(len(action_seq)):
        if action_seq[index] == 0:
            if pos_y >= -2 and pos_y <= 0:
                vel_y = 10
                vel_x = 20
                pos_y = pos_y + 1
            elif pos_y == 1:
                vel_y = 0
                vel_x = 20
                pos_y = pos_y
            else:
                vel_y = 10
                vel_x = 40
                pos_y = pos_y + 1
        elif action_seq[index] == 2:
            if pos_y >= -3 and pos_y <= -1:
                vel_y = -10
                vel_x = 40
                pos_y = pos_y - 1
            elif pos_y == -4:
                vel_y = 0
                vel_x = 40
                pos_y = pos_y
            else:
                vel_y = -10
                vel_x = 20
                pos_y = pos_y - 1
        elif action_seq[index] == 1:
            if pos_y >= -1 and pos_y <= 1:
                vel_y = 0
                vel_x = 20
                pos_y = pos_y
            else:
                vel_y = 0
                vel_x = 40
                pos_y = pos_y
        elif action_seq[index] == 4:
            vel_y = 0
            vel_x = 20
            pos_y = pos_y
        else:
            if pos_y >= -1 and pos_y <= 1:
                vel_y = 0
                vel_x = 20
                pos_y = pos_y
            else:
                vel_y = 0
                vel_x = 40
                pos_y = pos_y


        dump.append([vel_x, vel_y, index / 10.0])


    with open('cmds.json', 'w') as fout:
        json.dump(dump, fout)

