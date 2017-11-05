import json
import numpy as np
import argparse
import pdb
import math

MU1 = 5.0
MU2 = 10.0
NUM_OF_CARS = 10
CAR_LEN = 5.0
CAR_WIDTH = 3.0
HOR_VEL = 2.5
VER_VEL = 10.0
FREQUENCY = 10.0
MOVE_THRESH = 0.02

class car(object):
    '''
        (x_mu, y) represents the middle point of car's rear
    '''
    def __init__(self, x, y, car_len=CAR_LEN, car_width=CAR_WIDTH):
        self.length = car_len
        self.width = car_width
        self.x_mu = x
        self.x = self.x_mu + np.random.normal(loc=0.0, scale=0.1)
        self.y = y
        self.move_finish = True
        self.changing_lane = False
        self.target_loc = self.x_mu

        self.ver_v = VER_VEL
        self.t_interval = 1.0/FREQUENCY
        self.hor_v = 0.0
        self.traj = []
        self.traj.append([self.x, self.y, 0])

    def move(self, ver_v, hor_v, t_interval):
        self.x_mu += hor_v*t_interval
        self.y += ver_v*t_interval
        self.x = self.x_mu + np.random.normal(loc=0.0, scale=0.1)

    def step(self):
        if self.move_finish:
            rand = np.random.random()
            if rand < MOVE_THRESH:
                self.changing_lane = True
                self.move_finish = False
                if self.x_mu == MU1:
                    self.target_loc = MU2
                else:
                    self.target_loc = MU1

        if self.changing_lane:
            if self.target_loc == MU1:
                self.hor_v = -HOR_VEL
            else:
                self.hor_v = HOR_VEL
        else:
            self.hor_v = 0.0
        
        self.move(self.ver_v, self.hor_v, self.t_interval)
        last_t = self.traj[-1][2]

        self.traj.append([self.x, self.y, self.t_interval+last_t])
        
        if math.isclose(self.x_mu, self.target_loc, abs_tol=0.001):
            self.move_finish = True
            self.changing_lane = False

class car_model(object):
    def __init__(self, car_num):
        self.init(car_num)
    
    def init(self, car_num):
        self.car_num = car_num
        self.cars = []
        for i in range(self.car_num):
            x_mu, y = self.get_init_pos(i)
            self.cars.append(car(x_mu, y))

    def get_init_pos(self, num):
        rand = np.random.random()
        if rand < 0.5:
            x = MU1
        else:
            x = MU2

        y = 2*num*CAR_LEN + 0.5

        return x, y

    def step(self):
        for car in self.cars:
            car.step()

    def ret_traj(self):
        ret = dict()
        name = 'robot_'
        cnt = 0
        for car in self.cars:
            ret[name+str(cnt)] = car.traj
            cnt += 1
        return ret

if __name__=='__main__':
    cars = car_model(5)
    for _ in range(900):
        cars.step()        

    traj = cars.ret_traj()
    with open('future_pos.json', 'w') as fout:
        json.dump(traj, fout)