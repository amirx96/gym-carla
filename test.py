#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import gym
import gym_carla
import carla
import os
import random
from collections import deque
import tensorflow as tf
# from tf.keras import backend as bk
# from tf.keras.layers import Dense
# from tf.keras.models import Sequential
# from tf.keras.optimizers import Adam
# from tf.keras.losses import huber_loss
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as bk
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
import numpy as np


def main():
  # parameters for the gym_carla environment
  params = {
    'number_of_vehicles': 8,
    'number_of_walkers': 0,
    'display_size': 256,  # screen size of bird-eye render
    'max_past_step': 1,  # the number of past steps to draw
    'dt': 0.1,  # time interval between two frames
    'discrete': False,  # whether to use discrete control space
    'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
    'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
    'port': 2000,  # connection port
    'town': 'Town06',  # which town to simulate
    'task_mode': 'acc_1',  # mode of the task, [random, roundabout (only for Town03)]
    'max_time_episode': 1000,  # maximum timesteps per episode
    'max_waypt': 12,  # maximum number of waypoints
    'obs_range': 32,  # observation range (meter)
    'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
    'd_behind': 12,  # distance behind the ego vehicle (meter)
    'out_lane_thres': 2.0,  # threshold for out of lane
    'desired_speed': 20,  # desired speed (m/s)
    'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
    'display_route': True,  # whether to render the desired route
    'pixor_size': 64,  # size of the pixor labels
    'pixor': False,  # whether to output PIXOR observation
    'RGB_cam': True, # whether to use RGB camera sensor
  }

  # Set gym-carla environment
  env = gym.make('carla-v0', params=params)
  obs = env.reset()
  episode = 0
  while True:
    action = 1
    obs,r,done,info = env.step(action)

   #print(obs)
    if done:
      obs = env.reset()
      print("Episode %d, Reward % f" % (episode,r))
      episode +=1

if __name__ == '__main__':
  main()