import gym
import gym_carla
import carla
import os
import random
from collections import deque

import numpy as np
from collections import namedtuple
from stable_baselines import DQN
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.deepq.policies import LnCnnPolicy

def main():
  # parameters for the gym_carla environment
  params = {
    'number_of_vehicles': 25,
    'number_of_walkers': 0,
    'display_size': 256,  # screen size of bird-eye render
    'max_past_step': 1,  # the number of past steps to draw
    'dt': 0.1,  # time interval between two frames
    'discrete': True,  # whether to use discrete control space
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
    'desired_speed': 16.67,  # desired speed (m/s)
    'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
    'display_route': True,  # whether to render the desired route
    'pixor_size': 64,  # size of the pixor labels
    'pixor': False,  # whether to output PIXOR observation
    'RGB_cam': True, # whether to use RGB camera sensor
  }
  solver_params = {
    'layers': [64, 64, 64],
    'alpha': 0.001,
    'gamma': 0.99,
    'epsilon': 0.1,
    'replay_memory_size': 500000,
    'update_target_estimator_every': 10000,
    'batch_size': 64,
  }
  # Set gym-carla environment
  env = gym.make('carla-v0', params=params)
  #check_env(env)
  obs = env.reset()

  model = DQN('LnMlpPolicy', env, learning_rate=1e-3, prioritized_replay=True, verbose=1,tensorboard_log="./dqn")
  model.learn(total_timesteps=35000,tb_log_name="75k")

  model.save("deepq_carla")

  del model # remove to demonstrate saving and loading

  model = DQN.load("deepq_carla")

  obs = env.reset()

  for i in range(100):
    while True:
      action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
      if dones:
        obs = env.reset()
        break
    # env.render()


if __name__ == '__main__':
  main()