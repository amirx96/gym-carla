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
from collections import namedtuple
from collections import deque

class DQN():
  def __init__(self,env,options):
    assert (str(env.action_space).startswith('Discrete') or
    str(env.action_space).startswith('Tuple(Discrete')), str(self) + " cannot handle non-discrete action spaces"
    self.options = options
    self.env = env
    self.state_size = self.env.observation_space.shape[0]
    
    self.action_size = self.env.action_space.n
    print("state size:", self.state_size)
    print('action_size' , self.action_size)
    self.model = self._build_model()
    self.target_model = self._build_model()
    self.update_every = 0
    self.replay = deque([],maxlen=self.options['replay_memory_size'])
    self.e_greedy_policy = self.make_epsilon_greedy_policy()

  def _build_model(self):
      state_size = self.state_size
      action_size = self.action_size
      layers = self.options['layers']
      # Neural Net for Deep-Q learning Model
      model = Sequential()

      model.add(Dense(layers[0], input_dim=state_size, activation='relu'))
      for l in layers:
          model.add(Dense(l, activation='relu'))
      model.add(Dense(action_size, activation='linear'))
      model.compile(loss=Huber(),
                    optimizer=Adam(lr=self.options['alpha']))
      return model

  def update_target_model(self):
      # copy weights from model to target_model
      self.target_model.set_weights(self.model.get_weights())
  def make_epsilon_greedy_policy(self):
      """
      Creates an epsilon-greedy policy based on the given Q-function approximator and epsilon.
      Returns:
          A function that takes a state as input and returns a vector
          of action probabilities.
      """

      def policy_fn(state):
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        nA = self.action_size
        #print(state.reshape([1,-1]).shape)
        Q = self.model.predict(state.reshape([1,-1]))
        A_str =   np.argmax(Q)
        eps = self.options['epsilon']
        pi_a_s =  np.ones(nA, dtype=float) * eps / nA

        pi_a_s[A_str] += (1-eps)
        return pi_a_s

      return policy_fn
  def train_episode(self):
      """
      Perform a single episode of the Q-Learning algorithm for off-policy TD
      control using a DNN Function Approximation.
      Finds the optimal greedy policy while following an epsilon-greedy policy.
      Use:
          self.options.experiment_dir: Directory to save DNN summaries in (optional)
          self.options.replay_memory_size: Size of the replay memory
          self.options.update_target_estimator_every: Copy parameters from the Q estimator to the
              target estimator every N steps
          self.options.batch_size: Size of batches to sample from the replay memory
          self.env: OpenAI environment.
          self.options.gamma: Gamma discount factor.
          self.options.epsilon: Chance the sample a random action. Float betwen 0 and 1.
          new_state, reward, done, _ = self.step(action): To advance one step in the environment
          state_size = self.env.observation_space.shape[0]
          self.model: Q network
          self.target_model: target Q network
          self.update_target_model(): update target network weights = Q network weights
      """

      # Reset the environment
      state = self.env.reset()

      ################################
      #   YOUR IMPLEMENTATION HERE   #
      ################################
      M = self.options['replay_memory_size']
      C = self.options['update_target_estimator_every']
      bsize = self.options['batch_size']
      eps = self.options['epsilon']
      gamma = self.options['gamma']
      i = 0
      for t in range(1000):

          e_probs = self.e_greedy_policy(state)
          action = np.random.choice(np.arange(len(e_probs)),p=e_probs)
          next_state, reward, done, _ = self.env.step(action)
          
          self.replay.append((state,action,reward,next_state,done)) 
          
          actual_bsize = min(len(self.replay),bsize)
          mini_batch = random.sample(self.replay,actual_bsize) # sample a random subset of the replay memory according to batchsize


          ## VECTORIZED METHOD (FASTER)
          mb_states, mb_actions, mb_rewards, mb_nextstates, mb_done = map(np.array,zip(*mini_batch))
          model_qs = self.model.predict(mb_states)
          target_qs = np.amax(self.target_model.predict(mb_nextstates),axis=1) # basically need fit the model against target
          yjs = mb_rewards + np.invert(mb_done).astype('float32') * gamma * target_qs # invert done to basically add gamma*target_q_value if next step is not done
          model_target_update = model_qs
          for idx,yj in enumerate(yjs):
              model_target_update[idx,mb_actions[idx]] = yj

          if t < 30: # initally the training_batch size is too large for update the gradients reliabily. If I set it to 1 always, then it takes sooooooo long to train
              train_bsize = 1
          else:
              train_bsize = actual_bsize
          self.model.fit(x=mb_states,y=model_target_update,verbose=0,batch_size=train_bsize,shuffle=False,epochs=1)

          
          if self.update_every % C == 0:
              self.update_target_model()
          self.update_every +=1
          if done:
              break
          state = next_state
      return reward
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
    'RGB_cam': False, # whether to use RGB camera sensor
  }
  solver_params = {
    'layers': [64, 64, 64],
    'alpha': 0.001,
    'gamma': 0.99,
    'epsilon': 0.1,
    'replay_memory_size': 500000,
    'update_target_estimator_every': 10000,
    'batch_size': 128,
  }
  # Set gym-carla environment
  env = gym.make('carla-v0', params=params)
  obs = env.reset()
  episode = 0
  solver = DQN(env,solver_params)
  num_episodes = 2000
  for episode in range(num_episodes):
    reward = solver.train_episode()
    print("Episode %d, Reward % f" % (episode,reward))
  # while True:
  #   action = 4
  #   next_state,reward,done,_ = env.step(action)
  #   #print(next_state)
  #  #print(obs)
  #   if done:
  #     obs = env.reset()
  #     print("Episode %d, Reward % f" % (episode,r))
  #     episode +=1

if __name__ == '__main__':
  main()