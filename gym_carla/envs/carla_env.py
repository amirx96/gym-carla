#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import division

import copy
import numpy as np
import pygame
import random
import time
from skimage.transform import resize

import gym
from gym import spaces
from gym.utils import seeding
import carla

from gym_carla.envs.render import BirdeyeRender
from gym_carla.envs.route_planner import RoutePlanner
from gym_carla.envs.misc import *
from simple_pid import PID

class CarlaEnv(gym.Env):
  """An OpenAI gym wrapper for CARLA simulator."""

  def __init__(self, params):
    # parameters
    self.display_size = params['display_size']  # rendering screen size
    self.max_past_step = params['max_past_step']
    self.number_of_vehicles = params['number_of_vehicles']
    self.number_of_walkers = params['number_of_walkers']
    self.dt = params['dt']
    self.task_mode = params['task_mode']
    self.max_time_episode = params['max_time_episode']
    self.max_waypt = params['max_waypt']
    self.obs_range = params['obs_range']
    self.lidar_bin = params['lidar_bin']
    self.d_behind = params['d_behind']
    self.obs_size = int(self.obs_range/self.lidar_bin)
    self.out_lane_thres = params['out_lane_thres']
    self.desired_speed = params['desired_speed']
    self.max_ego_spawn_times = params['max_ego_spawn_times']
    self.display_route = params['display_route']
    self.use_rgb_camera = params['RGB_cam']
    self.traffic_vehicles = []
    #self.discrete_acc = [-1,-0.5,0.0,0.5,1.0] # discrete actions for throttle
    self.discrete_vel = [-1.0, 0.0, 1.0] # discrete actions for velocity
    self.discrete_actions = params['discrete'] # boolean to use discrete or continoius action space
    self.cur_action = None
    self.pedal_pid = PID(0.7,0.01,0.0)
    self.pedal_pid.output_limits = (-1,1)
    self.rl_speed = 0.0
    self.pedal_pid.setpoint = 0.0
    # Destination
    if params['task_mode'] == 'acc_1':
      self.dests = [[592.1,244.7,0]] # stopping condition in Town 06
    else:
      self.dests = None
    self.idle_timesteps = 0
    # action and observation spaces

    # self.action_space = spaces.Box(np.array([params['continuous_accel_range'][0]], dtype=np.float32), np.array([params['continuous_accel_range'][1]], dtype=np.float32))  # acc
    #self.action_space = spaces.Discrete(len(self.discrete_acc))

    if self.discrete_actions:
      self.action_space = spaces.Discrete(3) # slow down -1 m/s, do nothing, speed up 1 m/s
    else:
      self.action_space = spaces.Box( np.array([0.0]),np.array([30.0])) # speed is continous from 0 m.s to 30 m.s
    self.observation_space = spaces.Box(np.array([0, 0, -1.0]), np.array([40, 40, 100]), dtype=np.float32)

    # Connect to carla server and get world object
    print('connecting to Carla server...')
    self.client = carla.Client('localhost', params['port'])
    self.client.set_timeout(10.0)
    self.world = self.client.load_world(params['town'])
    print('Carla server connected!')
    self.map = self.world.get_map()
    self.tm = self.client.get_trafficmanager(int(8000))
    self.tm_port = self.tm.get_port()
    #print('Traffic Manager Port ' + self.tm_port)
    # Set weather
    self.world.set_weather(carla.WeatherParameters.ClearNoon)

    # Get spawn points
    self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
    self.walker_spawn_points = []
    for i in range(self.number_of_walkers):
      spawn_point = carla.Transform()
      loc = self.world.get_random_location_from_navigation()
      if (loc != None):
        spawn_point.location = loc
        self.walker_spawn_points.append(spawn_point)

    # Create the ego vehicle blueprint
    self.ego_bp = self._create_vehicle_bluepprint(params['ego_vehicle_filter'], color='49,8,8')

    # Collision sensor
    self.collision_hist = [] # The collision history
    self.collision_hist_l = 1 # collision history length
    self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')


    # Camera sensor
    self.camera_img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
    self.camera_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
    self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
    # Modify the attributes of the blueprint to set image resolution and field of view.
    self.camera_bp.set_attribute('image_size_x', str(self.obs_size))
    self.camera_bp.set_attribute('image_size_y', str(self.obs_size))
    self.camera_bp.set_attribute('fov', '110')
    # Set the time in seconds between sensor captures
    self.camera_bp.set_attribute('sensor_tick', '0.02')

    # Set fixed simulation step for synchronous mode
    self.settings = self.world.get_settings()
    self.settings.fixed_delta_seconds = self.dt

    # Record the time of total steps and resetting steps
    self.reset_step = 0
    self.total_step = 0
    
    # Initialize the renderer
    self._init_renderer()


  def reset(self):
    # Clear sensor objects
    self._clear_all_sensors()
    self.collision_sensor = None
    self.camera_sensor = None
    self.idle_timesteps = 0
    # Delete sensors, vehicles and walkers
    self._clear_all_actors(['sensor.other.collision', 'sensor.lidar.ray_cast', 'sensor.camera.rgb', 'vehicle.*', 'controller.ai.walker', 'walker.*'])

    # Disable sync mode
    self._set_synchronous_mode(False)


    # Spawn vehicles in same lane as ego vehicle and ahead
    ego_vehicle_traffic_spawns = get_spawn_points_for_traffic(40,-5,self.map,self.number_of_vehicles)
    random.shuffle(ego_vehicle_traffic_spawns)
    count = self.number_of_vehicles
    if count > 0:
      for spawn_point in ego_vehicle_traffic_spawns:
        if self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4]):
          count -= 1
        if count <= 0:
          break
    while count > 0:
      if self._try_spawn_random_vehicle_at(random.choice(ego_vehicle_traffic_spawns), number_of_wheels=[4]):
        count -= 1

    # set autopilot for all traffic vehicles:

    batch = []
    for vehicle in self.traffic_vehicles:
      batch.append(carla.command.SetAutopilot(vehicle,True))
    self.client.apply_batch_sync(batch)

    # Get actors polygon list
    self.vehicle_polygons = []
    vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
    self.vehicle_polygons.append(vehicle_poly_dict)
    self.walker_polygons = []
    walker_poly_dict = self._get_actor_polygons('walker.*')
    self.walker_polygons.append(walker_poly_dict)

    # Spawn the ego vehicle
    ego_spawn_times = 0
    while True:
      if ego_spawn_times > self.max_ego_spawn_times:
        self.reset()

      if self.task_mode == 'acc_1':
        wpt1 = get_waypoint_for_ego_spawn(road_id=39,lane_id=-5,s=0,map=self.map)
        transform = wpt1.transform
        transform.location.z += 2.0
      if self._try_spawn_ego_vehicle_at(transform):
        break
      else:
        print('trying to spawn %d' % ego_spawn_times)
        ego_spawn_times += 1
        time.sleep(0.1)

    # Add collision sensor
    self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego)
    self.collision_sensor.listen(lambda event: get_collision_hist(event))
    def get_collision_hist(event):
      impulse = event.normal_impulse
      intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
      self.collision_hist.append(intensity)
      if len(self.collision_hist)>self.collision_hist_l:
        self.collision_hist.pop(0)
    self.collision_hist = []


    # Add camera sensor
    if self.use_rgb_camera:
      self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego)
      self.camera_sensor.listen(lambda data: get_camera_img(data))
      def get_camera_img(data):
        array = np.frombuffer(data.raw_data, dtype = np.dtype("uint8"))
        array = np.reshape(array, (data.height, data.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.camera_img = array

    # Update timesteps
    self.time_step=0
    self.reset_step+=1

    # Enable sync mode
    self.settings.synchronous_mode = False
    if not self.use_rgb_camera:
      self.settings.no_rendering_mode = True
    self.world.apply_settings(self.settings)

    self.routeplanner = RoutePlanner(self.ego, self.max_waypt)
    self.waypoints, _, self.vehicle_hazards = self.routeplanner.run_step()
    # Set ego information for render
    self.birdeye_render.set_hero(self.ego, self.ego.id)

    return self._get_obs()
  
  def step(self, action):
    # Calculate acceleration and steering
    #acc = self.discrete_acc[action]
    v = self.ego.get_velocity()
    speed = np.sqrt(v.x**2 + v.y**2)

    if speed < 1e-1:
      self.idle_timesteps +=1
    else:
      self.idle_timesteps = 0
    
    if self.discrete_actions:
      self.rl_speed += self.discrete_vel[action] # [-1.0 0.0, 1.0]
    else:
      self.rl_speed = action # range from 0 to 30
    self.rl_speed = np.clip(self.rl_speed,0.0,30.0)

    pid = self.pedal_pid( -(self.rl_speed-speed))
    acc = pid
    # acc = self.pedal_pid(speed)


    # Convert acceleration to throttle and brake
    if acc > 0:
      throttle = np.clip(acc,0,1)
      brake = 0
    else:
      throttle = 0
      brake = np.clip(-acc,0,1)
    #print(acc,speed,self.desired_speed-speed)
    #print("rl-speed %.2f, pid %.2f, acc %.2f" % (self.rl_speed  , pid, acc) )



    # Apply control
    act = carla.VehicleControl(throttle=float(throttle), steer=0.0, brake=float(brake))
    self.ego.apply_control(act)

    self.world.tick()

    # Append actors polygon list
    vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
    self.vehicle_polygons.append(vehicle_poly_dict)
    while len(self.vehicle_polygons) > self.max_past_step:
      self.vehicle_polygons.pop(0)
    walker_poly_dict = self._get_actor_polygons('walker.*')
    self.walker_polygons.append(walker_poly_dict)
    while len(self.walker_polygons) > self.max_past_step:
      self.walker_polygons.pop(0)

    # route planner
    self.waypoints, _, self.vehicle_hazards = self.routeplanner.run_step()
    #print(self.vehicle_hazards)
    # state information
    info = {
      'waypoints': self.waypoints,
      
    }
    # Update timesteps
    self.time_step += 1
    self.total_step += 1

    return (self._get_obs(), self._get_reward(), self._terminal(), copy.deepcopy(info))

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def render(self, mode):
    pass

  def _create_vehicle_bluepprint(self, actor_filter, color=None, number_of_wheels=[4]):
    """Create the blueprint for a specific actor type.

    Args:
      actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.

    Returns:
      bp: the blueprint object of carla.
    """
    blueprints = self.world.get_blueprint_library().filter(actor_filter)
    blueprint_library = []
    for nw in number_of_wheels:
      blueprint_library = blueprint_library + [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == nw]
    bp = random.choice(blueprint_library)
    if bp.has_attribute('color'):
      if not color:
        color = random.choice(bp.get_attribute('color').recommended_values)
      bp.set_attribute('color', color)
    return bp

  def _init_renderer(self):
    """Initialize the birdeye view renderer.
    """
    pygame.init()
    self.pyfont = pygame.font.SysFont(None, 22)
    self.display = pygame.display.set_mode(
    (self.display_size * 3, self.display_size),
    pygame.HWSURFACE | pygame.DOUBLEBUF)

    pixels_per_meter = self.display_size / self.obs_range
    pixels_ahead_vehicle = (self.obs_range/2 - self.d_behind) * pixels_per_meter
    birdeye_params = {
      'screen_size': [self.display_size, self.display_size],
      'pixels_per_meter': pixels_per_meter,
      'pixels_ahead_vehicle': pixels_ahead_vehicle
    }
    self.birdeye_render = BirdeyeRender(self.world, birdeye_params)

  def _set_synchronous_mode(self, synchronous = True):
    """Set whether to use the synchronous mode.
    """
    self.settings.synchronous_mode = synchronous
    self.world.apply_settings(self.settings)

  def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
    """Try to spawn a surrounding vehicle at specific transform with random bluprint.

    Args:
      transform: the carla transform object.

    Returns:
      Bool indicating whether the spawn is successful.
    """
    blueprint = self._create_vehicle_bluepprint('vehicle.*', number_of_wheels=number_of_wheels)
    blueprint.set_attribute('role_name', 'autopilot')
    vehicle = self.world.try_spawn_actor(blueprint, transform)
    if vehicle is not None:
      self.traffic_vehicles.append(vehicle)
      #vehicle.set_autopilot()
      # batch = []
      # batch.append(carla.command.SetAutopilot(vehicle,True))
      # self.client.apply_batch_sync(batch) # not how this is supposed to be done but oh well
      #vehicle.enable_constant_velocity(np.random.uniform(low=18.0,high=30.0))
      vehicle.set_autopilot(True,self.tm_port)
      high = np.random.uniform(low=-20,high=-1)
      low = np.random.uniform(low=80,high=99)

      
      self.tm.vehicle_percentage_speed_difference(vehicle,random.choice([high,low])) # percentage difference between posted speed and vehicle speed. Negative is greater
      return True
    return False



  def _try_spawn_ego_vehicle_at(self, transform):
    """Try to spawn the ego vehicle at specific transform.
    Args:
      transform: the carla transform object.
    Returns:
      Bool indicating whether the spawn is successful.
    """
    vehicle = None
    # Check if ego position overlaps with surrounding vehicles
    overlap = False
    for idx, poly in self.vehicle_polygons[-1].items():
      poly_center = np.mean(poly, axis=0)
      ego_center = np.array([transform.location.x, transform.location.y])
      dis = np.linalg.norm(poly_center - ego_center)
      if dis > 8:
        continue
      else:
        overlap = True
        print('overlapping vehicle when trying to spawn')
        break

    if not overlap:
      vehicle = self.world.try_spawn_actor(self.ego_bp, transform)

    if vehicle is not None:
      self.ego=vehicle

      # batch = []
      # batch.append(carla.command.SetAutopilot(self.ego,False))
      # self.client.apply_batch_sync(batch)
      # self.tm.vehicle_percentage_speed_difference(self.ego,-30)
      #self.tm.distance_to_leading_vehicle(self.ego,20.0)
      #self.tm.set_global_distance_to_leading_vehicle(0.0)
      # self.tm.auto_lane_change(self.ego,False)

      return True
    print ('could not spawn vehicle')
    return False

  def _get_actor_polygons(self, filt):
    """Get the bounding box polygon of actors.

    Args:
      filt: the filter indicating what type of actors we'll look at.

    Returns:
      actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
    """
    actor_poly_dict={}
    for actor in self.world.get_actors().filter(filt):
      # Get x, y and yaw of the actor
      trans=actor.get_transform()
      x=trans.location.x
      y=trans.location.y
      yaw=trans.rotation.yaw/180*np.pi
      # Get length and width
      bb=actor.bounding_box
      l=bb.extent.x
      w=bb.extent.y
      # Get bounding box polygon in the actor's local coordinate
      poly_local=np.array([[l,w],[l,-w],[-l,-w],[-l,w]]).transpose()
      # Get rotation matrix to transform to global coordinate
      R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
      # Get global bounding box polygon
      poly=np.matmul(R,poly_local).transpose()+np.repeat([[x,y]],4,axis=0)
      actor_poly_dict[actor.id]=poly
    return actor_poly_dict

  def _get_obs(self):
    """Get the observations."""
    ## Birdeye rendering
    self.birdeye_render.vehicle_polygons = self.vehicle_polygons
    self.birdeye_render.walker_polygons = self.walker_polygons
    self.birdeye_render.waypoints = self.waypoints

    # birdeye view with roadmap and actors
    birdeye_render_types = ['roadmap', 'actors']
    if self.display_route:
      birdeye_render_types.append('waypoints')
    self.birdeye_render.render(self.display, birdeye_render_types)
    birdeye = pygame.surfarray.array3d(self.display)
    birdeye = birdeye[0:self.display_size, :, :]
    birdeye = display_to_rgb(birdeye, self.obs_size)

    # State observation
    ego_trans = self.ego.get_transform()
    ego_x = ego_trans.location.x
    ego_y = ego_trans.location.y
    ego_yaw = ego_trans.rotation.yaw/180*np.pi
    lateral_dis, w = get_preview_lane_dis(self.waypoints, ego_x, ego_y)
    delta_yaw = np.arcsin(np.cross(w, 
      np.array(np.array([np.cos(ego_yaw), np.sin(ego_yaw)]))))
    v = self.ego.get_velocity()
    speed = np.sqrt(v.x**2 + v.y**2)


    lead_vehicle = None
    if len(self.vehicle_hazards) > 0:
      dist_to_lead= 100
      for i in range(len(self.vehicle_hazards)):
        hazard_loc = self.vehicle_hazards[i].get_transform()
        dist_to_lead_ = ( (hazard_loc.location.x - ego_x)**2 + (hazard_loc.location.y - ego_y)**2)**(1/2)
        if dist_to_lead_ < dist_to_lead:
          dist_to_lead = dist_to_lead_
          lead_vehicle = self.vehicle_hazards[i]
    else:
      dist_to_lead = -1

    if lead_vehicle is not None:
      lead_v = lead_vehicle.get_velocity()
      lead_speed = np.sqrt(lead_v.x**2 + lead_v.y**2)
    else:
      lead_speed = -1

    
    state = np.array([speed, lead_speed, dist_to_lead])
    ## Get leading vehicle info
    



    ## Display camera image
    if self.use_rgb_camera:
      camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255
      camera_surface = rgb_to_display_surface(camera, self.display_size)
      self.display.blit(camera_surface, (self.display_size * 2, 0))
    else:
      camera = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)

    

    ## Display info statistics
    self.display.blit(self.pyfont.render('speed ' + str(round(speed,1)) + ' m/s', True, ( 255, 0, 0)),(self.display_size*1,80))
    self.display.blit(self.pyfont.render('lead_dist ' + str(round(dist_to_lead,1)) + ' m', True, ( 255, 255, 0)), (self.display_size*1,120))
    self.display.blit(self.pyfont.render('lead_v ' + str(round(lead_speed,1)) + ' m/s', True, ( 255, 255, 0)), (self.display_size*1,100))
    # Display on pygame
    pygame.display.flip()






    return state

  def _get_reward(self):
    """Calculate the step reward."""
    # reward for speed tracking
    v = self.ego.get_velocity()
    speed = np.sqrt(v.x**2 + v.y**2)
    r_speed = -abs(speed - self.desired_speed)
    
    # reward for collision
    r_collision = 0
    if len(self.collision_hist) > 0:
      r_collision = -1

    # reward for steering:

    # reward for out of lane
    ego_x, ego_y = get_pos(self.ego)
    dis, w = get_lane_dis(self.waypoints, ego_x, ego_y)
    r_out = 0
    # if abs(dis) > self.out_lane_thres:
    #   r_out = -1

    # longitudinal speed
    lspeed = np.array([v.x, v.y])
    lspeed_lon = np.dot(lspeed, w)

    # cost for too fast
    r_fast = 0
    if lspeed_lon > self.desired_speed:
      r_fast = -1
    # cost for too slow

    # cost for too fast
    r_slow = 0
    if lspeed_lon < self.desired_speed:
      r_slow = -1
    # cost for lateral acceleration
    r_lat = - abs(self.ego.get_control().steer) * lspeed_lon**2

    # cost for idling 
    r_idle = -1*self.idle_timesteps

    r = 200*r_collision + 1*lspeed_lon + 10*r_fast + 5*r_slow  - 0.1 + r_idle + 10*r_speed 
    print('reward [collision %.2f] [distance %.2f] [overspeed %.2f] [underspeed %.2f] [idle %f] [speed mismatch %.2f]' %  (200*r_collision , 1*lspeed_lon , 10*r_fast , 5*r_slow , r_idle , 15*r_speed) )
    return r

  def _terminal(self):
    """Calculate whether to terminate the current episode."""
    # Get ego state
    ego_x, ego_y = get_pos(self.ego)

    # If collides
    if len(self.collision_hist)>0: 
      return True

    # If reach maximum timestep
    if self.time_step>self.max_time_episode:
      return True

    # If at destination
    if self.dests is not None: # If at destination
      for dest in self.dests:
        if np.sqrt((ego_x-dest[0])**2+(ego_y-dest[1])**2)<4:
          return True

    # If out of lane
    dis, _ = get_lane_dis(self.waypoints, ego_x, ego_y)
    if abs(dis) > self.out_lane_thres:
      return True

    # if stopped for a vehicle ahead
    # TODO 
    if self.idle_timesteps > 500:
      print('terminal due to too many idle timesteps')
      return True
    return False

  def _clear_all_actors(self, actor_filters):
    """Clear specific actors."""
    for actor_filter in actor_filters:
      for actor in self.world.get_actors().filter(actor_filter):
        if actor.is_alive:
          if actor.type_id == 'controller.ai.walker':
            actor.stop()
          actor.destroy()
  def _clear_all_sensors(self):
    try:
      self.collision_sensor.destroy()
      self.camera_sensor.destroy()  
    except:
      pass