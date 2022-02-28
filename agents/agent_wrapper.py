import sys
sys.path.insert(1, '../')

import park 
import gym 
import torch
import numpy as np

class ParkAgent(gym.Env):
    def __init__(self, env_name):
        self.env = park.make(env_name)
        self.action_space = self.get_gym_space(self.env.action_space)
        if env_name == 'tf_placement_sim':
            self.observation_space = gym.spaces.Box(low = 0, high = 1, shape = (1, 4), dtype = np.float)
        else:
            self.observation_space = self.get_gym_space(self.env.observation_space)

    def step(self, action):
        return self.env.step(action)
        
    def reset(self):
        return self.env.reset()

    # def reset(self, trace_min = 0, trace_max = 1000):
    #     return self.env.reset(trace_min, trace_max) 

    def get_gym_space(self, space):
        if type(space) == park.spaces.Box:
            return self.get_gym_box_space(space)
        elif type(space) == park.spaces.Discrete:
            return self.get_gym_discrete_space(space)
        elif type(space) == park.spaces.Tuple:
            return self.get_gym_tuple_space(space)
        
        print(type(space))
        raise NotImplementedError

    def get_gym_tuple_space(self, space): 
        print(space[0].sample())
        print(space[1])
        return gym.spaces.Tuple(space)

    def get_gym_box_space(self, space): 
        return gym.spaces.Box(
            low = space.low,
            high = space.high,
            shape = space.shape, 
            dtype = space.dtype
        )
    
    def get_gym_discrete_space(self, space):
        return gym.spaces.Discrete(space.n)

class ParkDiscreteSpace(gym.spaces.Discrete):
    def __init__(self, discrete_space):
        self.park_discrete_space = discrete_space
        self.n = discrete_space.n
    
    def sample(self):
        return self.park_discrete_space.sample()

    def contains(self, x):
        return self.park_discrete_space.contains(x)

class ParkBoxSpace(gym.spaces.Box):
    def __init__(self, box_space):
        self.park_box_space = box_space
        self.shape = box_space.shape
        self.high = box_space.high
        self.low = box_space.low
        self.struct = box_space.struct
        self.dtype = box_space.dtype
    
    def sample(self):
        return self.park_box_space.sample()

    def contains(self, x):
        return self.park_box_space.contains(x)

class ParkTupleSpace(gym.spaces.Tuple): 
    def __init__(self, space): 
        self.tuple_space = space
    
    def sample(self): 
        return self.tuple_space.sample()
    
    def contains(self, x): 
        return self.tuple_space.contains(x)

