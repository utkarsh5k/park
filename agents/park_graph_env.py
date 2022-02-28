import sys
sys.path.insert(1, '../')

import park 
import gym 
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data

class GraphDictTransform(): 
    def __init__(self, graph): 
        self.name_to_index = {}
        self.index_to_name = {}

        graph_nodes = graph.nodes()
        index = 0 

        node_features = [] 
        for node in graph_nodes: 
            self.name_to_index[node] = index
            self.index_to_name[index] = node 
            index += 1 
            node_features.append(graph_nodes[node]['feature'])

        node_features = np.array(node_features)
        edge_matrix = np.zeros((node_features.shape[0], node_features.shape[0]))

        graph_edges = graph.edges() 
        for edge in graph_edges:
            edge_matrix[self.name_to_index[edge[0]], self.name_to_index[edge[1]]] = 1
        
        self.obs_dict = {
            'node_features': node_features,
            'adj_matrix': edge_matrix
            }
        
    def get_obs_as_dict(self):
        return self.obs_dict 


class ParkGraphEnv(gym.Env):
    def __init__(self, env_name):
        self.env = park.make(env_name)
        self.action_space = self.get_gym_space(self.env.action_space)
        node_feature_space = gym.spaces.Box(
            low = self.env.observation_space.node_feature_space.low, 
            high = self.env.observation_space.node_feature_space.high, 
            shape = self.env.observation_space.node_feature_space.shape)
        num_nodes = self.env.observation_space.node_feature_space.shape[0]
        edge_feature_space = gym.spaces.Box(
            low = 0, 
            high = 1, 
            shape = (num_nodes, num_nodes))
        self.observation_space = gym.spaces.Dict(
            {
                'node_features': node_feature_space,
                'adj_matrix': edge_feature_space
            })

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        print(f"Reward: {reward}")
        return GraphDictTransform(obs).get_obs_as_dict(), reward, done, info
        
    def reset(self):
        obs = self.env.reset()
        return GraphDictTransform(obs).get_obs_as_dict()

    # def reset(self, trace_min = 0, trace_max = 1000):
    #     return self.env.reset(trace_min, trace_max) 

    def get_gym_space(self, space):
        if type(space) == park.spaces.Box:
            return self.get_gym_box_space(space)
        elif type(space) == park.spaces.Discrete:
            return self.get_gym_discrete_space(space)
        
        print(type(space))
        raise NotImplementedError

    def get_gym_box_space(self, space): 
        return gym.spaces.Box(
            low = space.low,
            high = space.high,
            shape = space.shape, 
            dtype = space.dtype
        )
    
    def get_gym_discrete_space(self, space):
        return gym.spaces.Discrete(space.n)