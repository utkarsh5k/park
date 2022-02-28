from numpy import dtype
import park 
from PlotUtils import PlotHelper
from stable_baselines3 import PPO, A2C
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np 
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from agent_wrapper import ParkAgent
from park_graph_env import ParkGraphEnv

import warnings 
warnings.filterwarnings("ignore")

class GCNFeatureExtractor(BaseFeaturesExtractor): 
    def __init__(self, observation_space, num_node_features = 4): 
        super(GCNFeatureExtractor, self).__init__(observation_space, features_dim= 520)
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_node_features)

    def convert_adj_matrix_to_list(self, adj_matrix): 
        m, n = adj_matrix.shape
        adj_list = []
        for i in range(m):
            for j in range(n): 
                if adj_matrix[i][j] == 1: 
                    adj_list.append([i, j])
        
        return np.array(adj_list)

    def forward(self, obs):
        x = torch.tensor(obs['node_features'], dtype = torch.float)
        edge_index = torch.tensor(self.convert_adj_matrix_to_list(obs['adj_matrix'][0]), dtype = torch.long).t().contiguous()
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return torch.flatten(x, start_dim = 1)
        
class PGGCNAgent(): 
    def __init__(self, env): 
        self.env = env
        policy_kwargs = dict(
            features_extractor_class= GCNFeatureExtractor,
            #features_extractor_kwargs = dict(num_node_features = 4),
            activation_fn=torch.nn.ReLU,
            net_arch=[dict(pi=[16, 32], vf=[16, 32])])


        self.model = A2C("MultiInputPolicy", env, learning_rate = 0.0003, policy_kwargs = policy_kwargs, verbose=1)

    def train_model(self, load = False):
        if load: 
            self.load_saved_model()
            return 

        self.model.learn(total_timesteps = 1000)
        self.model.save("./trained_agents/tf_pggcn")

    def load_saved_model(self):
        self.model = A2C.load("./trained_agents/tf_pggcn")

class AlwaysAdmitAgent():
    def __init__(self):
        pass

    def predict(self, obs):
        return 1, None

class ModelRunner():
    def __init__(self):
        pass

    def run_model_on_env(self, env, model, trace_ids):
        rewards_all_traces = []
        for trace_id in trace_ids:
            print(f"Running trace #{trace_id}")
            obs = env.reset(trace_id, trace_id + 1)
            rewards = []
            done = False
            for _ in range(10000):
                action, _ = model.predict(obs)
                obs, reward, done, _ = env.step(action)
                rewards.append(reward)
                if done: 
                    obs = env.reset(trace_id, trace_id + 1)
            
            rewards_all_traces.append(rewards)

        rewards_all_traces = np.array(rewards_all_traces)
        return rewards_all_traces.mean(axis = 0)

if __name__ == '__main__':
    #env = ParkAgent('tf_placement_sim')
    env = ParkGraphEnv('tf_placement_sim')

    #print(env.observation_space)

    agent = PGGCNAgent(env)
    agent.train_model()
    

