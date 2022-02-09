import park 
import agents.agent_wrapper as agent_wrapper
from PlotUtils import PlotHelper
from stable_baselines3 import PPO, A2C
import torch
import numpy as np 

class A2CAgent(): 
    def __init__(self, env): 
        self.env = env
        policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[dict(pi=[16, 32], vf=[16, 32])])

        self.model = A2C("MlpPolicy", env, learning_rate = 0.0003, policy_kwargs = policy_kwargs)

    def train_model(self, load = False):
        if load: 
            self.load_saved_model()
            return 

        self.model.learn(total_timesteps = 1000000)
        self.model.save("./trained_agents/cache_a2c")

    def load_saved_model(self):
        self.model = A2C.load("./trained_agents/cache_a2c")

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
    env = agent_wrapper.ParkAgent('cache')

    rl_agent = A2CAgent(env)
    rl_agent.train_model(load = True)

    greedy_agent = AlwaysAdmitAgent()

    runner = ModelRunner()

    trace_ids = [np.random.randint(0, 1000) for _ in range(10)]
    plotter = PlotHelper('cache', 'a2c', 'LRU')
    
    rl_results = runner.run_model_on_env(env, rl_agent.model, trace_ids)
    plotter.plot_rewards(rl_results)
    greedy_results = runner.run_model_on_env(env, greedy_agent, trace_ids)
    env.close()

    plotter.plot_comparison(rl_results, greedy_results)

