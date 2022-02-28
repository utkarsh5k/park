import numpy as np 
from stable_baselines3 import A2C 
import park 

import agents.agent_wrapper as agent_wrapper
from agents.PlotUtils import PlotHelper


class CacheAgent(): 
    def __init__(self): 
        self.model = A2C.load("../agents/trained_agents/cache_a2c")

class AlwaysAdmitAgent(): 
    def __init__(self):
        pass

    def predict(self, obs):
        return 1, None

class ModelRunner():
    def __init__(self):
        pass

    def run_model_on_env(self, env, model, trace_id):
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
        
        return np.array(rewards)


class CacheCrossEntropyAttack(): 
    def __init__(self):
        self.cache_size_min = 1
        self.cache_size_max = 20480
        self.obj_id_min = 0
        self.obj_id_max = 1000
        self.time_width_max = 200
        self.cache_entries_sizes = []
    def sanitize(self, size):
        if size < 0: 
            return -int(size)
        elif size < self.cache_size_min: 
            return self.cache_size_min
        
        return int(size) 

    def init_baseline_sequence(self):
        self.init_sequence(int(self.cache_size_max / 2), int(self.cache_size_max / 2))

    def init_sequence(self, size_mean, size_var, file_id = 1000):
        file_name = f"../park/envs/cache/traces/test_trace/test_{file_id}.tr"
        cur_time = 0
        cache_entries = {}
        self.cache_entries_sizes = [] 
        with open(file_name, 'w') as attack_file: 
            for _ in range(10000):
                obj_id = np.random.randint(0, self.obj_id_max)
                obj_time = cur_time + np.random.randint(1, self.time_width_max)
                try:
                    obj_size = cache_entries[obj_id]
                except KeyError: 
                    obj_size = self.sanitize(np.random.normal(size_mean, size_var))
                    cache_entries[obj_id] = obj_size
                file_str = f"{obj_time} {obj_id} {obj_size}\n"
                attack_file.write(file_str)
                self.cache_entries_sizes.append(obj_size)
                cur_time = obj_time
        
    def run_attack(self, num_attacks = 100):
        self.init_baseline_sequence()
        attack_file_id = 1000
        env = agent_wrapper.ParkAgent('cache')
        a2c_agent = CacheAgent().model
        greedy_agent = AlwaysAdmitAgent()
        model_runner = ModelRunner()

        prev_divergence = 0
        cur_mean = cur_std = int(self.cache_size_max / 2)
        all_greedy_total_rewards = []
        all_rl_total_rewards = []

        for _ in range(num_attacks):
            greedy_rewards = model_runner.run_model_on_env(env, greedy_agent, attack_file_id)
            rl_rewards = model_runner.run_model_on_env(env, a2c_agent, attack_file_id)

            rl_total_reward = np.sum(rl_rewards)
            greedy_total_reward = np.sum(greedy_rewards)

            all_greedy_total_rewards.append(greedy_total_reward)
            all_rl_total_rewards.append(rl_total_reward)

            if greedy_total_reward - rl_total_reward > prev_divergence: 
                prev_divergence = greedy_total_reward - rl_total_reward
                print(f"Achieved divergence {prev_divergence}")
            else: 
                print("Could not find an improvement of attack!")

            cur_mean, cur_std = np.mean(self.cache_entries_sizes), int(self.cache_size_max)
            self.init_sequence(cur_mean, cur_std)

        plotter = PlotHelper('cache', 'a2c', 'lru')
        plotter.plot_comparison(all_rl_total_rewards, all_greedy_total_rewards, "_attack")
        env.close()    

if __name__ == '__main__':
    attacker = CacheCrossEntropyAttack()
    attacker.run_attack(100) 
            

