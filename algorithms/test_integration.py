import sys 
sys.path.insert(1, '../')

import park 
import train.agent_wrapper as agent_wrapper
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, A2C

env = agent_wrapper.ParkAgent('load_balance')

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()

rewards = []
for i in range(10000):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if done: 
        print(reward)
        obs = env.reset()

env.close()