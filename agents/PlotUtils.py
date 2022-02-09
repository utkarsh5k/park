from matplotlib import pyplot as plt 

class PlotHelper():
    def __init__(self, env_name, rl_agent_name, greedy_agent_name): 
        self.env_name = env_name
        self.rl_agent_name = rl_agent_name
        self.greedy_agent_name = greedy_agent_name

    def plot_comparison(self, rl_rewards, greedy_rewards, append = ""):
        x_axis = [i + 1 for i in range(len(rl_rewards))]
        rl_rewards_cumulative = self.get_cumulative_rewards(rl_rewards)
        greedy_rewards_cumulative = self.get_cumulative_rewards(greedy_rewards)

        plt.clf()
        plt.plot(x_axis, rl_rewards_cumulative, label = f"{self.rl_agent_name}")
        plt.plot(x_axis, greedy_rewards_cumulative, label = f"{self.greedy_agent_name}")
        plt.xlabel("Timesteps")
        plt.ylabel("Cumulative Rewards")
        plt.title(f"RL vs Greedy Performance: {self.env_name}")
        plt.legend()
        plt.savefig(f"../baseline-results/{self.env_name}_{self.rl_agent_name}_vs_{self.greedy_agent_name}{append}.png")

    def plot_rewards(self, rewards, greedy = False):
        x_axis = [i + 1 for i in range(len(rewards))]
        cumulative_rewards = self.get_cumulative_rewards(rewards)
        agent_name = self.rl_agent_name if not greedy else self.greedy_agent_name

        plt.clf()
        plt.plot(x_axis, cumulative_rewards)
        plt.xlabel("Timesteps")
        plt.ylabel("Cumulative Rewards")
        plt.title(f"Agent Performance: {agent_name}")
        plt.legend()
        plt.savefig(f"../baseline-results/{self.env_name}_{agent_name}.png")

    def get_cumulative_rewards(self, rewards):
        cumulative_rewards = [rewards[0]] 
        for i in range(len(rewards) - 1): 
            cumulative_rewards.append(rewards[i + 1] + cumulative_rewards[i]) 
        return cumulative_rewards
