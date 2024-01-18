import numpy as np

class Agent:
    def __init__(self):
        self.best_reward_per_episode = []  # List to store best reward per episode
        self.min_reward = -10  # Define the minimum reward value
        self.max_reward = 10  # Define the maximum reward value

    def step(self, current_obs, env, grid_tariff):
        soc, Pb, Pg = current_obs
        print("SOC: {:.2f}, Total Power: {:.2f}, Pb: {:.2f}, Pg: {:.2f}".format(soc, env.total_power, Pb, Pg))

        # Reward for maintaining SOC within the desired range [20, 90]
        reward_soc = 1 if 20 <= soc <= 90 else -1

        # Reward for balancing power exchange based on grid tariff
        if grid_tariff < 0.5:
            reward_power_balance = 1 if Pb > 0 and Pg > 0 else -1
        else:
            reward_power_balance = 1 if Pb > Pg else -1

        # Reward for earning more cost than spending cost
        cost_of_power = env.total_power * env.cost_of_unit

        # Modify reward_cost based on battery state and grid tariff
        if soc >= 90 and Pg > 0:
            reward_cost = cost_of_power - env.grid_payment_rate * Pg  # Sell excess power to the grid
        elif grid_tariff > 0.5 and Pb > Pg:
            reward_cost = cost_of_power - env.grid_payment_rate * (Pb - Pg)  # Reduce cost by using battery more
        else:
            reward_cost = -cost_of_power  # Pay for power if not selling and not using battery efficiently

        reward = reward_soc + reward_power_balance + reward_cost

        # Normalize the reward between the min_reward and max_reward
        normalized_reward = (reward - self.min_reward) / (self.max_reward - self.min_reward)
        normalized_reward = np.clip(normalized_reward, 0, 1)  # Clip the reward between 0 and 1

        # Add the normalized reward to the list
        self.best_reward_per_episode.append(normalized_reward)

        # Choose a random action for now
        action = env.action_space.sample()
        new_obs, _, done, _ = env.step(action)  # Replace the action with the desired action

        if done:
            new_obs = env.reset()  # Reset the environment if the episode has ended

        return new_obs
