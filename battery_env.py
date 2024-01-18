import numpy as np
import random
import math
from gym import Env
from gym.spaces import Discrete, Box

class BatteryEnv(Env):
    def __init__(self, power_range=(800, 1200)):
        self.action_space = Discrete(3)
        self.observation_space = Box(low=np.array([10, 50, 1000], dtype=np.float32), high=np.array([90, 450, 250000], dtype=np.float32))

        self.state = random.randint(10, 90)
        self.sim_duration = 3600

        self.Cbat = 230
        self.m = 2
        self.n = 0.05
        self.d = 44
        self.Qfade = 0.01
        self.Yhr = 8760  # Number of hours in a year

        self.dt = 1
        self.ti = 0
        self.tf = 10

        self.power_range = power_range
        self.cost_of_unit = 0.25
        self.grid_payment_rate = 0.2
        self.total_power = 0
        self.total_reward_per_episode = []  # Initialize total_reward_per_episode list
        self.total_reward = 0.0  # Initialize total_reward attribute
        self.Pb = 0  # Initialize Pb attribute
        self.Pg = 0  # Initialize Pg attribute
        
    def step(self, action):
        self.total_power = np.random.uniform(self.power_range[0], self.power_range[1])
        self.Pb = self.calculate_battery_power()
        self.Pg = self.total_power - self.Pb

        if action == 0:  # Charge
            self.state += 1
        elif action == 1:  # Discharge
            self.state -= 1
        self.state = np.clip(self.state, 10, 90)

        reward = self.calculate_reward()
        self.total_reward += reward
        done = self.sim_duration <= 0
        self.sim_duration -= 1

        return [self.state, self.Pb, self.Pg], reward, done, {}

    def reset(self):
        self.state = random.randint(10, 90)
        self.sim_duration = 3600
        self.Pb = 0
        self.Pg = 0
        self.total_reward_per_episode.append(self.total_reward)
        self.total_reward = 0.0
        return [self.state, self.Pb, self.Pg]

    def calculate_battery_power(self):
        soc_values = [self.state] * self.sim_duration

        Cdod_values = []
        Csoc_values = []
        Ctemp_values = []

        for t in range(len(soc_values)):
            SoC = soc_values[t]
            DoD = 90 - SoC

            Cdod = self.Cbat / (2 * self.m * DoD * self.Qfade * (self.n ** 2))
            Cdod_values.append(Cdod)

            total_soc = sum(soc_values[:t + 1])
            SOCAvg = total_soc / (t + 1)
            Csoc = (self.Cbat * self.m * SOCAvg - self.d) / (self.Qfade * self.n * self.Yhr)
            Csoc_values.append(Csoc)

            Pavg = self.total_power - self.Pb
            T = 25  # Default temperature (you might need to adjust this)
            L_T = 1  # Default heat loss (you might need to adjust this)
            dt = 1  # Assuming time steps of 1 hour
            Ctemp = self.Cbat * (sum(dt / (L_T) for _ in range(t + 1)))
            Ctemp_values.append(Ctemp)

        Cdod = max(Cdod_values)
        Csoc = max(Csoc_values)
        Ctemp = max(Ctemp_values)

        Cbd = max(Cdod, Csoc, Ctemp)
        Pb = Cbd / self.cost_of_unit

        return Pb

    def calculate_reward(self):
        reward_soc = 1 if 20 <= self.state <= 90 else -1
        reward_power_balance = 1 if self.Pb > self.Pg else -1
        cost_of_power = self.total_power * self.cost_of_unit
        reward_cost = cost_of_power - self.grid_payment_rate * (self.total_power - self.Pb - self.Pg) if self.total_power >= self.Pb + self.Pg else -cost_of_power
        reward = reward_soc + reward_power_balance + reward_cost
        return reward

class Agent:
    def __init__(self):
        self.best_reward_per_episode = []
        self.min_reward = -10
        self.max_reward = 10

    def step(self, current_obs, env):
        soc, Pb, Pg = current_obs

        reward_soc = 1 if 20 <= soc <= 90 else -1
        reward_power_balance = 1 if Pb > Pg else -1
        cost_of_power = env.total_power * env.cost_of_unit
        reward_cost = cost_of_power - env.grid_payment_rate * (env.total_power - Pb - Pg) if env.total_power >= Pb + Pg else -cost_of_power

        reward = reward_soc + reward_power_balance + reward_cost
        normalized_reward = (reward - self.min_reward) / (self.max_reward - self.min_reward)
        normalized_reward = np.clip(normalized_reward, 0, 1)
        self.best_reward_per_episode.append(normalized_reward)

        action = env.action_space.sample()
        new_obs, _, done, _ = env.step(action)

        if done:
            new_obs = env.reset()

        return new_obs

# Example usage
env = BatteryEnv()
agent = Agent()

obs = env.reset()
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
