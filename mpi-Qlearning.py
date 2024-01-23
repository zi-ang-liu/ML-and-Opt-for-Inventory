import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

'''create a gym environment for multi-period inventory problem'''
class InventoryEnv(gym.Env):
    metadata = {'render.modes': [None]}
    def __init__(self, max_order, max_inventory, min_inventory, mu, h, p):
        self.max_order = max_order
        self.max_inventory = max_inventory
        self.min_inventory = min_inventory
        self.mu = mu
        self.h = h
        self.p = p

        self.action_space = spaces.Discrete(max_order+1)
        self.observation_space = spaces.Discrete(max_inventory-min_inventory+1, start=min_inventory)

    def step(self, action):
        demand = np.random.poisson(self.mu)
        reward = -(self.h * max(0, self.observation + action - demand) + self.p * max(0, demand - (self.observation + action)))
        self.observation = self.observation + action - demand
        if self.observation > self.max_inventory:
            self.observation = self.max_inventory
        elif self.observation < self.min_inventory:
            self.observation = self.min_inventory
        done = False
        info = {}
        return self.observation, reward, False, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.observation = self.observation_space.sample()
        info = {}
        return self.observation, info

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass

def q_learning(alpha, epsilon, gamma, num_episodes, env, n_steps):
    # initialize Q(s,a)
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    # initialize rewards
    total_rewards = np.zeros((num_episodes,))

    # loop for each episode
    for i in range(num_episodes):

        # initialize state
        s, info = env.reset()
        total_reward = 0

        # loop for each step of episode
        for t in range(n_steps):

            # choose action from state using policy derived from Q (e-greedy)
            if np.random.rand() < epsilon:
                a = env.action_space.sample()
            else:
                # choose the action with highest Q(s,a), if multiple, choose randomly
                values_ = Q[s, :]
                a = np.random.choice([action_ for action_, value_ in enumerate(
                    values_) if value_ == np.max(values_)])

            # take action, observe reward and next state
            s_, r, terminated, truncated, info = env.step(a)

            # update Q
            Q[s, a] = Q[s, a] + alpha * \
                (r + gamma * np.max(Q[s_, :]) - Q[s, a])

            # update state
            s = s_

            # update total reward
            total_reward += r

            # until state is terminal
            if terminated:
                total_rewards[i] = total_reward
                break

    return Q, total_rewards

if __name__ == '__main__':
    max_order = 16
    max_inventory = 16
    min_inventory = -16
    mu = 5
    h = 1
    p = 10   

    n_steps = 200
    num_episodes = 20000
    
    env = InventoryEnv(max_order, max_inventory, min_inventory, mu, h, p)
    Q, total_rewards = q_learning(0.1, 0.1, 0.9, num_episodes, env, n_steps)
    
    # print the optimal policy
    policy = np.zeros((env.observation_space.n,))
    for s in range(min_inventory, max_inventory+1):
        policy[s] = np.argmax(Q[s, :])
        print(policy[s])

    # Set the figure size
    plt.figure(figsize=(6, 6))

    # Plot the policy
    for s in range(min_inventory, max_inventory+1):
        plt.plot(s, policy[s], 'k.')

    # Set the labels and title with increased font sizes
    plt.xlabel('State', fontsize=14)
    plt.ylabel('Action', fontsize=14)
    plt.title('Policy', fontsize=16)
    plt.savefig('MPIP_policy_qlearning.pdf')