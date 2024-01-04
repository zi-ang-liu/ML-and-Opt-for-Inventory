import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson

def value_iteration(dynamics, state_space, action_space, value, policy, theta=1e-4, gamma=0.9):
    # initialize value
    delta = np.inf
    k = 0
    while delta >= theta:
        k = k + 1
        value_old = value.copy()
        for state in state_space:
            # Update V[s].
            value[state] = max([sum([prob * (reward + gamma * value_old[next_state]) for (
                next_state, reward), prob in dynamics[state, action].items()]) for action in action_space])
            # print('State {}, value = {}'.format(state, value[state]))
        delta = np.max(np.abs(value - value_old))
        print('Iteration {}, delta = {}'.format(k, delta))

    for state in state_space:
        q_max_value = -np.inf
        for action in action_space:
            q_value_temp = sum([prob * (reward + gamma * value[next_state])
                             for (next_state, reward), prob in dynamics[state, action].items()])
            if q_value_temp > q_max_value:
                q_max_value = q_value_temp
                policy[state] = action
    return value, policy

def build_MPIP_dynamics(h, p, mu):
    """
    Build the dynamics of the Markov Process for the 
    Multi-Period Inventory Problem

    h: holding cost
    p: penalty cost
    mu: mean demand
    """

    EPSILON = 1e-4

    # compute the maximum demand
    for i in range(10*mu):
        if poisson.pmf(i, mu) < EPSILON:
            max_demand = i
            break

    min_state = -max_demand
    max_state = max_demand
    
    state_space = list(range(min_state, max_state+1))
    action_space = list(range(0, max_demand+1))
    
    # build probability dictionary
    prob_dict = {}
    for d in range(max_demand+1):
        prob_dict[d] = poisson.pmf(d, mu)

    # # build dynamics
    dynamics = {}
    for state in state_space:
        for action in action_space:
            dynamics[state, action] = {}
            for demand in range(max_demand+1):
                next_state = state + action - demand
                reward = -(h * max(0, state + action - demand) + p * max(0, demand - (state + action)))
                dynamics[state, action][next_state, reward] = prob_dict[demand]
    
    # build value and policy
    init_value = np.zeros(len(state_space))
    init_policy = np.zeros(len(state_space))

    return dynamics, state_space, action_space, init_value, init_policy


if __name__ == '__main__':

    h = 1 
    p = 10 
    mu = 5

    dynamics, state_space, action_space, init_value, init_policy = build_MPIP_dynamics(h, p, mu)
    value, policy = value_iteration(dynamics, state_space, action_space, init_value, init_policy)

    # plot policy
    # Set the figure size
    plt.figure(figsize=(6, 6))

    # Plot the policy
    for s in state_space:
        plt.plot(s, policy[s], 'k.')

    # Set the labels and title with increased font sizes
    plt.xlabel('State', fontsize=14)
    plt.ylabel('Action', fontsize=14)
    plt.title('Policy', fontsize=16)
    plt.show()
    # plt.savefig('MPIP_policy.pdf')