import numpy as np
import os
from ddpg_agent import DDPGAgent
from edgeRAD.utils.reward_util import *
from edgeRAD.utils.constants import *


def search_optimal_action(next_state):
    action = np.zeros(next_state.shape)
    optimal_reward = np.zeros(next_state.shape)
    for i in range(len(next_state)):
        s_i = next_state[i]
        max_reward = -10000
        for j in range(ACTION_LEN):
            reward = -(10 * np.power(j, 1.6) +
                       1000 * np.exp(-1.0 * polynomial_func(s_i) *
                                     (j + 1))) / 100
            if reward > max_reward:
                max_reward = reward
                action[i] = j
        optimal_reward[i] = max_reward
    return action, optimal_reward



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    current_path = os.path.dirname(os.path.realpath(__file__))
    model = current_path + '//models//'
    
    agent = DDPGAgent(STATE_LEN, ACTION_LEN)
    model_name = "temp"
    data_name = "temp"
    

    run_loaded_arrays = np.load(f'{current_path}/log/sars_to_show_{data_name}.npz')
    state = run_loaded_arrays['state'].reshape(-1)

    # action 是一个time_step * (action_len) 的数组
    action = np.floor(run_loaded_arrays["action"].reshape(-1, ACTION_LEN).reshape(-1)).astype(int)
    #action = np.round(run_loaded_arrays["action"].reshape(-1, ACTION_LEN).reshape(-1), 1)

    recovery_bounds = run_loaded_arrays["recovery_bound"]

    
    print(action.sum() / (action.size * ACTION_MAX_NUM))

    
    reward = run_loaded_arrays['reward']
    next_state = run_loaded_arrays['next_state'].reshape(-1)

    num_step = NUM_STEP
    sum_slice_length = 200
    x = np.arange(sum_slice_length)
    rewards_slice = []
    rewards_optimal_slice = []
    q_slice = []
    q_1_slice = []
    start_episode = 60
    start_i = 0
    slice_length = 2000

    fig, ax1 = plt.subplots()

    ax1.plot(np.arange(slice_length),
             next_state[start_episode * NUM_STEP * STATE_LEN + start_i:
                       start_episode * NUM_STEP * STATE_LEN + start_i + slice_length],
             label='state',
             color='b')

    ax1.set_xlabel('Time')
    ax1.set_ylim([0, 1.1])
    ax1.set_ylabel('State', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    unique_values, counts = np.unique(
        action[start_episode * NUM_STEP * STATE_LEN + start_i:
                      start_episode * NUM_STEP * STATE_LEN + start_i + slice_length],
        return_counts=True)
    count_dict = dict(zip(unique_values, counts))
    print(count_dict)
    

    ax2 = ax1.twinx()
    ax2.plot(np.arange(slice_length),
             action[start_episode * NUM_STEP * STATE_LEN + start_i:
                          start_episode * NUM_STEP * STATE_LEN + start_i + slice_length],
             label='action',
             color='r')    

    ax2.plot(np.arange(slice_length),
             search_optimal_action(
                 next_state[start_episode * NUM_STEP * STATE_LEN + start_i:
                           start_episode * NUM_STEP * STATE_LEN + start_i + slice_length])[0],
             label='action_optimal',
             color='g')

    ax2.set_ylim([-1,15])
    ax2.set_ylabel('Action', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    fig.tight_layout()
    ax1.legend(loc="upper left")
    ax2.legend(loc="right")
    plt.show()

    action_optimal, reward_optimal = search_optimal_action(
        next_state[start_episode * NUM_STEP * STATE_LEN:(start_episode + 2) * NUM_STEP * STATE_LEN])
    reward_optimal = reward_optimal.reshape(-1, STATE_LEN).sum(axis=1)


    plt.plot(x, reward_optimal[0:sum_slice_length], label=f"optimal reward")
    plt.plot(x,
             reward[start_episode * NUM_STEP:start_episode * NUM_STEP + sum_slice_length],
             label="reward")
    plt.legend(loc="best")
    plt.ylim([-60, 0])
    plt.show()

 