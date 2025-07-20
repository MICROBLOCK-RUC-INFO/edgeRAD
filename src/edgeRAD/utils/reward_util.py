import numpy as np
from edgeRAD.utils.constants import *



def calculate_reward_v3(st, st_1, at, service_names):
    action_len = len(at)  # at 现在是一维向量
    state_len = len(st_1)     # st_1 现在是一维向量
    reward = np.zeros(state_len, dtype=float)
    recovery_bound = RECOVERY_BOUND
    for j in range(action_len):
        reward[j] = reward_at_i_j_v2(st_1, at, j, state_len,
                                    action_len, recovery_bound)
    return reward

def polynomial_func(x):
    return 16.66666559 * np.power(x, 2) + -14.83333181 * x + 3.37857093


def reward_at_i_j_v2(s, a, j, state_len, action_len, recovery_bound):
    s_j = s[state_len - action_len + j]
    a_j = int(a[j])

    reward = -(10 * np.power(a_j, 1.6) +
               1000 * np.exp(-1.0 * polynomial_func(s_j) *
                             (a_j + 1))) / 100

    return reward


if __name__ == "__main__":
    for i in np.arange(0.0, 1.0, 0.01):
        max = -1000
        index = 0
        for j in range(0,100):
            reward = -(10 * np.power(j/10.0, 1.6) +
                  1000 * np.exp(-1.0 * (16.66666559 * np.power(i, 2) + -14.83333181 * i + 3.37857093) * (j/10.0 + 1))) / 100
            if reward > max:
                max = reward  
                index = j 
    
        print(i, index, max)