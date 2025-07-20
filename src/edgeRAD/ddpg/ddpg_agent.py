

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import torch.nn.functional as F
import time
from edgeRAD.utils.constants import *

# min_value = -22
# max_value = 15


def batch_min_max_normalize(tensor):
    # 计算批次级别的最小值和最大值
    # 执行 Min-Max 归一化
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor


def standardize_input(data):
    # 计算均值和标准差
    mean = torch.mean(data)
    std = torch.std(data)
    if std == 0:
        return data
    # 标准化数据
    standardized_data = (data - mean) / std

    return standardized_data


# Actor Network   
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=HIDDEN_DIM):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        self.fc5 = nn.Linear(hidden_dim, 1024)
        self.fc6 = nn.Linear(1024, hidden_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = self.fc4(x)
        x = torch.tanh(x)
        action = (x + 1) /2
        return action    


    
# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=HIDDEN_DIM):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, 1024)
        self.fc6 = nn.Linear(1024, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        # x = standardize_input(x)
        x = torch.cat([x, a], 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        return self.fc4(x)


# Replay Buffer
class ReplayMemory:
    def __init__(self, capacity):  # 构造函数
        self.buffer = deque(maxlen=capacity)  # deque是双向队列，可以从两端append和pop

    def add_memo(self, state, action, reward, next_state, done):
        state = np.expand_dims(state,
                               0)  # np.expand_dims()是为了增加一个维度，从(3,)变成(1,3)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))  # 从右端插入
        # print(self.buffer)

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size))
        # * is to unpack the list; zip is to combine the elements of the tuple
        return np.concatenate(state), action, reward, np.concatenate(
            next_state), done
        '''
        对state使用 np.concatenate()函数是因为state是一个list，里面的元素是ndarray，所以要把它们拼接起来
        '''

    def __len__(self):  # a special method to get the length of the buffer
        return len(self.buffer)

    def service_info(self, name, tensor, all_service):
        result = f"{name}:\n"
        result += f"  Shape: {tensor.shape}\n"
        rows = tensor.tolist()
        for i in range(len(rows)):
            result += f"  {all_service[i]}: {rows[i]}\n"
        return result

    def tensor_info(self, name, tensor):
        result = f"{name}:\n"
        if tensor is None:
            result += "  Values: None\n"
        if tensor.dim() == 1:
            result += f"  Shape: {tensor.shape}\n"
            result += f"  Values: {tensor.tolist()}\n"
        elif tensor.dim() == 2:
            result += f"  Shape: {tensor.shape}\n"
            for row in tensor.tolist():
                result += f"  Row: {row}\n"
        else:
            result += f"  Unknown tensor dimension: {tensor.dim()}\n"
        return result

    def array_info(self, name, array):
        result = f"{name}:\n"
        if array is None:
            result += "  Values: None\n"
        else:
            result += f"  Shape: {array.shape}\n"
            if array.ndim == 1:
                result += f"  Values: {array.tolist()}\n"
            elif array.ndim == 2:
                for row in array.tolist():
                    result += f"  Row: {row}\n"
            else:
                result += f"  Unknown array dimension: {array.ndim}\n"
        return result

    def number_info(self, name, number):
        result = f"{name}:\n"
        if number is None:
            result += "  Values: None\n"
        else:
            result += f"  Values: {number}\n"
        return result


# DDPG Agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim,
                           action_dim).to(device)  # move nn to device
        self.actor_target = Actor(state_dim, action_dim).to(
            device)  # same structure as actor
        self.actor_target.load_state_dict(
            self.actor.state_dict())  # copy the current nn's weights of actor
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=LR_ACTOR)  # retrieves the parameters

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(),
                                           lr=LR_CRITIC)

        self.replay_buffer = ReplayMemory(MEMORY_SIZE)

        self.data_to_show = {
            "reward": [],
            "next_q_value": [],
            "q_value": [],
            "critic_loss": [],
            "actor_loss": []
        }

    def get_action(self, state):
        # 神经网络需要按批次输入，所以需要将向量转化为1*dim的形式输入
        state = torch.FloatTensor(state).unsqueeze(0).to(
            device)  # unsqueeze(0) add a dimension from (3,) to (1,3)
        action = self.actor(state)
        return action.detach().cpu().numpy(
        )[0]  # detach the tensor from the current graph and convert it to numpy
        '''
        .cpu() is a method that moves a tensor from GPU memory to CPU memory. 
        This is useful if you want to perform operations on the tensor using NumPy on the CPU.
        '''

    def update(self, logger=None):
        if len(self.replay_buffer) < batch_size:
            return  # skip the update if the replay buffer is not filled enough

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            batch_size)
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(np.vstack(actions)).to(device)
        # actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # Critic update
        next_actions = self.actor_target(next_states)
        target_Q = self.critic_target(next_states, next_actions.detach(
        ))  # .detach() means the gradient won't be backpropagated to the actor
        self.data_to_show["reward"].append(
            rewards.detach().cpu().numpy().reshape(-1))
        self.data_to_show["next_q_value"].append(
            target_Q.detach().cpu().numpy().reshape(-1))
        target_Q = rewards + (GAMMA * target_Q * (1 - dones))
        current_Q = self.critic(states, actions)
        self.data_to_show["q_value"].append(
            current_Q.detach().cpu().numpy().reshape(-1))
        critic_loss = nn.MSELoss()(
            current_Q,
            target_Q.detach())  # nn.MSELoss() means Mean Squared Error
        self.data_to_show["critic_loss"].append(critic_loss.item())
        self.critic_optimizer.zero_grad(
        )  # .zero_grad() clears old gradients from the last step
        critic_loss.backward(
        )  # .backward() computes the derivative of the loss
        self.critic_optimizer.step()  # .step() is to update the parameters

        # Actor update
        actor_loss = -self.critic(states, self.actor(
            states)).mean()  # .mean() is to calculate the mean of the tensor
        self.data_to_show["actor_loss"].append(actor_loss.item())
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        if not logger is None:
            logger.debug(f"critic loss: {critic_loss.item()}")
            logger.debug(f"actor loss: {actor_loss.item()}")
        # Update target networks
        for target_param, param in zip(self.actor_target.parameters(),
                                       self.actor.parameters()):
            target_param.data.copy_(TAU * param.data +
                                    (1.0 - TAU) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(),
                                       self.critic.parameters()):
            target_param.data.copy_(TAU * param.data +
                                    (1.0 - TAU) * target_param.data)