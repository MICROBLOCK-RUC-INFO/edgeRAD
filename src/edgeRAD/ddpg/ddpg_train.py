import numpy as np
import torch
import os
import time
from ddpg_agent import DDPGAgent
import random
#from edgeRAD.simulator.service_simulator import *
from edgeRAD.stream.state_simulator import *
import logging
from edgeRAD.utils.constants import *


logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)
log_file_path = f'./log/log_at_{int(time.time())}.log'
if not os.path.exists(log_file_path):
    with open(log_file_path, 'w') as f:
        pass
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)
file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(file_format)
logger.addHandler(file_handler)

# Initialize service simulator
env = Service_Simulator(EXP_MODE, SERVICE_NAME)
print(EXP_MODE)

# Initialize agent
current_path = os.path.dirname(os.path.realpath(__file__))
model = current_path + '/models/'
if not os.path.exists(model):
    os.makedirs(model)

agent = DDPGAgent(STATE_LEN, ACTION_LEN)

try:
    agent.actor.load_state_dict(torch.load(model + 'ddpg_actor.pth'))
    agent.critic.load_state_dict(torch.load(model + 'ddpg_critic.pth'))
    agent.actor_target.load_state_dict(torch.load(model + 'ddpg_actor.pth'))
    agent.critic_target.load_state_dict(torch.load(model + 'ddpg_critic.pth'))
except Exception as e:
    print("No model found. Using random policy")
    pass

# Training Loop
reward_buffer = np.empty(shape=NUM_EPISODE)
data_to_show = {"state": [], "action": [], "recovery_bound": [], "reward": [], "reward_array": [], "next_state": []}

try:
    for episode_i in range(NUM_EPISODE):
    
        is_print = episode_i % 20 == 0
        #is_print = True
        start_time = time.time()
        state = env.reset()
        episode_reward = 0

        for step_i in range(NUM_STEP):
            if EXP_MODE != "simulation":
                print("STEP_i: ", step_i)

            epsilon = np.interp(episode_i * NUM_STEP + step_i,
                              [0, EPSILON_DECAY],
                              [EPSILON_START, EPSILON_END])

            
            if is_print:
                logger.debug("Using actor")
            action = agent.get_action(state)
            #print("action", action)   
            #action = smooth_iter(action)            

            if is_print:
                logger.debug(f"\nstart of episode {episode_i}, step {step_i}\n")
                next_state, reward, done, truncation, info, reward_array = env.step_with_eyes_open(
                    state, action * ACTION_MAX_NUM, logger)
            else:
                next_state, reward, done, truncation, info, reward_array = env.step_with_eyes_open(
                    state, action * ACTION_MAX_NUM)

            agent.replay_buffer.add_memo(state, action, reward, next_state, done)

            if EXP_MODE != "simulation":
                print("next_state: ", next_state)
                print("action: ", action)
            
            if is_print:
                logger.debug(f"State: {state}")
                logger.debug(f"Action: {action * ACTION_MAX_NUM}")
                logger.debug(f"Recovery bound: {RECOVERY_BOUND}")
                logger.debug(f"Reward: {reward}")
                logger.debug(f"Reward array: {reward_array}")
                logger.debug(f"Next state: {next_state}")

            data_to_show["state"].append(state)
            data_to_show["action"].append(action * ACTION_MAX_NUM)
            data_to_show["recovery_bound"].append(RECOVERY_BOUND)
            data_to_show["reward"].append(reward)
            data_to_show["reward_array"].append(reward_array)
            data_to_show["next_state"].append(next_state)

            state = next_state
            episode_reward += reward

            if is_print:
                logger.debug(
                    f"\nend of episode {episode_i}, step {step_i}, episode_reward {episode_reward}, reward {reward}\n"
                )
                agent.update(logger)
            else:
                agent.update()

            if done:
                break

        reward_buffer[episode_i] = episode_reward
        print(
            f"Episode: {episode_i + 1}, Reward: {round(episode_reward, 2)}, using time {time.time()-start_time} s"
        )
        
        # Save models and data every SAVE_INTERVAL_EPISODE episodes
        if (episode_i + 1) % SAVE_INTERVAL_EPISODE == 0:
            save_timestamp = time.strftime("%Y%m%d%H%M%S")
            
            # Save intermediate training data
            np.savez(f"{current_path}/log/sars_to_show_temp.npz",
                     state=np.array(data_to_show["state"]),
                     action=np.array(data_to_show["action"]),
                     recovery_bound=np.array(data_to_show["recovery_bound"]),
                     reward=np.array(data_to_show["reward"]),
                     next_state=np.array(data_to_show["next_state"]))
            
            # Save current best models (overwrite)
            torch.save(agent.actor.state_dict(), model + 'ddpg_actor_temp.pth')
            torch.save(agent.critic.state_dict(), model + 'ddpg_critic_temp.pth')
            
            # Save the intermediate rewards
            np.savetxt(current_path + f'/log/ddpg_reward_temp.txt', reward_buffer[:episode_i+1])

except KeyboardInterrupt as e:
    print("Interrupted by Ctrl+C")
finally:
    timestamp = time.strftime("%Y%m%d%H%M%S")
    
    # Save training data
    np.savez(f"{current_path}/log/sars_to_show_{timestamp}.npz",
             state=np.array(data_to_show["state"]),
             action=np.array(data_to_show["action"]),
             recovery_bound=np.array(data_to_show["recovery_bound"]),
             reward=np.array(data_to_show["reward"]),
             next_state=np.array(data_to_show["next_state"]))
    
    # Save models
    # torch.save(agent.actor.state_dict(), model + f'ddpg_actor_{timestamp}.pth')
    # torch.save(agent.critic.state_dict(), model + f'ddpg_critic_{timestamp}.pth')
            
    torch.save(agent.actor.state_dict(), model + 'ddpg_actor_temp.pth')
    torch.save(agent.critic.state_dict(), model + 'ddpg_critic_temp.pth')
            

    # Save the rewards
    np.savetxt(current_path + f'/log/ddpg_reward_{timestamp}.txt', reward_buffer)