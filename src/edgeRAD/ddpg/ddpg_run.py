import numpy as np
import torch
import os
import time
from ddpg_agent import DDPGAgent
import logging
from edgeRAD.utils.constants import *
from edgeRAD.utils.reward_util import *
from edgeRAD.stream.reliablity_stream import *
from edgeRAD.detection.divergence import *
from collections import deque
from sklearn.ensemble import IsolationForest

logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)

log_file_path = f'./log/log_run_at_{int(time.time())}.log'
if not os.path.exists(log_file_path):
    with open(log_file_path, 'w') as f:
        pass

file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)
file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(file_format)
logger.addHandler(file_handler)

# Initialize agent
current_path = os.path.dirname(os.path.realpath(__file__))
model = current_path + '/models/'
if not os.path.exists(model):
    os.makedirs(model)

agent = DDPGAgent(STATE_LEN, ACTION_LEN)
model_name = "temp"

def do_action_at(count, loop_num):
    count = int(np.floor(count))


    # 如果为0，就什么都不做
    if count == 0:
        return []

    # 生成分布位置，范围在 [0, total_loops)
    positions = np.linspace(0, loop_num, count + 1, endpoint=False)[1:]
    # 向下取整为循环索引（0-based）
    indices = np.floor(positions).astype(int)
    # print(indices)

    return indices.tolist()

#print(os.path.exists(model + f'ddpg_actor_{model_name}.pth'))
try:
    agent.actor.load_state_dict(
            torch.load(model + f'ddpg_actor_{model_name}.pth',
                       map_location=torch.device('cpu')))
    agent.critic.load_state_dict(
            torch.load(model + f'ddpg_critic_{model_name}.pth',
                       map_location=torch.device('cpu')))
except Exception as e:
    print("No model found.")
    exit(1) 

data_to_show = {"current_state": [], "action": [], "recovery_bound": [], "reward": [], "reward_array": [], "next_state": []}

start_time = int(time.time())
current_time = start_time + STATE_LEN

reliablity = Reliablity_Stream(EXP_MODE)
print(EXP_MODE)
sampler = StreamingHistoricalSampler(RESERVOIR_LEN, FEATURE_LEN)

s_history = []
state = []
Xh = [[0.0]*FEATURE_LEN for _ in range(RESERVOIR_LEN)]
r_batch = reliablity.init_stream()
sampler.r_history.extend(r_batch)

anomaly_num = 0
latency_sum = 0.0
latency_avg = 0.0
window = deque([None]*2, maxlen=2) #state最新两个值

# engine = MysqlEngine()

for i in range(STATE_LEN):
    diff_vals = []
    for j in range(ACTION_MAX_NUM):
        # print('begin_r_batch')
        r_batch = reliablity.get_next_stream()
        # print('r_batch: ', r_batch)
        sampler.r_history.extend(r_batch)
        Xn = sampler.recent()
        cur_weight = [1.0] * len(r_batch)
        Xh = sampler.update(cur_weight, Xh) 

        diff = js_divergence(Xh, Xn) 
        diff = 1 - min(diff, 0.015) / 0.015 * 0.5 
        diff_vals.append(diff)

        window.append((0, i, j, diff)) # 第一个轮数，第二个状态序号，第三个动作序号，第四个状态值
    
    diff_med = np.median(diff_vals)
    state.append(diff_med)

s_history.extend(state)
iso = IsolationForest(contamination=0.01,
                      n_estimators=20,
                      max_samples='auto',
                      random_state=42)
# pre_diff = 0.0
try:
    episode_reward = 0
    last_pos = -1 #上个状态还没有拉取的可靠性数据流长度，默认是-1
    for i in range(NUM_STEP * 10):  
        if EXP_MODE != "simulation":
            print("NUM_SETP: ", i)

        if i % NUM_STEP == 0:
            start_time = time.time()

        action = agent.get_action(state) * ACTION_MAX_NUM 
        # print("state: ", state)
        # print("action: ", action)

        next_state = []
        for j in range(STATE_LEN):

            if action[j] < 1.0:
                hist_min = np.min(sampler.r_history[-100:])
                thresholds = (0.8, 0.7)
                level_map = [0, 1, 2]
                action[j] = level_map[sum(hist_min < t for t in thresholds)]

                if action[j] < 1:
                    r_batch = reliablity.get_next_stream(ACTION_MAX_NUM - last_pos - 1)
                    sampler.r_history.extend(r_batch)
                    last_pos = -1 
                    next_state.append(1.0) #将状态设为1.0，不做检测
                    if EXP_MODE != "simulation":
                        print("state: ", 1.0)
                    
                    continue


            diff_vals = []
            #根据count确定异常检测的时间点和相应间隔
            detections = do_action_at(action[j], ACTION_MAX_NUM)
            detections.insert(0,last_pos)
            time_spans = np.diff(detections)

            exe_time = 0.0 #除了拉取可靠性数据流的执行时间，用于控制在拉取数据前的睡眠时间
            for idx, span in enumerate(time_spans, start=1):
                # print(exe_time)
                r_batch = reliablity.get_next_stream(span, exe_time)
                start = time.time()
                sampler.r_history.extend(r_batch)
                Xn = sampler.recent()

                cur_weight = [1.0] * len(r_batch)
                Xh = sampler.update(cur_weight, Xh)                    
                
                diff = js_divergence(Xh, Xn)      
                diff = 1 - min(diff, 0.015) / 0.015 * 0.5  #从0-0.015映射到1.0-0.5之间 
                diff_vals.append(diff)

                # start = time.time()
                # y_iso = iso.fit_predict(Xn)            # 正常→1，异常→-1
                # iso_anomalies = np.where(y_iso == -1)[0] #定位Xn中异常具体的位置
                # end = time.time()
                # print(f"time: {(end - start)*1000:.2f} ms")
                # print("IsolationForest 异常索引：", iso_anomalies)

                window.append((i, j, detections[idx], diff)) # 

                first, *rest = window
                if first[3] > ANOMALY_THR and all(item[3] < ANOMALY_THR for item in rest):
                    anomaly_num += 1

                    pre, cur = window[-2], window[-1]
                    # print(pre, cur)
                    pre_state_idx, pre_action_idx = pre[1], pre[2]
                    cur_state_idx, cur_action_idx = cur[1], cur[2]
                    # print('state: ', window[2], window[3])
                    gap = cur_state_idx + 1 if pre_state_idx == ACTION_MAX_NUM - 1 and cur_state_idx != ACTION_MAX_NUM - 1 else (cur_state_idx - pre_state_idx)
                    #计算异常发现延迟
                    if gap == 0:
                        latency = cur_action_idx - pre_action_idx
                    else:
                        latency = (gap - 1) * ACTION_MAX_NUM + (cur_action_idx - pre_action_idx) + ACTION_MAX_NUM

                    logger.debug(f"window: {pre}, {cur}")
                    logger.debug(f"lantecy: {latency}, action: {action}")

                    latency_sum += latency 
                    latency_avg = (latency_sum/anomaly_num) * (1/2)

                    if EXP_MODE != "simulation": 
                        print("anomaly_latency: ", anomaly_num, latency_avg)        

                end = time.time()
                exe_time = (end - start) #
                # print(f"time: {exe_time:.3f} s")

            diff_med = np.median(diff_vals)
            next_state.append(diff_med)

            if EXP_MODE != "simulation":
                print("state: ", diff_med)
            
            #调整拉取数据流的起始
            last_pos = detections[-1] - ACTION_MAX_NUM        

        s_history.extend(next_state)
     

        if EXP_MODE != "simulation":
            print("next_state: ", next_state)
            print("action: ", action)
            
        reward = calculate_reward_v3(state, next_state, action, [SERVICE_NAME])

        data_to_show["current_state"].append(state)
        data_to_show["action"].append(action)
        data_to_show["reward"].append(reward.sum())
        data_to_show["reward_array"].append(reward)
        data_to_show["next_state"].append(next_state)

        state = next_state   

        episode_reward += reward.sum()
        if (i+1) % NUM_STEP == 0:
            logger.debug(f"Episode: {int((i+1)/NUM_STEP)}, Reward: {round(episode_reward, 2)}, Anomlay: {anomaly_num}, Latency: {latency_avg}, using time {time.time()-start_time} s")

            print(
               f"Episode: {int((i+1)/NUM_STEP)}, Reward: {round(episode_reward, 2)}, Anomlay: {anomaly_num}, Latency: {latency_avg}, using time {time.time()-start_time} s"
            )
            episode_reward = 0     

    # Save training data
    np.savez(f"{current_path}/log/sars_run_to_show_{model_name}.npz",
                state=np.array(data_to_show["current_state"]),
                action=np.array(data_to_show["action"]),
                recovery_bound=np.array(data_to_show["recovery_bound"]),
                reward=np.array(data_to_show["reward"]),
                next_state=np.array(data_to_show["next_state"]))   

except KeyboardInterrupt as e:
    print("Interrupted by Ctrl+C")

    np.savez(f"{current_path}/log/sars_run_to_show_{model_name}.npz",
                state=np.array(data_to_show["current_state"]),
                action=np.array(data_to_show["action"]),
                recovery_bound=np.array(data_to_show["recovery_bound"]),
                reward=np.array(data_to_show["reward"]),
                next_state=np.array(data_to_show["next_state"]))


# # 1. 第一条线：0, 1, 2, …, N-1
# x_trace = np.arange(len(sampler.r_history))

# # 2. 第二条线：也从 0 到 N-1，但分成 M 段
# x_state = np.linspace(
#     0,
#     len(sampler.r_history) - 1,  # 上限要用 r_history 的长度
#     len(s_history)                       # 采样点数保持和 score 一致
# )

# import matplotlib.pyplot as plt
# # 创建图和第一个 y 轴
# fig, ax1 = plt.subplots(figsize=(8, 4))

# # 绘制 trace 曲线（左 y 轴）
# color1 = 'tab:blue'
# ax1.set_ylabel('trace', color=color1)
# ax1.plot(x_trace, sampler.r_history, color=color1, label='trace')
# ax1.tick_params(axis='y', labelcolor=color1)
# ax1.set_ylim(0.4, 1.05)

# # 创建第二个 y 轴，共享 x 轴
# ax2 = ax1.twinx()
# color2 = 'tab:red'
# ax2.set_ylabel('state', color=color2)
# ax2.plot(x_state, s_history, color=color2,label='state', alpha=0.6)
# ax2.tick_params(axis='y', labelcolor=color2)
# ax2.set_ylim(0.4, 1.05)

# # 设置图标题和图例
# plt.title('trace and state')
# fig.tight_layout()
# plt.show()