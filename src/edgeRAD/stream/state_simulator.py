import time
import numpy as np
from edgeRAD.utils.reward_util import *
from edgeRAD.utils.constants import *
from edgeRAD.stream.patterns import *
from edgeRAD.stream.reliablity_stream import *
from edgeRAD.detection.divergence import *
from sklearn.ensemble import IsolationForest

class Service_Simulator:
    def __init__(self, mode, service) -> None:
        self.mode = mode
        self.engine = MysqlEngine()
        self.name = service
        self.reliablity = Reliablity_Stream(EXP_MODE)
        self.sampler = StreamingHistoricalSampler(RESERVOIR_LEN, FEATURE_LEN)
        self.Xh = [[0.0]*FEATURE_LEN for _ in range(RESERVOIR_LEN)]
        self.last_pos = -1  #上次调用还没有拉取的可靠性数据流长度，初始是-1

        self.iso = IsolationForest(contamination=0.01,
                      n_estimators=20,
                      max_samples='auto',
                      random_state=42)

    def reset(self):
        return self.init_state()

    def init_state(self):
        state = []
        r_batch = self.reliablity.init_stream()
        self.sampler.r_history.extend(r_batch)

        for _ in range(STATE_LEN):
            r_batch = self.reliablity.get_next_stream()
            self.sampler.r_history.extend(r_batch)
            Xn = self.sampler.recent()
            cur_weight = [1.0] * len(r_batch)
            self.Xh = self.sampler.update(cur_weight, self.Xh)                    
 
            diff = js_divergence(self.Xh, Xn) 
            diff = 1 - min(diff, 0.015) / 0.015 * 0.5 

            state.append(diff)
        return np.array(state, dtype=np.float32)
    
    
    def step_with_eyes_open(self, state, action, logger=None):

        next_state = []
        for j in range(STATE_LEN):

            diff_vals = []
            #根据count确定异常检测的时间点和相应间隔
            detections = self.do_action_at(action[j], ACTION_MAX_NUM)
            detections.insert(0, self.last_pos)
            time_spans = np.diff(detections)
            # print(detections)
            # print(time_spans)
            for span in time_spans:
                r_batch = self.reliablity.get_next_stream(span)
                self.sampler.r_history.extend(r_batch)

                Xn = self.sampler.recent()
                cur_weight = [1.0] * len(r_batch)
                Xh = self.sampler.update(cur_weight,self.Xh)                    
                
                diff = js_divergence(Xh, Xn)      
                diff = 1 - min(diff, 0.015) / 0.015 * 0.5  #从0-0.015映射到1.0-0.5之间 
                diff_vals.append(diff)

            diff_med = np.median(diff_vals)
            next_state.append(diff_med)

            # start = time.time()
            # y_iso = self.iso.fit_predict(Xn)            # 正常→1，异常→-1
            # iso_anomalies = np.where(y_iso == -1)[0] #定位Xn中异常具体的位置
            # end = time.time()
            # print(f"time: {(end - start)*1000:.2f} ms")
            # print("IsolationForest 异常索引：", iso_anomalies)
            
            #调整拉取数据流的起始
            self.last_pos = detections[-1] - ACTION_MAX_NUM 

        reward = calculate_reward_v3(state, next_state, action, [self.name])
        next_state = np.array(next_state, dtype=np.float32)
        
        return next_state, reward.sum(), False, None, None, reward

    def do_action_at(self, count, loop_num):
        count = int(np.floor(count))

        if count == 0:
            count = 1

        #如果为0，就什么都不做
        # if count == 0:
        #     return []

        # 生成分布位置，范围在 [0, total_loops)
        positions = np.linspace(0, loop_num, count + 1, endpoint=False)[1:]
        # 向下取整为循环索引（0-based）
        indices = np.floor(positions).astype(int)
        # print(indices)

        return indices.tolist()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    env = Service_Simulator(EXP_MODE, SERVICE_NAME)
    state = env.reset()
    score = []
    score.extend(state)
    #print(state)
    start = time.time()
    for i in range(100):
        action = np.ones(10, dtype=np.float64)
        next_state, reward, done, truncation, info, reward_array = env.step_with_eyes_open(state, action)

        state = next_state
        score.extend(state)
    
    end = time.time()
    print(f"time: {(end - start)*1000:.2f} ms")

    env.sampler.r_history = env.sampler.r_history[RESERVOIR_LEN + FEATURE_LEN + STATE_LEN:]
    score = score[STATE_LEN:]

    print(len(score),len(env.sampler.r_history))
    #print(score)

    # 1. 第一条线：0, 1, 2, …, N-1
    x_trace = np.arange(len(env.sampler.r_history))

    # 2. 第二条线：也从 0 到 N-1，但分成 M 段
    x_state = np.linspace(
        0,
        len(env.sampler.r_history) - 1,  # 上限要用 r_history 的长度
        len(score)                       # 采样点数保持和 score 一致
    )

    # 创建图和第一个 y 轴
    fig, ax1 = plt.subplots(figsize=(8, 4))

    # 绘制 trace 曲线（左 y 轴）
    color1 = 'tab:blue'
    ax1.set_ylabel('trace', color=color1)
    ax1.plot(x_trace, env.sampler.r_history, color=color1, label='trace')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0.4, 1.05)

    # 创建第二个 y 轴，共享 x 轴
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('state', color=color2)
    ax2.plot(x_state, score, color=color2,label='state', alpha=0.6)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0.4, 1.05)

    # 设置图标题和图例
    plt.title('trace and state')
    fig.tight_layout()
    plt.show()