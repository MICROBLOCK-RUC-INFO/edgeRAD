import numpy as np
#from scipy.linalg import svd
from edgeRAD.detection.skip_sample import StreamingHistoricalSampler
from edgeRAD.stream.reliablity_stream import *
from collections import Counter
import random

#计算 Jensen-Shannon 散度
def js_divergence(P: np.ndarray, Q: np.ndarray, eps: float = 1e-12) -> float:
    P = np.clip(P, eps, 1)
    Q = np.clip(Q, eps, 1)
    P /= P.sum()
    Q /= Q.sum()
    M = 0.5 * (P + Q)
    return 0.5 * (np.sum(P * np.log(P / M)) + np.sum(Q * np.log(Q / M)))


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    sampler = StreamingHistoricalSampler(RESERVOIR_LEN, FEATURE_LEN)
    Xh = [[0.0]*FEATURE_LEN for _ in range(RESERVOIR_LEN)]  # 预先创建

    env = Reliablity_Stream(EXP_MODE)
    r_batch = env.init_stream()
    sampler.r_history.extend(r_batch)
    Xn = sampler.recent()
    w_batch = [1.0] * len(r_batch)
    Xh = sampler.update(w_batch, Xh)

    state = []

    trace = []

    import time
    start = time.time()
    for i in range(0,10000):
        r_batch = env.get_next_stream()
        sampler.r_history.extend(r_batch)
        Xn = sampler.recent()
        w_batch = [1.0] * len(r_batch)
        Xh = sampler.update(w_batch, Xh)

        
        diff = js_divergence(Xh, Xn)
        diff = 1 - min(diff, 0.015) / 0.015 * 0.5 
        state.append(diff)
        

        trace = np.concatenate((trace, r_batch), axis=0)

    sampler.fill_feature(Xh)

    end = time.time()
    print(f"time: {(end - start)*1000:.2f} ms")


    x_trace = np.arange(len(trace))   # 0,1,...,99
    x_state = np.linspace(0, len(trace)-1, len(state))  # 映射到 [0, 99]

    # 创建图和第一个 y 轴
    fig, ax1 = plt.subplots(figsize=(8, 4))

    # 绘制 trace 曲线（左 y 轴）
    color1 = 'tab:blue'
    ax1.set_ylabel('trace', color=color1)
    ax1.plot(x_trace, trace, color=color1, label='trace (len=100)')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0.4, 1.05)

    # 创建第二个 y 轴，共享 x 轴
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('state', color=color2)
    ax2.plot(x_state, state, color=color2, label='state (len=1000)', alpha=0.6)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0.4, 1.05)

    # 设置图标题和图例
    plt.title('trace and state')
    fig.tight_layout()
    plt.show()

    indexes = sampler.indexes  # List[Optional[int]]

    # 统计频数
    counter = Counter(indexes)

    # 可视化
    values = list(counter.keys())
    counts = list(counter.values())

    plt.bar(values, counts, width=2, color='skyblue')
    plt.xlabel("Index Value")
    plt.ylabel("Frequency")
    plt.title("Distribution of updater._indexes")
    plt.grid(True)
    plt.show()

    

