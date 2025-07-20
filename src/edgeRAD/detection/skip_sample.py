import random
import numpy as np
from typing import List, Optional
from edgeRAD.detection.trunc_binomial import FastTruncatedBinomialSampler  
from edgeRAD.stream.reliablity_stream import *
from collections import Counter

class StreamingHistoricalSampler:
    def __init__(self, V: int, D: int):
        self.rand = random.Random()
        self.V = V
        self.D = D
        # 截断二项式采样
        self.sampler = FastTruncatedBinomialSampler(self.V)
        # 全局累积权重和跳过阈值
        self.W = 0.0
        self.theta_skip = float('inf')
        # 存储所有历史 r 值
        self.r_history: List[float] = []
        # 全局位置计数（索引增量）
        self.global_pos = 0
        # 采样点数量
        self.k_num = 0

        # 每次采样的全局位置索引
        self.indexes: List[Optional[int]] = [None] * self.V
        #self.recent_indexes: List[Optional[int]] = [None] * self.V

    def update(
        self,
        weight_vals: List[float],
        Xh: List[List[float]]
    ) -> List[List[float]]:


        for i in range(self.global_pos, len(self.r_history) - self.D + 1):
            # 更新全局权重与阈值
            position = i - self.global_pos
            if self.theta_skip == float('inf'):
                p = self.rand.random()
                self.theta_skip = self.W / ((1.0 - p) ** (1.0 / self.V)) if self.V > 0 else float('inf')
            
            w_val = weight_vals[position]
            self.W += w_val
            # 达到采样条件
            if self.W > self.theta_skip:
                # 记录本次采样全局位置
                self.k_num += 1
                for idx in self.sampler.sample_positions(w_val / self.W):
                    self.indexes[idx] = self.global_pos

                # 重置阈值
                self.theta_skip = float('inf')
            
            # 全局位置递增
            self.global_pos += 1

        return Xh
    
    def fill_feature(
        self,
        Xh: List[List[float]]
    ) -> List[List[float]]:
        for i, start in enumerate(self.indexes):
            if start is not None and start + self.D <= len(self.r_history):
                for d in range(self.D):
                    Xh[i][d] = self.r_history[start + d] 
            else:
                raise IndexError(f"Index out of range for channel {i}: start={start}, len(r_history)={len(self.r_history)}, D={self.D}")   



    def recent(self
    ) -> List[List[float]]:
        Xn = []
        if len(self.r_history) >= self.V + self.D - 1:
            start_pos = len(self.r_history) - self.V - self.D + 1  # 起始位置
            for i in range(self.V):
                row = self.r_history[start_pos + i : start_pos + i + self.D]
                Xn.append(row)
                #self.recent_indexes[i] = start_pos + i
        return Xn
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    sampler = StreamingHistoricalSampler(V=50, D=50)
    Xh = [[0.0]*50 for _ in range(50)]  # 预先创建

    trace = []

    import time
    start = time.time()
    for i in range(0,100):
        r_batch = [random.random() for _ in range(10)]
        sampler.r_history.extend(r_batch)
        w_batch = [1.0] * len(r_batch)
        Xn = sampler.recent()
        Xh = sampler.update(w_batch, Xh)

        trace = np.concatenate((trace, r_batch), axis=0)


    end = time.time()
    print(f"time: {(end - start)*1000:.2f} ms")

    #采样点记录
    indexes = sampler.indexes 

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

