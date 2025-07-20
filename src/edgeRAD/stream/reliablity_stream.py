import time
from edgeRAD.utils.constants import *
import random
from edgeRAD.stream.patterns import *
from edgeRAD.detection.skip_sample import *
from edgeRAD.utils.mysqlEngine import MysqlEngine
# import pandas as pd

# from collections import Counter

class Reliablity_Stream:
    def __init__(self, mode) -> None:
        self.mode = mode
        self.name = SERVICE_NAME
        self.engine = MysqlEngine()
        self.start_time = int(time.time()*10) #以0.1秒为单位
    
    def init_stream(self):
        if self.mode == "simulation":
            init_reliablity = []
            self.current_time = self.start_time + (RESERVOIR_LEN + FEATURE_LEN)

            pattern_length = random.randint(3,6)*10
            self.pattern = generate_pattern(0.95, pattern_length, 0.95)

            for j in range(self.current_time - (RESERVOIR_LEN + FEATURE_LEN), self.current_time):
                idx = (j - self.start_time) % len(self.pattern)
                value = self.pattern[idx]
                init_reliablity .append(value)

                if idx == len(self.pattern) - 1: 
                    end_state = random.uniform(0.8, 1.0)
                    pattern_length = random.randint(3,6)*10
                    self.pattern = generate_pattern(self.pattern[idx], pattern_length, end_state)
                    self.start_time = self.current_time   
            
            init_reliablity = np.array(init_reliablity, dtype=np.float32)
        
        else:
            print("begin_init_stream...")

            self.current_time = self.start_time
            time_span = RESERVOIR_LEN + FEATURE_LEN + 100
            time.sleep(time_span * 0.1)

            while True:
                self.current_time += time_span
                print("start_end: ", self.start_time, self.current_time)
                init_reliablity = self.read_database(self.start_time, self.current_time)
                print(len(init_reliablity))

                # 如果读到足够的点，就退出循环
                if len(init_reliablity) >= (RESERVOIR_LEN + FEATURE_LEN):
                    break

                # 否则等待一段时间后重新尝试（比如 100 毫秒，避免频繁数据库压力）
                if len(init_reliablity) == 0:
                    time_span = 1
                else:
                    time_span = RESERVOIR_LEN + FEATURE_LEN + 100

                time.sleep(time_span * 0.1)      
            init_reliablity = np.array(init_reliablity, dtype=np.float32)

            print("init_stream: ", init_reliablity)

        return init_reliablity

    def get_next_stream(self, time_span = 1, exe_span = 0):
        if self.mode == "simulation":
            next_reliablity = []
            self.current_time += time_span
            for j in range(self.current_time - time_span, self.current_time):
                idx = (j - self.start_time) % len(self.pattern)
                value = self.pattern[idx]
                next_reliablity.append(value)

                if idx == len(self.pattern) - 1: 
                    end_state = random.uniform(0.8, 1.0)
                    pattern_length = random.randint(3,6)*10
                    self.pattern = generate_pattern(self.pattern[idx], pattern_length, end_state) 
                    self.start_time = self.current_time    
            
            next_reliablity = np.array(next_reliablity, dtype=np.float32)

        else:
            #确定睡眠时间的因子
            factor = 0.1 - exe_span + 0.01
            if factor < 0:
                factor = 0.0

            time.sleep(time_span * factor) 
            self.current_time = self.start_time + time_span
            next_reliablity = self.read_database(self.start_time, self.current_time)
            next_reliablity = np.array(next_reliablity, dtype=np.float32) 
        
        return next_reliablity
    
    def read_database(self, start_time, current_time):
        #将stat_time和current_time转换为毫秒，再查询
        sql_str = f"SELECT end_time, r from service_state where service_name='{self.name}' and end_time < {current_time * 100} and end_time >= {start_time * 100}"
        rl_frame = self.engine.invoke_sql(sql_str)
        
        reliablity = rl_frame["r"].tolist()

        if reliablity:
            self.start_time = (rl_frame["end_time"].max() / 100).astype(int) + 1
        else:
            self.start_time += 1            

        return reliablity

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    env = Reliablity_Stream(EXP_MODE)
    
    Xh = [[0.0] * FEATURE_LEN for _ in range(RESERVOIR_LEN)]
    sampler = StreamingHistoricalSampler(RESERVOIR_LEN, FEATURE_LEN)
    
    r_stream = env.init_stream()
    sampler.r_history.extend(r_stream)
    Xn = sampler.recent()
    weights = [1.0 for _ in range(len(r_stream))]
    Xh = sampler.update(weights, Xh)

    
    trace = r_stream
    # trace = []

    import time
    start = time.time()

    for i in range(0,10000):
        r_stream = env.get_next_stream()
        sampler.r_history.extend(r_stream)
        Xn = sampler.recent()
        weights = [1.0 for _ in range(len(r_stream))]
        Xh = sampler.update(weights, Xh)

        trace = np.concatenate((trace, r_stream), axis=0)

    end = time.time()

    print(f"time: {(end - start)*1000:.2f} ms")

    np.array(trace, dtype=np.float32)  
    plt.plot(np.arange(len(trace)), trace)
    plt.show()

    print(len(trace))

    sampled = trace

    # 把数字转为字符串并以逗号拼接
    parts = []
    for x in sampled:
        # 向下截断到两位小数
        truncated = np.floor(x * 100) / 100
        # 格式化为两位小数
        s = f"{truncated:.3f}"
        # 如果第二位是 0，则去掉最后一个字符，保留一位小数
        if s.endswith("0"):
            s = s[:-1]
        parts.append(s)

    print(len(parts))

    # 用逗号连接并打印
    result = ",".join(parts)
    
    with open("trace.txt", "w", encoding="utf-8") as f:
        f.write(result)

    print("写入完成，已生成 trace.txt")