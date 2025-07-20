import random
#from matplotlib import pyplot as plt
import numpy as np
from edgeRAD.utils.constants import *
import time

pattern_min_r = RECOVERY_BOUND - 0.1

def initialize_pattern(length):
    if length < 1:
        raise ValueError("length should be at least 1")

    return [1.0 for _ in range(length)]


def generate_pattern(r_start, pattern_length, r_end, logger=None):
    choice = random.randint(0, 20)
    if choice == 0:
        if not logger is None:
            logger.debug("recovery pattern type: U_shape")
        pattern = UShapePattern().generate_pattern(r_start, pattern_length,
                                                   r_end)
    elif choice == 1:
        if not logger is None:
            logger.debug("recovery pattern type: w_shape")
        pattern = WShapePattern().generate_pattern(r_start, pattern_length,
                                                   r_end)
    elif choice == 2:
        if not logger is None:
            logger.debug("recovery pattern type: v_shape")
        pattern = VShapePattern().generate_pattern(r_start, pattern_length,
                                                   r_end)
    elif choice == 3:
        pattern = SuddenPattern().generate_pattern(r_start, pattern_length,
                                                   r_end)
    else:
        pattern = HorizonPattern().generate_pattern(r_start, pattern_length,
                                                   r_end)
    return pattern


from abc import ABC, abstractmethod


class PatternGenerator(ABC):
    @abstractmethod
    def generate_pattern(self, r_start, pattern_length, r_end):
        pass

class SuddenPattern(PatternGenerator):
    def generate_pattern(self, r_start, pattern_length, r_end):
        pattern = [random.uniform(0.5,0.6)] * (pattern_length)
        #pattern = [random.uniform(0.58, 0.6) for _ in range(pattern_length)]
        return pattern


class HorizonPattern(PatternGenerator):
    def generate_pattern(self, r_start, pattern_length, r_end):
        pattern = [1.0] * pattern_length
        #pattern = [random.uniform(0.995, 1.0) for _ in range(pattern_length)]
        return pattern

class UShapePattern(PatternGenerator):
    def generate_pattern(self, r_start, pattern_length, r_end):
        if pattern_length < 3:
            raise ValueError("pattern_length should be at least 3")

        # 随机扰动参数
        perturb_r_min = random.uniform(0, 0.4)

        # 分配三个阶段的随机长度比例
        proportions = [random.uniform(0.4, 0.5), random.uniform(0.2, 0.35)]
        proportions.append(1.0 - sum(proportions))

        # 计算各阶段的长度
        decline_length = int(pattern_length * proportions[0])
        min_period_length = int(pattern_length * proportions[1])
        recovery_length = pattern_length - decline_length - min_period_length

        # 确保总长度为pattern_length
        total_length = decline_length + min_period_length + recovery_length
        if total_length != pattern_length:
            recovery_length += pattern_length - total_length

        # 设置最低点的可靠性值
        #r_min = pattern_min_r + perturb_r_min
        r_min = random.uniform(0.9,0.7)
        #r_min = 0.2
        
        #print(pattern_min_r,perturb_r_min)

        # 初始化 pattern 列表
        pattern = [1.0] * pattern_length

        # 下降期
        for i in range(decline_length):
            perturb_value = 0
            # perturb_value = random.uniform(-0.02, 0.02)
            pattern[i] = (r_start - (r_start - r_min) *
                          (i / decline_length)) + perturb_value
            pattern[i] = min(max(pattern[i], 0.0), 1.0)

        # 低稳定期
        for i in range(decline_length, decline_length + min_period_length):
            perturb_value = 0
            # perturb_value = random.uniform(-0.02, 0.02)
            pattern[i] = r_min + perturb_value
            pattern[i] = min(max(pattern[i], 0.0), 1.0)

        # 平滑过渡到恢复期
        for i in range(decline_length + min_period_length, pattern_length):
            perturb_value = 0
            # perturb_value = random.uniform(-0.02, 0.02)
            pattern[i] = (r_min + (r_end - r_min) * (
                (i - decline_length - min_period_length) / recovery_length)
                          ) + perturb_value
            pattern[i] = min(max(pattern[i], 0.0), 1.0)

        return pattern


class WShapePattern(PatternGenerator):
    def generate_pattern(self, r_start, pattern_length, r_end):
        if pattern_length < 4:
            raise ValueError("pattern_length should be at least 4")

        # 随机扰动参数
        perturb_r_min = random.uniform(0, 0.4)
        perturb_middle = random.uniform(0.1, 0.15)

        # 分配四个阶段的随机长度比例
        proportions = [
            random.uniform(0.4, 0.5),
            random.uniform(0.2, 0.25),
            random.uniform(0.15, 0.18)
        ]
        proportions.append(1.0 - sum(proportions))

        # 计算各阶段的长度
        decline_length1 = int(pattern_length * proportions[0])
        recovery_length1 = int(pattern_length * proportions[1])
        decline_length2 = int(pattern_length * proportions[2])
        recovery_length2 = pattern_length - decline_length1 - recovery_length1 - decline_length2

        # 确保总长度为 pattern_length
        total_length = decline_length1 + recovery_length1 + decline_length2 + recovery_length2
        if total_length != pattern_length:
            recovery_length2 += pattern_length - total_length

        # 设置最低点的可靠性值
        #r_min = pattern_min_r + perturb_r_min
        r_min = random.uniform(0.9,0.7)
        r_middle = r_min + perturb_middle

        # 初始化 pattern 列表
        pattern = [1.0] * pattern_length

        # 第一个下降期
        for i in range(decline_length1):
            perturb_value = 0
            # perturb_value = random.uniform(-0.02, 0.02)
            pattern[i] = (r_start - (r_start - r_min) *
                          (i / decline_length1)) + perturb_value
            pattern[i] = min(max(pattern[i], 0.0), 1.0)

        # 第一个恢复期
        for i in range(decline_length1, decline_length1 + recovery_length1):
            perturb_value = 0
            # perturb_value = random.uniform(-0.02, 0.02)
            pattern[i] = (r_min + (r_middle - r_min) * (
                (i - decline_length1) / recovery_length1)) + perturb_value
            pattern[i] = min(max(pattern[i], 0.0), 1.0)
        # 第二个下降期
        for i in range(decline_length1 + recovery_length1,
                       decline_length1 + recovery_length1 + decline_length2):
            perturb_value = 0
            # perturb_value = random.uniform(-0.02, 0.02)
            pattern[i] = (r_middle - (r_middle - r_min) * (
                (i - decline_length1 - recovery_length1) / decline_length2)
                          ) + perturb_value
            pattern[i] = min(max(pattern[i], 0.0), 1.0)

        # 第二个恢复期
        for i in range(decline_length1 + recovery_length1 + decline_length2,
                       pattern_length):
            perturb_value = 0
            # perturb_value = random.uniform(-0.02, 0.02)
            pattern[i] = (r_min + (r_end - r_min) * (
                (i - decline_length1 - recovery_length1 - decline_length2) /
                recovery_length2)) + perturb_value
            pattern[i] = min(max(pattern[i], 0.0), 1.0)

        return pattern


class VShapePattern(PatternGenerator):
    def generate_pattern(self, r_start, pattern_length, r_end):
        if pattern_length < 2:
            raise ValueError("pattern_length should be at least 2")

        # 随机扰动参数
        perturb_r_min = random.uniform(0, 0.4)

        # 分配两个阶段的随机长度比例
        proportions = [random.uniform(0.6, 0.75)]
        proportions.append(1.0 - proportions[0])

        # 计算各阶段的长度
        decline_length = int(pattern_length * proportions[0])
        recovery_length = pattern_length - decline_length

        # 确保总长度为 pattern_length
        total_length = decline_length + recovery_length
        if total_length != pattern_length:
            recovery_length += pattern_length - total_length

        # 设置最低点的可靠性值
        #r_min = pattern_min_r + perturb_r_min
        r_min = random.uniform(0.9,0.7)

        # 初始化 pattern 列表
        pattern = [1.0] * pattern_length

        # 下降期
        for i in range(decline_length):
            perturb_value = 0
            # perturb_value = random.uniform(-0.02, 0.02)
            pattern[i] = (r_start - (r_start - r_min) *
                          (i / decline_length)) + perturb_value
            pattern[i] = min(max(pattern[i], 0.0), 1.0)

        # 恢复期
        for i in range(decline_length, pattern_length):
            perturb_value = 0
            # perturb_value = random.uniform(-0.02, 0.02)
            pattern[i] = (r_min + (r_end - r_min) * (
                (i - decline_length) / recovery_length)) + perturb_value
            pattern[i] = min(max(pattern[i], 0.0), 1.0)

        return pattern




# 生成 pattern
# pattern = generate_w_shape_pattern()
if __name__ == "__main__":
    from matplotlib import pyplot as plt

    start_time = int(time.time())
    current_time = start_time + STATE_LEN
    pattern_length = random.randint(10,30)*10
    pattern = VShapePattern().generate_pattern(0.95, pattern_length, 0.95)
    state = []
    for i in range(0,NUM_STEP):
        for j in range(current_time - STATE_LEN, current_time):
            idx = (j - start_time) % len(pattern)
            #print(current_time, start_time, idx)
            state.append(pattern[idx])
            if idx == len(pattern) - 1: 
                end_state = random.uniform(0.7, 1.0)
                pattern_length = random.randint(3,6)*10
                pattern = generate_pattern(pattern[idx], pattern_length, end_state) 
                #print("pattern_len:", len(pattern)) 
                start_time = current_time
   
        current_time += STATE_LEN
       
    
    np.array(state, dtype=np.float32)  
    plt.plot(np.arange(len(state)), state)
    plt.show()