import random
import math
from typing import List, Dict
import time

class FastTruncatedBinomialSampler:
    def __init__(self, V: int):
        if V <= 0:
            raise ValueError("V must be positive")
        self.V = V
        self.rand = random.Random()
        # 结果缓存，避免多次创建列表
        self._positions = [0] * V

    def sample_positions(self, p: float) -> List[int]:
        """
        1) 精确采样截断二项分布得到 k>=0
        2) 部分 Fisher–Yates 在 O(k) 内生成 k 个位置
        返回长度为 k 的列表，包含 [0..V-1] 中随机选出的 k 个不重复位置
        """
        k = self._sample_truncated_binomial(p)
        if k == 0:
            return []
        # 部分 Fisher–Yates 打乱前 k 步，利用懒映射
        mapping: Dict[int, int] = {}
        for i in range(k):
            j = i + self.rand.randrange(self.V - i)
            vi = mapping.get(i, i)
            vj = mapping.get(j, j)
            mapping[i] = vj
            mapping[j] = vi
            self._positions[i] = vj
        # 返回恰好长度为 k 的新列表
        return self._positions[:k]

    def _sample_truncated_binomial(self, p: float) -> int:
        """
        精确采样一次截断二项分布 (k>=0)
        时间复杂度: O(k*)，期望 O((V*p)/(1-(1-p)**V))

        特殊情况处理：
          p <= 0   -> 返回 0
          p >= 1   -> 返回 V
        """
        # 特殊边界
        if p <= 0:
            return 0
        if p >= 1:
            return self.V

        log1mP = math.log1p(-p)
        P0 = math.exp(self.V * log1mP)
        Z = 1.0 - P0

        u_prime = self.rand.random() * Z
        # 初始 k = 1 时的概率 P1
        Pk = (self.V * p * math.exp((self.V - 1) * log1mP)) / Z
        cum = Pk
        k = 1
        # 累加直到超过 u_prime
        while cum < u_prime:
            k += 1
            # 递推 Pk = P(k-1) * (V-(k-1))/k * p/(1-p)
            Pk = Pk * (self.V - (k - 1)) / k * (p / (1 - p))
            cum += Pk
        return k


if __name__ == "__main__":
    sampler = FastTruncatedBinomialSampler(50)

    p = 0.02
    start = time.perf_counter()
    result = sampler.sample_positions(p)
    end = time.perf_counter()

    elapsed_sec = end - start
    elapsed_us = elapsed_sec * 1e3  # 转换为微秒

    print(f"Sampled value: {result}")
    print(f"Elapsed time: {elapsed_sec:.9f} seconds ({elapsed_us:.3f} ms)")
    

