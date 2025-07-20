import pymysql
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np

# 1. 建立数据库连接（请替换 host/user/password 等）
conn = pymysql.connect(
    host='10.77.70.181',
    port=3306,
    user='root',
    password='1234qwer',
    database='service_monitoring',
    charset='utf8'
)


start = time.time()
# 2. 执行 SQL 并读取到 DataFrame
sql = "SELECT r FROM service_state WHERE service_name = 'D' and qps > 5"
# 注意：如果你有时间戳或自增 id 字段，最好在 ORDER BY 中指定，这样画出来的曲线才有意义
df = pd.read_sql(sql, conn)


# 3. 关闭连接
conn.close()

# 4. 画图
plt.figure(figsize=(8,4))
plt.plot(
    df.index,
    df['r'],
    marker='o',
    markersize=1,     # 点的大小，默认为 6，改小到 3
    linewidth=1       # 如果也想细一些线条，可以调小 linewidth
)
plt.xlabel('样本序号')
plt.ylabel('r 值')
plt.title("Service A 的 r 值曲线")
plt.grid(True)
plt.tight_layout()
plt.show()