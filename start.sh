#!/bin/sh

# 启动 Java 服务，后台运行，日志输出到 hello.log
nohup java -server \
    -Xms1536m \
    -Xmx1536m \
    -XX:MinHeapFreeRatio=20 \
    -XX:MaxHeapFreeRatio=80 \
    -XX:GCTimeRatio=19 \
    -XX:AdaptiveSizePolicyWeight=10 \
    -XX:+UseG1GC \
    -XX:+AlwaysPreTouch \
    -XX:+HeapDumpOnOutOfMemoryError \
    -XX:HeapDumpPath=/heap.log \
    -jar hello.jar > src/edgeRAD/ddpg/log/hello.log 2>&1 &

#    -jar hello.jar > src/edgeRAD/ddpg/log/hello.log 2>&1 &
#    -jar hello.jar > logs/hello.log 2>&1 &





JAVA_PID=$!
echo "Started Java process (PID=$JAVA_PID), logging to hello.log"


# 执行 Python 脚本
cd src/edgeRAD/ddpg
echo "executing... ddpg_train.py"
python -u ddpg_train.py

#> logs/ddpg.log 2>&1 