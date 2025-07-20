# 第一阶段：使用 openjdk:8 作为基础镜像
FROM openjdk:8 AS jdk-builder


# 第二阶段：使用 edge-rl:base 作为最终镜像
FROM edge-rl:base

# 从第一阶段复制所需文件
COPY --from=jdk-builder /usr/local/openjdk-8 /usr/local/openjdk-8


#设置工作目录
WORKDIR /app/edgeRAD
COPY hello.jar start.sh /app/edgeRAD/
COPY config /app/edgeRAD/config
COPY src /app/edgeRAD/src

# 设置可执行权限
RUN chmod +x hello.jar \
    start.sh \
    src/edgeRAD/ddpg/ddpg_run.py

# 设置环境变量
ENV JAVA_HOME=/usr/local/openjdk-8
ENV PATH="$JAVA_HOME/bin:${PATH}"
ENV PYTHONPATH=$PYTHONPATH:/app/edgeRAD/src

# 暴露端口
#EXPOSE 9998

# 启动脚本
CMD ["sh", "-c", "/app/edgeRAD/start.sh"]