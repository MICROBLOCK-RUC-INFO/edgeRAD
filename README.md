edgeRAD：Resource-Efficient Reliability Anomaly Detection for Edge Services via Deep Reinforcement Learning


config/application.yml contains the service configuration settings<br>
docker-compose.yaml contains the container startup configuration

1. Build the base image edge-r:base <br>
./base_build.sh

2. Package the service and anomaly-detection module into images<br>
./build.sh

3. Start the containers (both the service and the detection module will run)<br>
docker-compose -f docker-compose.yaml up

5. Register service<br>
cd doc<br>
./regist.sh

6. send a request to the service<br> 
http://10.47.10.60:9998/trace/invoke?serviceName=A&traceID=23&returnImmediately=true&reqID=-1
