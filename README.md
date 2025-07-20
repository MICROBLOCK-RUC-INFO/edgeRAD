edgeRAD：Resource-Efficient Reliability Anomaly Detection for Edge Services via Deep Reinforcement Learning

1. Build the docker image edge-r:base <br>
./base_build.sh

2. Package the service and anomaly-detection module into docker image service-rl:latest<br>
./build.sh

3. Start the containers (both the service and the detection module will run)<br>
docker-compose -f docker-compose.yaml up<br>
output:<br>
Container service-rl  Created<br>
Attaching to service-rl<br>
service-rl  | Started Java process (PID=8), logging to hello.log<br>
service-rl  | executing... ddpg_train.py<br>
service-rl  | Using device:  cpu<br>

    EXP_MODE = simulation (default) or real

4. Register service<br>
cd doc<br>
./regist.sh

5. send a request to the service<br> 
http://10.47.10.60:9998/trace/invoke?serviceName=A&traceID=23&returnImmediately=true&reqID=-1

* config/application.yml contains the service configuration settings<br>
* docker-compose.yaml contains the container startup configuration<br>
* hello.jar is a configurable service that enables the flexible specification of inter-service invocation relationships via a configuration file, such as post_data.json
