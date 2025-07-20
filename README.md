edgeRAD：Resource-Efficient Reliability Anomaly Detection for Edge Services via Deep Reinforcement Learning

Prerequisites
- Docker >= 20.10  
- Docker Compose >= 1.29  
- docker images: openjdk:8

1. Build the docker image edge-r:base <br>
./base_build.sh

2. Package service (hello.jar) and anomaly-detection module into docker image service-rl:latest<br>
./build.sh

- hello.jar is a configurable service that enables the flexible specification of inter-service invocation relationships via a configuration file, such as post_data.json<br>
* config/application.yml contains the service configuration settings

3. Start a container (both the service and the detection module will run)<br>
docker-compose -f docker-compose.yaml up<br>
output:<br>
Container service-rl  Created<br>
Attaching to service-rl<br>
service-rl  | Started Java process (PID=8), logging to hello.log<br>
service-rl  | executing... ddpg_train.py<br>
service-rl  | Using device:  cpu<br>

- docker-compose.yaml contains the container startup configuration<br>
- EXP_MODE = simulation (default) (Set EXP_MODE=real to obtain live data stream instead of simulation)

4. Register service (wait more than 10 seconds after starting the container)<br>
cd doc<br>
./regist.sh

5. send a request to the service<br> 
curl "http://10.47.10.60:9998/trace/invoke?serviceName=A&traceID=23&returnImmediately=true&reqID=-1"




