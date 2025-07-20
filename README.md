# edgeRAD
Resource-Efficient Reliability Anomaly Detection for Edge Services via Deep Reinforcement Learning


config/application.yml contains the service configuration settings
docker-compose.yaml contains the container startup configuration

1. Build the base image edge-r:base
   
./base_build.sh

3. Package the service and anomaly-detection module into images
   
./build.sh

4. Start the containers (both the service and the detection module will run)
   
docker-compose -f docker-compose.yaml up

6. Register service
   
cd doc
./regist.sh

8. send a request to the service
   
http://10.47.10.60:9998/trace/invoke?serviceName=A&traceID=23&returnImmediately=true&reqID=-1
