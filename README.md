# edgeRAD
Resource-Efficient Reliability Anomaly Detection for Edge Services via Deep Reinforcement Learning


# config/application.yml contains the service’s configuration settings
# docker-compose.yaml contains the container startup configuration

# Build the base image edge-r:base
./base_build.sh

# Package the service and anomaly-detection module into images
./build.sh

# Start the containers (both the service and the detection module will run)
docker-compose -f docker-compose.yaml up

cd doc

# Register the service
./regist.sh

# Send a request to the service
http://10.47.10.60:9998/trace/invoke?serviceName=A&traceID=23&returnImmediately=true&reqID=-1
