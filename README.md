# Object Detection Demo


This here is a demo project that performs object detection on live video stream using Pytorch. Please note that cuda has not been used in the project and hence the app can be really slow

![alt text](assets/index.jpeg?raw=true "Title")
## Docker Hub

The docker image for the project can be found [here](https://hub.docker.com/repository/docker/mkprsnt/objdetectordemo).

To run the application with video feed, pls use the following command.

```sh
docker run -p 8000:8000 --device /dev/video0 mkprsnt/objdetectordemo
```
## Kubernetes
Parameters for deployemnt to kubernetes like number of replicas, ports etc., are specified in the deployment.yaml file   
```sh
##Deployment 
spec:
  replicas: 1
  selector:
    matchLabels:
    .
    .
    .
    .
##Service
    ports:
    - protocol: TCP
      port: 8000
```
   
After deciding the appropriate parameters for your deployment, the app can be deployed in kubernetes using the following commands

```sh
kubectl apply -f deployment.yaml
```
Thats all Folks!
