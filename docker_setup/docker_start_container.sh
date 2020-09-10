sudo docker run -it --env="DISPLAY" --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --volume="/home/lwidowski/RoboticsProjects/catkin_ws/src/turtlebot3_machine_learning/docker_setup/cfg/:/root/cfg:rw" docker_setup_ros_ml

