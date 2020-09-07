sudo docker run -it --env="DISPLAY" --env="QT_X11_NO_MITSHM=1" -v "/home/lmueller/saved_models_2:/root/catkin_ws/src/turtlebot3_machine_learning/turtlebot3_dqn/save_model:rw" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" docker_setup_ros_ml

