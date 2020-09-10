echo "Stage: $1";
roscore &
sleep 2
roslaunch turtlebot3_dqn turtlebot3_environment.launch stage:="$1" &
sleep 2
roslaunch turtlebot3_dqn turtlebot3_dqn.launch stage:="$1" &
sleep 2
roslaunch turtlebot3_dqn result_graph.launch
