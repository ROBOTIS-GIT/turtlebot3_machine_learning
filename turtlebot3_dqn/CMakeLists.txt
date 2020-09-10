################################################################################
# Set minimum required version of cmake, project name and compile options
################################################################################
cmake_minimum_required(VERSION 2.8.3)
project(turtlebot3_dqn)

################################################################################
# Find catkin packages and libraries for catkin and system dependencies
################################################################################
find_package(catkin REQUIRED COMPONENTS rospy)

################################################################################
# Setup for python modules and scripts
################################################################################
catkin_python_setup()

################################################################################
# Declare ROS messages, services and actions
################################################################################

################################################################################
# Declare ROS dynamic reconfigure parameters
################################################################################

################################################################################
# Declare catkin specific configuration to be passed to dependent projects
################################################################################
catkin_package(
  CATKIN_DEPENDS rospy
)

################################################################################
# Build
################################################################################

################################################################################
# Install
################################################################################
catkin_install_python(PROGRAMS
  nodes/turtlebot3_dqn.py
  nodes/turtlebot3_dqn_her.py
  nodes/result_graph.py
  nodes/action_graph.py
  nodes/moving_obstacle.py
  nodes/combination_obstacle_1.py
  nodes/combination_obstacle_2.py
  src/${PROJECT_NAME}/environment_stage_1.py
  src/${PROJECT_NAME}/environment_stage_2.py
  src/${PROJECT_NAME}/environment_stage_3.py
  src/${PROJECT_NAME}/environment_stage_4.py
  src/${PROJECT_NAME}/respawnGoal.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch save_model
	DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

################################################################################
# Test
################################################################################
