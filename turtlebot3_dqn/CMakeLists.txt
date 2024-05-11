################################################################################
# Set minimum required version of cmake, project name and compile options
################################################################################
cmake_minimum_required(VERSION 2.8.3)
project(turtlebot3_dqn)

################################################################################
# Find catkin packages and libraries for catkin and system dependencies
################################################################################
find_package(catkin REQUIRED COMPONENTS rospy roscpp std_msgs message_generation)

add_service_files(
  FILES
  PtModel.srv
  C2SPtModel.srv
  S2CPtModel.srv
  LocalTrain.srv
)

################################################################################
# Setup for python modules and scripts
################################################################################
catkin_python_setup()

generate_messages(
  DEPENDENCIES
  std_msgs
)

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
  CATKIN_DEPENDS roscpp rospy std_msgs message_runtime
)

################################################################################
# Build
################################################################################
include_directories(
  ${catkin_INCLUDE_DIRS}
)
################################################################################
# Install
################################################################################
catkin_install_python(PROGRAMS
  nodes/turtlebot3_dqn_stage_1
  nodes/turtlebot3_dqn_stage_2
  nodes/turtlebot3_dqn_stage_3
  nodes/turtlebot3_dqn_stage_4
  nodes/result_graph
  nodes/action_graph
  nodes/moving_obstacle
  nodes/save_result.py
  nodes/combination_obstacle_1
  nodes/combination_obstacle_2
  src/${PROJECT_NAME}/environment_stage_1.py
  src/${PROJECT_NAME}/environment_stage_2.py
  src/${PROJECT_NAME}/environment_stage_3.py
  src/${PROJECT_NAME}/environment_stage_4.py
  src/${PROJECT_NAME}/respawnGoal.py
  nodes/upload_model.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch save_model
	DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

################################################################################
# Test
################################################################################
