^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package turtlebot3_machine_learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1.0.3 (2026-01-19)
------------------
* Added ament_cmake as a buildtool_depend
* Contributors: Hyungyu Kim

1.0.2 (2026-01-06)
------------------
* Fixed a bug in the JSON file where the step parameter was incorrectly named; renamed it to step_counter.
* Changed the system arguments to be passed as ROS parameters for execution.
* Added a use_gpu parameter to allow selection of whether to use GPU.
* Added a model_file parameter to enable loading an existing trained model and continuing training.
* Renamed the load_model variable to use_pretrained_model for clarity.
* Changed model_path from a class variable to a local variable.
* Introduced lazy import for TensorFlow modules.
* Contributors: Hyungyu Kim

1.0.1 (2025-05-02)
------------------
* Support for ROS 2 Jazzy version
* Gazebo simulation support for the package
* Contributors: ChanHyeong Lee

1.0.0 (2025-04-17)
------------------
* Support for ROS 2 Humble version
* Renewal of package structure
* Improved behavioral rewards for agents
* Contributors: ChanHyeong Lee
