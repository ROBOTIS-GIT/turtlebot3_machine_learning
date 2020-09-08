FROM osrf/ros:kinetic-desktop-full

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-kinetic-robot=1.3.2-0* \
    && rm -rf /var/lib/apt/lists/*

COPY scripts/ my_scripts/

# APT
RUN apt update
RUN apt upgrade --yes

#Anaconda 
RUN apt install wget
RUN cd tmp/ && wget https://repo.continuum.io/archive/Anaconda2-5.0.1-Linux-x86_64.sh  &&  yes yes | bash Anaconda2-5.0.1-Linux-x86_64.sh
RUN python -V

#source
RUN /bin/bash -c "source ~/.bashrc"

#PIP 
RUN apt install python-pip --yes
RUN /tmp/yes/bin/pip install --upgrade pip
RUN /tmp/yes/bin/pip uninstall numpy --yes && /tmp/yes/bin/pip install numpy==1.16
RUN /tmp/yes/bin/pip install psycopg2

# ROS dependency packages
RUN /tmp/yes/bin/pip install rosinstall msgpack defusedxml netifaces
RUN /tmp/yes/bin/pip install catkin_pkg
RUN /tmp/yes/bin/pip install empy
RUN /tmp/yes/bin/pip install pyqtgraph
RUN yes yes | apt install ros-kinetic-interactive-markers ros-kinetic-gazebo-ros

#Tensorflow
# if want to use with conda
# RUN /bin/bash -c "source /tmp/yes/bin/activate"  && conda create -n tensorflow pip python=2.7 
RUN /tmp/yes/bin/pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.8.0-cp27-none-linux_x86_64.whl
RUN /tmp/yes/bin/pip install keras==2.1.5

#Catkin Workspace
RUN mkdir -p /root/catkin_ws/src
RUN git clone https://github.com/ROBOTIS-GIT/turtlebot3.git /root/catkin_ws/src/turtlebot3 --branch kinetic-devel
RUN git clone https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git /root/catkin_ws/src/turtlebot3_msgs --branch kinetic-devel
RUN git clone https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git /root/catkin_ws/src/turtlebot3_simulations --branch kinetic-devel
RUN touch ros_own_entrypoint.sh && sed "2d" ros_entrypoint.sh > ros_own_entrypoint.sh
RUN /bin/bash -c '. /opt/ros/kinetic/setup.bash; cd ~/catkin_ws; catkin_make'
RUN echo "REBUILD FROM HERE"
RUN git clone https://github.com/AlessioAnalytics/turtlebot3_machine_learning.git /root/catkin_ws/src/turtlebot3_machine_learning
RUN /bin/bash -c '. /opt/ros/kinetic/setup.bash; cd ~/catkin_ws; catkin_make'

RUN echo "export TURTLEBOT3_MODEL=burger" >> ~/.bashrc
RUN echo "source ros_own_entrypoint.sh" >> ~/.bashrc
RUN echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc 
RUN echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc

RUN sed -i 's/<xacro:arg name="laser_visual" default="false"\/>/<xacro:arg name="laser_visual" default="true"\/>/g' /root/catkin_ws/src/turtlebot3/turtlebot3_description/urdf/turtlebot3_burger.gazebo.xacro 
RUN sed -i 's/<samples>360<\/samples>/<samples>24<\/samples>/g' /root/catkin_ws/src/turtlebot3/turtlebot3_description/urdf/turtlebot3_burger.gazebo.xacro

#Other
RUN apt install vim --yes
RUN mkdir /root/logs
RUN mkdir /root/cfg
