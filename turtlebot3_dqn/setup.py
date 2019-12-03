import glob
import os

from setuptools import find_packages
from setuptools import setup

package_name = 'turtlebot3_dqn'

setup(
    name=package_name,
    version='2.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob.glob(os.path.join('launch', 'turtlebot3_dqn_stage1.launch.py'))),
        ('share/' + package_name + '/launch', glob.glob(os.path.join('launch', 'turtlebot3_dqn_stage2.launch.py'))),
        ('share/' + package_name + '/launch', glob.glob(os.path.join('launch', 'turtlebot3_dqn_stage3.launch.py'))),
        ('share/' + package_name + '/launch', glob.glob(os.path.join('launch', 'turtlebot3_dqn_stage4.launch.py'))),
    ],
    install_requires=['setuptools','launch'],
    zip_safe=True,
    author=['Gilbert', 'Ryan Shim'],
    author_email=['kkjong@robotis.com', 'jhshim@robotis.com'],
    maintainer='Pyo',
    maintainer_email='pyo@robotis.com',
    keywords=['ROS', 'ROS2', 'examples', 'rclpy'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Topic :: Software Development',
    ],
    description=(
        'DQN for TurtleBot3.'
    ),
    license='Apache License, Version 2.0',
    entry_points={
        'console_scripts': [
            'dqn_agent = turtlebot3_dqn.dqn_agent.dqn_agent:main', 
            'dqn_environment = turtlebot3_dqn.dqn_environment.dqn_environment:main', 
            'dqn_gazebo = turtlebot3_dqn.dqn_gazebo.dqn_gazebo:main', 
            'dqn_test = turtlebot3_dqn.dqn_test.dqn_test:main', 
        ],
    },
)
