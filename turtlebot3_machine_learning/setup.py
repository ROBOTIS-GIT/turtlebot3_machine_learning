from setuptools import find_packages
from setuptools import setup

package_name = 'turtlebot3_machine_learning'
authors_info = [
    ('Gilbert', 'kkjong@robotis.com'),
    ('Ryan Shim', 'N/A'),
    ('ChanHyeong Lee', 'dddoggi1207@gmail.com'),
]
authors = ', '.join(author for author, _ in authors_info)
author_emails = ', '.join(email for _, email in authors_info)

setup(
    name=package_name,
    version='1.0.1',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author=authors,
    author_email=author_emails,
    maintainer='Pyo',
    maintainer_email='pyo@robotis.com',
    description='ROS 2 mata package for TurtleBot3 machine learning',
    license='Apache 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
