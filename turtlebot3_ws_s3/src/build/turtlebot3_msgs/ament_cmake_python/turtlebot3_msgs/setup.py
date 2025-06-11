from setuptools import find_packages
from setuptools import setup

setup(
    name='turtlebot3_msgs',
    version='2.4.0',
    packages=find_packages(
        include=('turtlebot3_msgs', 'turtlebot3_msgs.*')),
)
