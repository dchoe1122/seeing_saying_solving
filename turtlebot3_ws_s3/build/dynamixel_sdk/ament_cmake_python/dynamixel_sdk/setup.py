from setuptools import find_packages
from setuptools import setup

setup(
    name='dynamixel_sdk',
    version='3.8.4',
    packages=find_packages(
        include=('dynamixel_sdk', 'dynamixel_sdk.*')),
)
