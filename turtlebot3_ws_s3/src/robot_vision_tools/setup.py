from setuptools import setup

package_name = 'robot_vision_tools'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your@email.com',
    description='Tools for processing and saving robot vision data',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_image_saver = robot_vision_tools.camera_image_saver:main',
            'nav2_state_monitor = robot_vision_tools.nav2_state_monitor:main',
            'simplified_demo = robot_vision_tools.simplified_demo:main',
        ],
    },
)
