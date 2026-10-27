import os
from glob import glob

from setuptools import find_packages, setup


package_name = 'tb4_gesture_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='engquan5',
    maintainer_email='engquan5@todo.todo',
    description='Gesture recognition and TurtleBot 4 behavior control.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gesture_recognition_node = '
            'tb4_gesture_control.gesture_recognition_node:main',
            'robot_behavior_node = '
            'tb4_gesture_control.robot_behavior_node:main',
        ],
    },
)
