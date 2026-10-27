import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory(
        'tb4_gesture_control'
    )
    parameters = os.path.join(
        package_share, 'config', 'gesture_control.yaml'
    )
    return LaunchDescription([
        Node(
            package='tb4_gesture_control',
            executable='gesture_recognition_node',
            name='gesture_recognition_node',
            output='screen',
            parameters=[parameters],
        ),
        Node(
            package='tb4_gesture_control',
            executable='robot_behavior_node',
            name='robot_behavior_node',
            output='screen',
            parameters=[parameters],
        ),
    ])
