import math
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from tb4_gesture_control.gesture_control import (
    behavior_segments,
    distance_target_for_behavior,
    distance_motion_command,
    parse_gesture_message,
)


class RobotBehaviorNode(Node):
    def __init__(self):
        super().__init__('robot_behavior_node')
        self.declare_parameter('gesture_topic', '/detected_gesture')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('control_timeout_sec', 1.0)
        self.declare_parameter('scan_timeout_sec', 1.0)
        self.declare_parameter('obstacle_stop_distance', 0.45)
        self.declare_parameter('require_laser_scan', True)

        self.control_timeout = float(
            self.get_parameter('control_timeout_sec').value
        )
        self.scan_timeout = float(
            self.get_parameter('scan_timeout_sec').value
        )
        self.obstacle_stop_distance = float(
            self.get_parameter('obstacle_stop_distance').value
        )
        self.require_laser_scan = bool(
            self.get_parameter('require_laser_scan').value
        )
        if (
            self.control_timeout <= 0.0
            or self.scan_timeout <= 0.0
        ):
            raise ValueError('Control and scan timeouts must be positive.')
        if self.obstacle_stop_distance <= 0.0:
            raise ValueError('obstacle_stop_distance must be positive.')

        self.state = 'idle'
        self.selected_behavior = None
        self.active_behavior = None
        self.last_gesture = 'none'
        self.last_control_time = None
        self.scan_ranges = None
        self.scan_time = None
        self.current_pose = None
        self.odom_time = None
        self.distance_start_pose = None
        self.distance_target = 0.0
        self.distance_started = 0.0
        self.segments = ()
        self.segment_index = 0
        self.segment_started = 0.0

        cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
        self.publisher = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.status_publisher = self.create_publisher(
            String, '/gesture_control/status', 10
        )
        self.gesture_subscription = self.create_subscription(
            String,
            str(self.get_parameter('gesture_topic').value),
            self.gesture_callback,
            10,
        )
        self.scan_subscription = self.create_subscription(
            LaserScan,
            str(self.get_parameter('scan_topic').value),
            self.scan_callback,
            qos_profile_sensor_data,
        )
        self.odom_subscription = self.create_subscription(
            Odometry,
            str(self.get_parameter('odom_topic').value),
            self.odom_callback,
            10,
        )
        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info(
            f'Waiting for gestures; velocity commands go to {cmd_vel_topic}. '
            'A selected behavior only runs after thumbs_up confirmation.'
        )
        self.publish_status('Ready: select a behavior')

    def gesture_callback(self, message):
        parsed = parse_gesture_message(message.data)
        if parsed is None:
            self.get_logger().warning(
                f'Ignoring malformed gesture message: {message.data!r}'
            )
            return

        gesture, _confidence = parsed
        self.last_control_time = time.monotonic()
        if gesture == self.last_gesture:
            return
        self.last_gesture = gesture

        if gesture == 'open_palm':
            self.stop_behavior('open_palm emergency stop')
        elif gesture in (
            'wave',
            'pointing_up',
            'closed_fist',
            'two_fingers_circle',
            'two_fingers_forward',
        ):
            if self.state in ('idle', 'selected'):
                self.selected_behavior = gesture
                self.state = 'selected'
                self.publish_status(
                    f'Selected: {self.behavior_name(gesture)}. '
                    'Show THUMBS UP to run.'
                )
                self.get_logger().info(
                    f'Selected {gesture}; show thumbs_up to confirm.'
                )
        elif gesture == 'thumbs_up':
            if self.state == 'selected' and self.selected_behavior:
                self.start_behavior(self.selected_behavior)
            else:
                self.get_logger().info(
                    'thumbs_up ignored: select a behavior first.'
                )
                self.publish_status(
                    'No behavior selected. Show a movement gesture first.'
                )

    def scan_callback(self, message):
        self.scan_ranges = (
            list(message.ranges),
            message.angle_min,
            message.angle_increment,
            message.range_min,
            message.range_max,
        )
        self.scan_time = time.monotonic()

    def odom_callback(self, message):
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (
                orientation.w * orientation.z
                + orientation.x * orientation.y
            ),
            1.0 - 2.0 * (
                orientation.y * orientation.y
                + orientation.z * orientation.z
            ),
        )
        self.current_pose = (position.x, position.y, yaw)
        self.odom_time = time.monotonic()

    def start_behavior(self, behavior):
        if behavior in (
            'closed_fist',
            'two_fingers_forward',
        ):
            if (
                self.current_pose is None
                or self.odom_time is None
                or time.monotonic() - self.odom_time > self.scan_timeout
            ):
                self.publish_status(
                    'Cannot move 1 m: waiting for fresh /odom.'
                )
                self.get_logger().error(
                    'Cannot start distance move without fresh odometry.'
                )
                return
            self.segments = ()
            self.segment_index = 0
            self.distance_start_pose = self.current_pose
            self.distance_target = distance_target_for_behavior(behavior)
            self.distance_started = time.monotonic()
            self.selected_behavior = None
            self.active_behavior = behavior
            self.state = 'moving_distance'
            self.publish_status(f'Running: {self.behavior_name(behavior)}')
            self.get_logger().info(
                f'Running {behavior}; odometry target is '
                f'{self.distance_target:.1f} m.'
            )
            return

        self.segments = behavior_segments(behavior)
        self.segment_index = 0
        self.segment_started = time.monotonic()
        self.selected_behavior = None
        self.active_behavior = behavior
        self.state = 'running'
        self.publish_status(f'Running: {self.behavior_name(behavior)}')
        self.get_logger().info(f'Running confirmed behavior: {behavior}.')

    def control_loop(self):
        now = time.monotonic()
        if self.state == 'idle':
            return

        if (
            self.last_control_time is None
            or now - self.last_control_time > self.control_timeout
        ):
            self.selected_behavior = None
            self.stop_behavior('gesture control signal timed out')
            return

        if self.state == 'moving_distance':
            self.control_distance_move(now)
            return
        if self.state != 'running':
            return

        duration, linear_x, angular_z = self.segments[self.segment_index]
        while now - self.segment_started >= duration:
            self.segment_started += duration
            self.segment_index += 1
            if self.segment_index >= len(self.segments):
                self.state = 'idle'
                self.segments = ()
                self.publish_stop()
                completed_behavior = self.active_behavior
                self.active_behavior = None
                self.publish_status(
                    f'Completed: {self.behavior_name(completed_behavior)}. '
                    'Ready for next gesture.'
                )
                self.get_logger().info('Behavior completed.')
                return
            duration, linear_x, angular_z = self.segments[
                self.segment_index
            ]

        if self.obstacle_detected(linear_x, angular_z, now):
            self.stop_behavior('laser safety stop: obstacle or stale scan')
            return

        command = Twist()
        command.linear.x = linear_x
        command.angular.z = angular_z
        self.publisher.publish(command)

    def control_distance_move(self, now):
        if (
            self.current_pose is None
            or self.distance_start_pose is None
            or self.odom_time is None
            or now - self.odom_time > self.scan_timeout
        ):
            self.stop_behavior('odometry unavailable during distance move')
            return
        if now - self.distance_started > 25.0:
            self.stop_behavior('distance move timed out')
            return

        linear_x, angular_z, completed, progress = distance_motion_command(
            self.distance_start_pose,
            self.current_pose,
            self.distance_target,
        )
        if completed:
            completed_behavior = self.active_behavior
            self.state = 'idle'
            self.active_behavior = None
            self.distance_start_pose = None
            self.publish_stop()
            self.publish_status(
                f'Completed: {self.behavior_name(completed_behavior)} '
                f'({progress:.2f} m measured). Ready.'
            )
            self.get_logger().info(
                f'Distance behavior completed at {progress:.2f} m odometry.'
            )
            return
        if self.obstacle_detected(linear_x, angular_z, now):
            self.stop_behavior('laser safety stop during distance move')
            return

        command = Twist()
        command.linear.x = linear_x
        command.angular.z = angular_z
        self.publisher.publish(command)

    def obstacle_detected(self, linear_x, angular_z, now):
        if linear_x == 0.0 and angular_z == 0.0:
            return False
        if not self.require_laser_scan:
            return False
        if (
            self.scan_ranges is None
            or self.scan_time is None
            or now - self.scan_time > self.scan_timeout
        ):
            return True

        ranges, angle_min, angle_increment, range_min, range_max = (
            self.scan_ranges
        )
        if angular_z != 0.0:
            direction = None
        else:
            direction = 0.0 if linear_x >= 0.0 else math.pi

        valid_ranges = []
        for index, distance in enumerate(ranges):
            if (
                not math.isfinite(distance)
                or not range_min <= distance <= range_max
            ):
                continue
            angle = angle_min + index * angle_increment
            if direction is not None:
                angle_difference = math.atan2(
                    math.sin(angle - direction),
                    math.cos(angle - direction),
                )
                if abs(angle_difference) > math.radians(35.0):
                    continue
            valid_ranges.append(distance)

        return (
            not valid_ranges
            or min(valid_ranges) < self.obstacle_stop_distance
        )

    def stop_behavior(self, reason):
        stopped_behavior = self.active_behavior or self.selected_behavior
        self.state = 'idle'
        self.selected_behavior = None
        self.active_behavior = None
        self.distance_start_pose = None
        self.segments = ()
        self.segment_index = 0
        self.publish_stop()
        if reason == 'open_palm emergency stop':
            status = 'Stopped by OPEN PALM. Ready.'
        elif reason == 'gesture control signal timed out':
            status = 'Control signal lost. Robot stopped.'
        elif reason.startswith('laser safety stop'):
            status = 'Obstacle detected. Robot stopped.'
        elif reason == 'odometry unavailable during distance move':
            status = 'Odometry lost. Distance move stopped.'
        elif reason == 'distance move timed out':
            status = 'Distance target not reached in time. Robot stopped.'
        elif stopped_behavior:
            status = (
                f'Stopped: {self.behavior_name(stopped_behavior)}. '
                'Ready.'
            )
        else:
            status = 'Robot stopped. Ready.'
        self.publish_status(status)
        self.get_logger().info(f'Robot stopped: {reason}.')

    @staticmethod
    def behavior_name(behavior):
        names = {
            'wave': 'WAVE',
            'pointing_up': 'POINTING UP',
            'closed_fist': 'CLOSED FIST: BACKWARD 1 M',
            'two_fingers_circle': 'TWO-FINGER V: CIRCLE',
            'two_fingers_forward': 'TWO FINGERS TOGETHER: FORWARD 1 M',
        }
        return names.get(behavior, 'behavior')

    def publish_status(self, status):
        message = String()
        message.data = status
        self.status_publisher.publish(message)

    def publish_stop(self):
        self.publisher.publish(Twist())

    def destroy_node(self):
        self.publish_stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RobotBehaviorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
