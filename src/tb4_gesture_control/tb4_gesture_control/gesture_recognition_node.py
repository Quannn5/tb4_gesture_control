import os
import time

import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from tb4_gesture_control.gesture_control import (
    WaveDetector,
    classify_finger_pose,
    count_extended_fingers,
)

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


class GestureRecognitionNode(Node):
    def __init__(self):
        super().__init__('gesture_recognition_node')
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('camera_width', 640)
        self.declare_parameter('camera_height', 480)
        self.declare_parameter('capture_rate_hz', 15.0)
        self.declare_parameter('model_path', '~/.ros/gesture_recognizer.task')
        self.declare_parameter('confidence_threshold', 0.60)
        self.declare_parameter('stable_frames', 3)
        self.declare_parameter('show_window', True)
        self.declare_parameter('gesture_topic', '/detected_gesture')
        self.declare_parameter(
            'status_topic', '/gesture_control/status'
        )

        model_path = os.path.expanduser(
            str(self.get_parameter('model_path').value)
        )
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f'MediaPipe model not found at {model_path}. Download the '
                'Gesture Recognizer .task model; see the package README.'
            )

        confidence_threshold = float(
            self.get_parameter('confidence_threshold').value
        )
        stable_frames = int(self.get_parameter('stable_frames').value)
        capture_rate = float(self.get_parameter('capture_rate_hz').value)
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError('confidence_threshold must be between 0 and 1.')
        if stable_frames < 1:
            raise ValueError('stable_frames must be at least one.')
        if capture_rate <= 0.0:
            raise ValueError('capture_rate_hz must be greater than zero.')
        self.confidence_threshold = confidence_threshold
        self.stable_frames = stable_frames
        self.show_window = bool(self.get_parameter('show_window').value)
        self.window_name = 'TurtleBot 4 hand gestures'
        self.window_ready = False
        self.candidate_gesture = 'none'
        self.candidate_count = 0
        self.stable_gesture = 'none'
        self.stable_confidence = 0.0
        self.robot_status = 'Waiting for behavior selection'
        self.wave_detector = WaveDetector()
        self.last_camera_warning = 0.0

        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError as error:
            raise RuntimeError(
                'MediaPipe is required. Install it in the ROS Python '
                'environment as described in the package README.'
            ) from error

        options = vision.GestureRecognizerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=confidence_threshold,
            min_hand_presence_confidence=confidence_threshold,
            min_tracking_confidence=confidence_threshold,
        )
        self.mp = mp
        self.recognizer = vision.GestureRecognizer.create_from_options(options)

        camera_index = int(self.get_parameter('camera_index').value)
        self.camera = cv2.VideoCapture(camera_index)
        self.camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            int(self.get_parameter('camera_width').value),
        )
        self.camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            int(self.get_parameter('camera_height').value),
        )
        if not self.camera.isOpened():
            self.camera.release()
            self.recognizer.close()
            raise RuntimeError(
                f'Could not open webcam index {camera_index}. Check camera '
                'permissions and the camera_index parameter.'
            )

        gesture_topic = str(self.get_parameter('gesture_topic').value)
        self.publisher = self.create_publisher(String, gesture_topic, 10)
        self.status_subscription = self.create_subscription(
            String,
            str(self.get_parameter('status_topic').value),
            self.status_callback,
            10,
        )
        self.timer = self.create_timer(1.0 / capture_rate, self.process_frame)
        self.get_logger().info(
            f'Webcam {camera_index} is active; publishing '
            f'gesture|confidence heartbeats to {gesture_topic}.'
        )

    def status_callback(self, message):
        self.robot_status = message.data

    def classify(self, result, timestamp):
        if not result.hand_landmarks:
            self.wave_detector.reset()
            return 'none', 0.0

        landmarks = result.hand_landmarks[0]
        landmark_points = [
            (landmark.x, landmark.y) for landmark in landmarks
        ]
        finger_count = count_extended_fingers(landmark_points)
        categories = result.gestures[0] if result.gestures else ()
        category = categories[0] if categories else None
        confidence = float(category.score) if category is not None else 0.0
        finger_pose = classify_finger_pose(landmark_points)
        if finger_pose is not None:
            self.wave_detector.reset()
            return finger_pose, max(confidence, 0.80)
        if finger_count >= 3:
            if self.wave_detector.update(landmarks[0].x, timestamp):
                return 'wave', max(confidence, 0.80)
            if finger_count == 4:
                return 'open_palm', confidence
            return 'none', confidence
        self.wave_detector.reset()

        labels = {
            'Pointing_Up': 'pointing_up',
            'Thumb_Up': 'thumbs_up',
            'Open_Palm': 'open_palm',
            'Closed_Fist': 'closed_fist',
        }
        gesture = (
            labels.get(category.category_name, 'none')
            if category is not None
            else 'none'
        )
        if category is None:
            return 'none', confidence
        if confidence < self.confidence_threshold:
            return 'none', confidence
        return gesture, confidence

    def process_frame(self):
        success, frame = self.camera.read()
        if not success:
            now = time.monotonic()
            if now - self.last_camera_warning >= 2.0:
                self.get_logger().warning(
                    'Could not read a webcam frame; the gesture heartbeat '
                    'is paused so the robot watchdog can stop motion.'
                )
                self.last_camera_warning = now
            return

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = int(time.monotonic() * 1000)
        mp_image = self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB,
            data=rgb_frame,
        )
        result = self.recognizer.recognize_for_video(mp_image, timestamp_ms)
        raw_gesture, confidence = self.classify(
            result, timestamp_ms / 1000.0
        )

        if raw_gesture == self.candidate_gesture:
            self.candidate_count += 1
        else:
            self.candidate_gesture = raw_gesture
            self.candidate_count = 1
        if self.candidate_count >= self.stable_frames:
            self.stable_gesture = raw_gesture
            self.stable_confidence = confidence

        self.publish_gesture(
            self.stable_gesture,
            self.stable_confidence,
        )
        if self.show_window:
            self.show_frame(frame, result.hand_landmarks)

    def publish_gesture(self, gesture, confidence):
        message = String()
        message.data = f'{gesture}|{confidence:.3f}'
        self.publisher.publish(message)

    def show_frame(self, frame, hand_landmarks):
        height, width = frame.shape[:2]
        for hand in hand_landmarks:
            points = [
                (
                    min(max(int(landmark.x * width), 0), width - 1),
                    min(max(int(landmark.y * height), 0), height - 1),
                )
                for landmark in hand
            ]
            for start, end in HAND_CONNECTIONS:
                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (255, 180, 0),
                    2,
                    cv2.LINE_AA,
                )
            for point in points:
                cv2.circle(frame, point, 4, (0, 255, 255), -1, cv2.LINE_AA)

        label = f'{self.stable_gesture} ({self.stable_confidence:.2f})'
        cv2.putText(
            frame,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        status_background = frame.copy()
        cv2.rectangle(
            status_background,
            (0, height - 58),
            (width, height),
            (20, 20, 20),
            -1,
        )
        cv2.addWeighted(status_background, 0.70, frame, 0.30, 0, frame)
        status = self.robot_status[:72]
        cv2.putText(
            frame,
            status,
            (16, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        try:
            if not self.window_ready:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                self.window_ready = True
            cv2.imshow(self.window_name, frame)
            cv2.waitKey(1)
        except cv2.error as error:
            self.get_logger().error(
                f'Could not display the webcam window: {error}'
            )
            self.show_window = False

    def destroy_node(self):
        self.camera.release()
        self.recognizer.close()
        if self.window_ready:
            cv2.destroyWindow(self.window_name)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = GestureRecognitionNode()
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
