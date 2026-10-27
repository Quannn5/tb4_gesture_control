"""Shared gesture protocol and bounded motion sequences."""

from collections import deque
import math
from typing import Optional, Sequence, Tuple


GESTURES = {
    'none',
    'wave',
    'pointing_up',
    'two_fingers_circle',
    'two_fingers_forward',
    'thumbs_up',
    'open_palm',
    'closed_fist',
}

MotionSegment = Tuple[float, float, float]
Landmark2D = Tuple[float, float]
Pose2D = Tuple[float, float, float]


class WaveDetector:
    def __init__(self):
        self.samples = deque()
        self.last_direction = 0
        self.wave_until = 0.0

    def reset(self):
        self.samples.clear()
        self.last_direction = 0
        self.wave_until = 0.0

    def update(self, x_position, timestamp):
        self.samples.append((timestamp, x_position))
        while self.samples and timestamp - self.samples[0][0] > 0.5:
            self.samples.popleft()

        if len(self.samples) < 2:
            return timestamp < self.wave_until

        delta = self.samples[-1][1] - self.samples[-2][1]
        if abs(delta) >= 0.012:
            direction = 1 if delta > 0 else -1
            direction_reversed = (
                self.last_direction and direction != self.last_direction
            )
            self.last_direction = direction
            horizontal_span = (
                max(sample[1] for sample in self.samples)
                - min(sample[1] for sample in self.samples)
            )
            if direction_reversed and horizontal_span >= 0.12:
                self.wave_until = timestamp + 1.2
        return timestamp < self.wave_until


def count_extended_fingers(landmarks: Sequence[Landmark2D]) -> int:
    """Count extended index, middle, ring, and pinky fingers."""
    if len(landmarks) < 21:
        return 0

    def wrist_distance(index):
        x, y = landmarks[index]
        wrist_x, wrist_y = landmarks[0]
        return math.hypot(x - wrist_x, y - wrist_y)

    extended_count = 0
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        pip_distance = wrist_distance(pip)
        if pip_distance > 0.0 and wrist_distance(tip) / pip_distance >= 1.04:
            extended_count += 1
    return extended_count


def classify_finger_pose(
    landmarks: Sequence[Landmark2D],
) -> Optional[str]:
    """Distinguish a two-finger V from pinched two- and three-finger poses."""
    finger_count = count_extended_fingers(landmarks)
    if finger_count != 2:
        return None

    wrist_x, wrist_y = landmarks[0]

    def wrist_distance(index):
        x, y = landmarks[index]
        return math.hypot(x - wrist_x, y - wrist_y)

    hand_scale = wrist_distance(12)
    if hand_scale == 0.0:
        return None

    if finger_count == 2:
        tip_gap = math.hypot(
            landmarks[8][0] - landmarks[12][0],
            landmarks[8][1] - landmarks[12][1],
        )
        separation = tip_gap / hand_scale
        if separation <= 0.14:
            return 'two_fingers_forward'
        if separation >= 0.34:
            return 'two_fingers_circle'
        return None

    return None


def distance_motion_command(
    start_pose: Pose2D,
    current_pose: Pose2D,
    target_distance: float,
) -> Tuple[float, float, bool, float]:
    """Track a signed distance from a start pose using odometry feedback."""
    start_x, start_y, start_yaw = start_pose
    current_x, current_y, current_yaw = current_pose
    delta_x = current_x - start_x
    delta_y = current_y - start_y
    progress = (
        delta_x * math.cos(start_yaw)
        + delta_y * math.sin(start_yaw)
    )
    remaining = target_distance - progress
    if abs(remaining) <= 0.05:
        return 0.0, 0.0, True, progress

    heading_error = math.atan2(
        math.sin(start_yaw - current_yaw),
        math.cos(start_yaw - current_yaw),
    )
    angular = max(-0.3, min(0.3, 1.0 * heading_error))
    speed = min(0.15, max(0.05, 0.5 * abs(remaining)))
    linear = math.copysign(speed, remaining)
    return linear, angular, False, progress


def parse_gesture_message(message: str) -> Optional[Tuple[str, float]]:
    """Parse ``gesture|confidence`` messages from the recognition node."""
    gesture, separator, confidence_text = message.partition('|')
    if not separator or gesture not in GESTURES:
        return None

    try:
        confidence = float(confidence_text)
    except ValueError:
        return None

    if not 0.0 <= confidence <= 1.0:
        return None
    return gesture, confidence


def distance_target_for_behavior(behavior: str) -> float:
    """Return the signed one-meter target for a distance behavior."""
    if behavior == 'closed_fist':
        return -1.0
    if behavior == 'two_fingers_forward':
        return 1.0
    raise ValueError(f'Unsupported distance behavior: {behavior}')


def behavior_segments(behavior: str) -> Tuple[MotionSegment, ...]:
    """Return bounded (duration, linear, angular) motion segments."""
    if behavior == 'wave':
        return (
            (0.65, 0.0, 0.45),
            (0.15, 0.0, 0.0),
            (0.65, 0.0, -0.45),
            (0.15, 0.0, 0.0),
            (0.65, 0.0, 0.45),
            (0.15, 0.0, 0.0),
            (0.65, 0.0, -0.45),
        )

    if behavior == 'pointing_up':
        segments = []
        for _ in range(4):
            segments.extend((
                (1.4, 0.18, 0.0),
                (0.2, 0.0, 0.0),
                (1.4, -0.18, 0.0),
                (0.2, 0.0, 0.0),
            ))
        return tuple(segments)

    if behavior in (
        'closed_fist',
        'two_fingers_forward',
    ):
        linear_speed = 0.15
        direction = distance_target_for_behavior(behavior)
        duration = 1.0 / linear_speed
        return ((duration, direction * linear_speed, 0.0),)

    if behavior == 'two_fingers_circle':
        angular_speed = 0.30
        linear_speed = 0.65 * angular_speed
        duration = (2.0 * math.pi) / angular_speed
        return ((duration, linear_speed, angular_speed),)

    raise ValueError(f'Unsupported behavior: {behavior}')
