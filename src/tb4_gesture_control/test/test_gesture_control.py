import math

import pytest

from tb4_gesture_control.gesture_control import (
    WaveDetector,
    behavior_segments,
    classify_finger_pose,
    count_extended_fingers,
    distance_target_for_behavior,
    distance_motion_command,
    parse_gesture_message,
)


@pytest.mark.parametrize(
    ('message', 'expected'),
    [
        ('wave|0.900', ('wave', 0.9)),
        ('none|0.000', ('none', 0.0)),
        ('thumbs_up|1.000', ('thumbs_up', 1.0)),
        ('closed_fist|0.800', ('closed_fist', 0.8)),
        ('two_fingers_circle|0.800', ('two_fingers_circle', 0.8)),
        ('two_fingers_forward|0.800', ('two_fingers_forward', 0.8)),
    ],
)
def test_parse_gesture_message(message, expected):
    assert parse_gesture_message(message) == expected


@pytest.mark.parametrize(
    'message',
    [
        'wave',
        'unknown|0.9',
        'three_fingers_backward|0.8',
        'wave|not-a-number',
        'wave|1.1',
        'wave|-0.1',
    ],
)
def test_parse_gesture_message_rejects_invalid_input(message):
    assert parse_gesture_message(message) is None


def test_pointing_up_is_four_bounded_forward_backward_cycles():
    segments = behavior_segments('pointing_up')
    moving = [segment for segment in segments if segment[1] != 0.0]
    assert len(moving) == 8
    assert sum(segment[1] > 0.0 for segment in moving) == 4
    assert sum(segment[1] < 0.0 for segment in moving) == 4
    assert max(abs(segment[1]) for segment in moving) <= 0.18
    assert all(segment[0] == 1.4 for segment in moving)
    assert all(
        segment[0] * abs(segment[1]) == pytest.approx(0.252)
        for segment in moving
    )


@pytest.mark.parametrize(
    ('behavior', 'expected_speed'),
    [
        ('closed_fist', -0.15),
        ('two_fingers_forward', 0.15),
    ],
)
def test_pinched_finger_behaviors_drive_straight_one_meter(
    behavior, expected_speed
):
    duration, linear_speed, angular_speed = behavior_segments(behavior)[0]
    assert duration == pytest.approx(1.0 / 0.15)
    assert linear_speed == expected_speed
    assert angular_speed == 0.0
    assert duration * abs(linear_speed) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ('behavior', 'target'),
    [
        ('closed_fist', -1.0),
        ('two_fingers_forward', 1.0),
    ],
)
def test_distance_behavior_targets(behavior, target):
    assert distance_target_for_behavior(behavior) == target


def test_two_finger_v_runs_one_full_circle():
    duration, linear_speed, angular_speed = behavior_segments(
        'two_fingers_circle'
    )[0]
    assert linear_speed / angular_speed == pytest.approx(0.65)
    assert duration * angular_speed == pytest.approx(2.0 * math.pi)


def test_distance_motion_commands_backward_for_negative_target():
    linear, angular, completed, progress = distance_motion_command(
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        -1.0,
    )
    assert linear == pytest.approx(-0.15)
    assert angular == 0.0
    assert not completed
    assert progress == 0.0


def test_distance_motion_only_completes_after_odometry_reaches_target():
    linear, angular, completed, progress = distance_motion_command(
        (0.0, 0.0, 0.0),
        (-0.96, 0.0, 0.0),
        -1.0,
    )
    assert linear == 0.0
    assert angular == 0.0
    assert completed
    assert progress == pytest.approx(-0.96)


def test_unknown_behavior_is_rejected():
    with pytest.raises(ValueError):
        behavior_segments('not-a-behavior')


def test_wave_detector_requires_horizontal_direction_changes():
    detector = WaveDetector()
    samples = [
        (0.0, 0.30),
        (0.1, 0.36),
        (0.2, 0.42),
        (0.3, 0.48),
        (0.4, 0.44),
    ]

    assert [detector.update(x, timestamp) for timestamp, x in samples] == [
        False,
        False,
        False,
        False,
        True,
    ]
    assert detector.update(0.44, 0.5)
    assert detector.update(0.44, 1.4)
    assert not detector.update(0.44, 1.6)


def test_finger_counter_detects_two_extended_fingers():
    landmarks = [(0.5, 0.9)] * 21
    landmarks[5] = (0.43, 0.65)
    landmarks[6] = (0.42, 0.48)
    landmarks[8] = (0.40, 0.20)
    landmarks[9] = (0.50, 0.62)
    landmarks[10] = (0.50, 0.40)
    landmarks[12] = (0.55, 0.15)
    landmarks[13] = (0.58, 0.65)
    landmarks[14] = (0.61, 0.50)
    landmarks[16] = (0.62, 0.60)
    landmarks[17] = (0.62, 0.68)
    landmarks[18] = (0.66, 0.56)
    landmarks[20] = (0.68, 0.64)

    assert count_extended_fingers(landmarks) == 2


def test_finger_counter_detects_one_extended_finger():
    landmarks = [(0.5, 0.9)] * 21
    landmarks[6] = (0.42, 0.48)
    landmarks[8] = (0.40, 0.20)
    landmarks[10] = (0.50, 0.40)
    landmarks[12] = (0.50, 0.62)
    landmarks[14] = (0.61, 0.50)
    landmarks[16] = (0.62, 0.60)
    landmarks[18] = (0.66, 0.56)
    landmarks[20] = (0.68, 0.64)

    assert count_extended_fingers(landmarks) == 1


def test_finger_counter_detects_three_extended_fingers():
    landmarks = [(0.5, 0.9)] * 21
    for tip, pip, x in ((8, 6, 0.40), (12, 10, 0.47), (16, 14, 0.54)):
        landmarks[pip] = (x, 0.60)
        landmarks[tip] = (x, 0.25)
    landmarks[18] = (0.62, 0.60)
    landmarks[20] = (0.62, 0.65)

    assert count_extended_fingers(landmarks) == 3


def make_finger_pose(finger_tips):
    landmarks = [(0.5, 0.9)] * 21
    finger_indices = ((8, 6), (12, 10), (16, 14), (20, 18))
    for (tip_index, pip_index), tip in zip(finger_indices, finger_tips):
        if tip is None:
            landmarks[pip_index] = (0.55, 0.70)
            landmarks[tip_index] = (0.52, 0.84)
        else:
            tip_x, tip_y = tip
            landmarks[tip_index] = tip
            landmarks[pip_index] = (
                0.5 + (tip_x - 0.5) * 0.60,
                0.9 + (tip_y - 0.9) * 0.60,
            )
    return landmarks


def test_two_separated_fingers_classify_as_circle():
    landmarks = make_finger_pose([
        (0.36, 0.20),
        (0.64, 0.20),
        None,
        None,
    ])

    assert classify_finger_pose(landmarks) == 'two_fingers_circle'


def test_two_together_fingers_classify_as_forward():
    landmarks = make_finger_pose([
        (0.48, 0.20),
        (0.52, 0.20),
        None,
        None,
    ])

    assert classify_finger_pose(landmarks) == 'two_fingers_forward'


def test_ambiguous_two_finger_gap_does_not_select_motion():
    landmarks = make_finger_pose([
        (0.44, 0.20),
        (0.56, 0.20),
        None,
        None,
    ])

    assert classify_finger_pose(landmarks) is None


def test_three_fingers_together_do_not_select_a_movement():
    landmarks = make_finger_pose([
        (0.47, 0.20),
        (0.50, 0.18),
        (0.53, 0.20),
        None,
    ])

    assert classify_finger_pose(landmarks) is None
