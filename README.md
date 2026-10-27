# Bài tập tuần 6: Điều khiển TurtleBot 4 bằng cử chỉ tay

Hệ thống dùng webcam, OpenCV và MediaPipe để nhận dạng cử chỉ tay. ROS 2
publish cử chỉ lên `/detected_gesture`; node điều khiển nhận cử chỉ, chờ xác
nhận bằng ngón cái rồi gửi vận tốc cho TurtleBot 4 qua `/cmd_vel`.

## Yêu cầu

- Ubuntu 22.04, ROS 2 Humble và TurtleBot 4 Gazebo simulation
- Python 3, OpenCV và MediaPipe Gesture Recognizer
- Webcam laptop hoặc webcam USB

Cài các phụ thuộc:

```bash
sudo apt install python3-opencv
python3 -m pip install --user 'mediapipe==0.10.21' 'numpy<2'
mkdir -p ~/.ros
curl -L \
  https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task \
  -o ~/.ros/gesture_recognizer.task
```

## Build

```bash
source /opt/ros/humble/setup.bash
cd ~/tb4_gesture_control
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select tb4_gesture_control
source install/setup.bash
```

## Chạy demo

Terminal 1 — khởi động TurtleBot 4 trong Gazebo, không mở RViz:

```bash
source /opt/ros/humble/setup.bash
source ~/tb4_gesture_control/install/setup.bash
ros2 launch turtlebot4_ignition_bringup turtlebot4_ignition.launch.py \
  world:=warehouse rviz:=false
```

Terminal 2 — đợi `/scan` và `/odom` hoạt động, sau đó chạy webcam và node
điều khiển:

```bash
source /opt/ros/humble/setup.bash
source ~/tb4_gesture_control/install/setup.bash
ros2 launch tb4_gesture_control gesture_control.launch.py
```

## Cử chỉ và hành vi

| Cử chỉ | Hành vi |
|---|---|
| Một ngón trỏ | Robot tiến/lùi 4 lần |
| Hai ngón tạo chữ V | Đi một vòng tròn |
| Hai ngón chụm | Tiến 1 m |
| Nắm tay | Lùi 1 m |
| Ngón cái | Xác nhận và chạy hành vi đã chọn |
| Xòe bàn tay | Dừng ngay lập tức |
| Vẫy tay | Robot xoay trái/phải chào lại |

Robot chỉ chạy hành vi đã chọn sau khi nhận cử chỉ ngón cái. Hãy đợi cửa sổ
webcam hiển thị trạng thái đã chọn trước khi xác nhận. Nắm tay hiện chọn lùi
1 m; ba ngón tay không gán hành vi.

Các chuyển động tiến/lùi dùng `/odom` để đo quãng đường. Kiểm tra vùng phía
sau robot trước khi thử lùi; lidar sẽ dừng robot nếu phát hiện vật cản gần.
Nếu có hành vi tự động khác đang chạy, `motion_control` có thể bỏ qua lệnh
`/cmd_vel`.

## ROS nodes và topics

- `gesture_recognition_node`: đọc webcam, nhận diện cử chỉ, hiển thị tên và
  confidence, landmarks; publish `std_msgs/msg/String` lên `/detected_gesture`
  theo dạng `gesture|confidence`.
- `robot_behavior_node`: subscribe `/detected_gesture`, `/scan`, `/odom`;
  publish `geometry_msgs/msg/Twist` lên `/cmd_vel` và trạng thái lên
  `/gesture_control/status`.

Theo dõi trong terminal khác:

```bash
source /opt/ros/humble/setup.bash
source ~/tb4_gesture_control/install/setup.bash
ros2 topic echo /detected_gesture
ros2 topic echo /gesture_control/status
ros2 topic echo /cmd_vel
```

Thay `camera_index` và `model_path` trong
`src/tb4_gesture_control/config/gesture_control.yaml` nếu cần.
