# CookingBot
KNU_캡스톤 디자인
open manipulator x가 서로 마주보는 상태에서 바닥에 있는 물건을 grap 할 수 있고, 실시간으로 카메라의 화면을 송출하는 code까지 구현.
![Screenshot from 2025-05-06 22-36-06](https://github.com/user-attachments/assets/dbcace9d-5c38-4581-bba4-0ebe8190ce5d)

무주코(수동조작) open → rosbag 기록 시작 → 녹화 중지 및 저장 → .act 파일로 변환 → Mujoco에서 재생 방법 procedure

1. ROS 2 환경 설정 & 퍼블리셔 실행(터미널 A)

source /opt/ros/humble/setup.bash

python3 manipulate_with_ros.py

(실행 안되면 경로 CookingBot/ Mujoco로 수정)

2. rosbag 기록(터미널 B)

source /opt/ros/humble/setup.bash

ros2 bag record -o demo_# /joint_states

(demo_#에 숫자 집어넣기)

3. home 에 생성된 demo_# 뒤에 _0 붙여서 db3 파일과 형태 맞추고 CookingBot/ Mujoco로 이동

4. .act 파일 형태로 변환.

source /opt/ros/humble/setup.bash
python3 db3_to_act_ros2.py demo_#_0 /joint_states demo_#.act

5. vscode의 run_act.py 에서 act_path를 demo_#으로 맞춰주고 실행하면 끝.

cd ~/CookingBot/Mujoco
python3 run_act.py
