# CookingBot
KNU_캡스톤 디자인
omx 한 팔 로봇을 수동 조작 및 pick n place 작업 수행할 수 있음
한 팔 로봇이 publish 하는 joint states를 subscribe하여 rosbag(db3)로 저장 및 .act로 변환하여 Mujoco에서 다시 재현 --완료--

open manipulator x가 서로 마주보는 상태에서 바닥에 있는 물건을 grap 할 수 있고, 실시간으로 카메라의 화면을 송출하는 code까지 구현.
![Screenshot from 2025-05-06 22-36-06](https://github.com/user-attachments/assets/dbcace9d-5c38-4581-bba4-0ebe8190ce5d)

# 0510
1. 현재 양팔 로봇에서도 manual 실행 코드를 만듦.(left, right joint1 제외 2,3,4,gripper는 정상 작동, joint1은 민감도를 5배 올려도 미세하게 돌아가는 문제가 있음)
2. 코드 실행 상 joint states를 topic으로 publish 하도록 수정완료, rosbag으로 이를 subscribe하여 db3 형태로 저장
3. 저장된 db3 파일을 검증하기 위해 .act 형태로 변환하는 코드(db3_to_act.py) 작성 완료, 이를 다시 Mujoco에서 실행하는 코드(run_act.py) 작성 완료
@ 코드 실행까지 되지만, 구독된 토픽을 따라 정상적인 동작을 출력하지 못함.(원본과 다름)-렌더링 or 코드 문제로 추정 중

버그 모두 수정 후에 할 일
-> db3 파일을 .act 변환없이 바로 HDF5형태로 변환하기(ACT 학습을 위한)
