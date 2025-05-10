#!/usr/bin/env python3
"""
양팔 로봇 수동 조작 코드 - ROS 2 JointState 발행 기능 포함
키보드로 로봇 관절 개별 제어 가능
"""

import numpy as np
import mujoco
import time
import os
from mujoco.glfw import glfw
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class RobotJointPublisher(Node):
    def __init__(self, model, data):
        super().__init__('robot_joint_publisher')
        self.model = model
        self.data = data
        
        # 관절 이름 설정 (XML 파일 기준)
        self.left_joint_names = [f"left_joint{i}" for i in range(1, 5)]
        self.left_joint_names.extend(["left_gripper_left_joint", "left_gripper_right_joint"])
        
        self.right_joint_names = [f"right_joint{i}" for i in range(1, 5)]
        self.right_joint_names.extend(["right_gripper_left_joint", "right_gripper_right_joint"])
        
        # 모든 관절 이름
        self.joint_names = self.left_joint_names + self.right_joint_names
        
        # JointState 메시지 발행자
        self.publisher = self.create_publisher(JointState, 'joint_states', 10)
        
        # 타이머 설정 (100Hz)
        self.timer = self.create_timer(0.01, self.publish_joint_states)
        
        self.get_logger().info('Joint state publisher initialized')
    
    def publish_joint_states(self):
        try:
            # JointState 메시지 생성
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = self.joint_names
            
            # 관절 위치 설정
            positions = []
            velocities = []
            
            # 왼쪽 관절
            for name in self.left_joint_names:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if joint_id >= 0:
                    qpos_addr = self.model.jnt_qposadr[joint_id]
                    qvel_addr = self.model.jnt_dofadr[joint_id]
                    positions.append(float(self.data.qpos[qpos_addr]))
                    velocities.append(float(self.data.qvel[qvel_addr]))
                else:
                    positions.append(0.0)
                    velocities.append(0.0)
            
            # 오른쪽 관절
            for name in self.right_joint_names:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if joint_id >= 0:
                    qpos_addr = self.model.jnt_qposadr[joint_id]
                    qvel_addr = self.model.jnt_dofadr[joint_id]
                    positions.append(float(self.data.qpos[qpos_addr]))
                    velocities.append(float(self.data.qvel[qvel_addr]))
                else:
                    positions.append(0.0)
                    velocities.append(0.0)
            
            msg.position = positions
            msg.velocity = velocities
            msg.effort = [0.0] * len(self.joint_names)
            
            self.publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f'Error publishing joint states: {e}')

# 전역 변수
paused = False
selected_actuator = 0  # 현재 선택된 액추에이터 인덱스
step_size = 0.05       # 관절 조작 단위
left_joints = []       # 왼쪽 액추에이터 인덱스
right_joints = []      # 오른쪽 액추에이터 인덱스
model = None           # MuJoCo 모델 - 전역으로 참조 가능하게 설정
data = None            # MuJoCo 데이터 - 전역으로 참조 가능하게 설정

# 초기화 및 메인 함수
def main():
    global left_joints, right_joints, model, data
    
    # MuJoCo 모델 로드
    print("Current working directory:", os.getcwd())
    model_path = os.path.join(os.path.dirname(__file__), "dual_arm_robot.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    print("Model loaded successfully")
    
    # ROS 2 초기화
    rclpy.init()
    ros_node = RobotJointPublisher(model, data)
    
    # 액추에이터 인덱스 설정
    try:
        left_joints = [model.actuator(f"left_actuator_joint{i}").id for i in range(1,5)]
        left_joints.append(model.actuator("left_actuator_gripper_joint").id)
        right_joints = [model.actuator(f"right_actuator_joint{i}").id for i in range(1,5)]
        right_joints.append(model.actuator("right_actuator_gripper_joint").id)
        print("Actuators loaded successfully")
    except Exception as e:
        print(f"Error accessing actuators: {e}")
        # 액추에이터 이름 출력
        print("Available actuator names:")
        for i in range(model.nu):
            print(f"  Actuator {i}: {model.actuator(i).name}")
        raise
    
    # GLFW 초기화
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")
    
    # 윈도우 생성
    window = glfw.create_window(1200, 900, "Manual Robot Control", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create window")
    
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    
    # 키보드 콜백 설정
    glfw.set_key_callback(window, keyboard)
    
    # 카메라 및 장면 설정
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    
    # 카메라 초기 설정
    cam.distance = 1.2
    cam.elevation = -20.0
    cam.azimuth = 0.0
    cam.lookat = np.array([0.0, 0.0, 0.3])
    
    # 초기 위치 설정
    reset_position()
    
    # 메인 루프
    print("=== Manual Robot Control Started ===")
    print("ROS 2 topic publishing: /joint_states")
    print("\nKeyboard Controls:")
    print("- 1-5: Select left arm joints")
    print("- 6-0: Select right arm joints")
    print("- UP/DOWN: Increase/decrease joint value")
    print("- R: Reset position")
    print("- P: Toggle pause")
    print("- ESC: Exit")
    
    # 마지막 업데이트 시간
    last_update_time = time.time()
    
    try:
        while not glfw.window_should_close(window):
            # 시간 업데이트
            current_time = time.time()
            dt = current_time - last_update_time
            last_update_time = current_time
            
            # ROS 2 콜백 실행
            rclpy.spin_once(ros_node, timeout_sec=0)
            
            if not paused:
                # MuJoCo 시뮬레이션 스텝
                mujoco.mj_step(model, data)
            
            # 렌더링
            viewport = mujoco.MjrRect(0, 0, *glfw.get_framebuffer_size(window))
            mujoco.mjv_updateScene(model, data, opt, None, cam,
                                mujoco.mjtCatBit.mjCAT_ALL.value, scene)
            mujoco.mjr_render(viewport, scene, context)
            
            # 오버레이 텍스트 (ASCII로 제한)
            all_actuators = left_joints + right_joints
            text = [
                "Manual Robot Control",
                f"Selected joint: {'Left' if selected_actuator < len(left_joints) else 'Right'} {'Gripper' if selected_actuator % 5 == 4 else 'Joint ' + str((selected_actuator % 5) + 1)}",
                f"Current value: {data.ctrl[all_actuators[selected_actuator]]:.4f}",
                "",
                "Controls:",
                "1-5: Select left arm joints",
                "6-0: Select right arm joints",
                "UP/DOWN: Adjust joint value",
                "R: Reset position",
                "P: Toggle pause",
                "ESC: Exit"
            ]
            overlay_text = "\n".join(text)
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,
                             mujoco.mjtGridPos.mjGRID_TOPLEFT,
                             viewport, overlay_text, "", context)
            
            glfw.swap_buffers(window)
            
            # 이벤트 처리
            glfw.poll_events()
            
            # 프레임 속도 제어
            time.sleep(0.01)
    finally:
        # 종료 처리
        ros_node.destroy_node()
        rclpy.shutdown()
        glfw.terminate()
        print("Simulation terminated")

# 키보드 콜백 함수
def keyboard(window, key, scancode, action, mods):
    global paused, selected_actuator
    
    # 키가 눌렸을 때만 처리
    if action == glfw.PRESS or action == glfw.REPEAT:
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        
        # 관절 선택 (1-5: 왼쪽, 6-0: 오른쪽)
        elif key == glfw.KEY_1:
            selected_actuator = 0  # 왼쪽 관절 1
        elif key == glfw.KEY_2:
            selected_actuator = 1  # 왼쪽 관절 2
        elif key == glfw.KEY_3:
            selected_actuator = 2  # 왼쪽 관절 3
        elif key == glfw.KEY_4:
            selected_actuator = 3  # 왼쪽 관절 4
        elif key == glfw.KEY_5:
            selected_actuator = 4  # 왼쪽 그리퍼
        elif key == glfw.KEY_6:
            selected_actuator = 5  # 오른쪽 관절 1
        elif key == glfw.KEY_7:
            selected_actuator = 6  # 오른쪽 관절 2
        elif key == glfw.KEY_8:
            selected_actuator = 7  # 오른쪽 관절 3
        elif key == glfw.KEY_9:
            selected_actuator = 8  # 오른쪽 관절 4
        elif key == glfw.KEY_0:
            selected_actuator = 9  # 오른쪽 그리퍼
        
        # 관절 값 조정
        elif key == glfw.KEY_UP:
            adjust_joint(1)
        elif key == glfw.KEY_DOWN:
            adjust_joint(-1)
        
        # 시뮬레이션 리셋
        elif key == glfw.KEY_R:
            reset_position()
        
        # 일시정지
        elif key == glfw.KEY_P:
            paused = not paused

# 관절 값 조정 함수
def adjust_joint(direction):
    global selected_actuator, data, model
    
    # 전역 변수 확인
    if data is None or model is None:
        print("Error: MuJoCo model or data not initialized")
        return
        
    # 선택된 액추에이터의 인덱스 계산
    all_actuators = left_joints + right_joints
    if selected_actuator < len(all_actuators):
        actuator_index = all_actuators[selected_actuator]
        
        # 그리퍼인 경우 더 작은 조정 단위 사용
        adjust_value = step_size * direction
        if selected_actuator % 5 == 4:  # 그리퍼 인덱스
            adjust_value = adjust_value * 0.2
        
        # 값 조정
        data.ctrl[actuator_index] += adjust_value
        
        # 관절 범위 제한
        if actuator_index < model.nu:
            range_min = model.actuator_ctrlrange[actuator_index, 0]
            range_max = model.actuator_ctrlrange[actuator_index, 1]
            data.ctrl[actuator_index] = max(min(data.ctrl[actuator_index], range_max), range_min)

# 초기 위치 설정
def reset_position():
    global data
    
    # 전역 변수 확인
    if data is None:
        print("Error: MuJoCo data not initialized")
        return
        
    # 왼쪽 관절 초기화
    for idx, angle in zip(left_joints, [0.0, -0.3, 0.6, 0.0, 0.01]):
        data.ctrl[idx] = angle
    
    # 오른쪽 관절 초기화
    for idx, angle in zip(right_joints, [0.0, -0.3, 0.6, 0.0, 0.01]):
        data.ctrl[idx] = angle

# 메인 함수 실행
if __name__ == "__main__":
    main()