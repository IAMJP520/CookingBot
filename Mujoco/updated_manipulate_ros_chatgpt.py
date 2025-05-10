#!/usr/bin/env python3
"""
양팔 로봇 수동 조작 코드 - ROS 2 JointState 발행 기능 포함
키보드로 로봇 관절 개별 제어 가능
joint1 증폭 로직 추가 및 키 입력 조정값 증가
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
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = self.joint_names
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
model = None           # MuJoCo 모델
data = None            # MuJoCo 데이터

# 디버그 정보 출력 함수
def print_debug_info():
    print("\n--- 디버그 정보 ---")
    print(f"모델 관절 수: {model.njnt}")
    print(f"모델 액추에이터 수: {model.nu}")
    print("\n관절 정보:")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            joint_type = model.jnt_type[i]
            joint_type_name = ["FREE", "BALL", "SLIDE", "HINGE"][joint_type]
            joint_axis = model.jnt_axis[i]
            joint_qpos = data.qpos[model.jnt_qposadr[i]]
            joint_range = [model.jnt_range[i, 0], model.jnt_range[i, 1]]
            print(f"  Joint {i}: {name}, 타입: {joint_type_name}, 축: {joint_axis}, 위치: {joint_qpos:.4f}, 범위: {joint_range}")
    print("\n액추에이터 정보:")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        joint_id = model.actuator_trnid[i, 0]
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) if joint_id >= 0 else "unknown"
        ctrl_range = [model.actuator_ctrlrange[i, 0], model.actuator_ctrlrange[i, 1]]
        print(f"  Actuator {i}: {name}, 제어 관절: {joint_name}, 제어 값: {data.ctrl[i]:.4f}, 범위: {ctrl_range}")
    print("--- 디버그 정보 끝 ---")

# 초기화 및 메인 함수
def main():
    global left_joints, right_joints, model, data
    try:
        model_path = os.path.join(os.path.dirname(__file__), "dual_arm_robot.xml")
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        rclpy.init()
        ros_node = RobotJointPublisher(model, data)
        # 액추에이터 인덱스 설정
        left_actuator_names = [
            "left_actuator_joint1", "left_actuator_joint2",
            "left_actuator_joint3", "left_actuator_joint4",
            "left_actuator_gripper_joint"
        ]
        right_actuator_names = [
            "right_actuator_joint1", "right_actuator_joint2",
            "right_actuator_joint3", "right_actuator_joint4",
            "right_actuator_gripper_joint"
        ]
        left_joints = []
        for name in left_actuator_names:
            act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if act_id >= 0:
                left_joints.append(act_id)
            else:
                print(f"Warning: Actuator {name} not found")
        right_joints = []
        for name in right_actuator_names:
            act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if act_id >= 0:
                right_joints.append(act_id)
            else:
                print(f"Warning: Actuator {name} not found")
        print(f"Left actuator IDs: {left_joints}")
        print(f"Right actuator IDs: {right_joints}")
        print_debug_info()
    except Exception as e:
        print(f"Error accessing actuators: {e}")
        print("Available actuator names:")
        for i in range(model.nu):
            print(f"  Actuator {i}: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)}")
        raise
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")
    window = glfw.create_window(1200, 900, "Manual Robot Control", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create window")
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    glfw.set_key_callback(window, keyboard)
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    cam.distance = 1.2
    cam.elevation = -20.0
    cam.azimuth = 0.0
    cam.lookat = np.array([0.0, 0.0, 0.3])
    reset_position()
    print("=== Manual Robot Control Started ===")
    print("ROS 2 topic publishing: /joint_states")
    print("\nKeyboard Controls:")
    print("- 1-5: Select left arm joints")
    print("- 6-0: Select right arm joints")
    print("- UP/DOWN: Increase/decrease joint value")
    print("- LEFT/RIGHT: Fine adjustment")
    print("- R: Reset position")
    print("- P: Toggle pause")
    print("- D: Print debug info")
    print("- ESC: Exit")
    last_update_time = time.time()
    try:
        while not glfw.window_should_close(window):
            current_time = time.time()
            dt = current_time - last_update_time
            last_update_time = current_time
            rclpy.spin_once(ros_node, timeout_sec=0)
            if not paused:
                mujoco.mj_step(model, data)
            viewport = mujoco.MjrRect(0, 0, *glfw.get_framebuffer_size(window))
            mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene)
            mujoco.mjr_render(viewport, scene, context)
            all_actuators = left_joints + right_joints
            if selected_actuator < len(all_actuators):
                actuator_index = all_actuators[selected_actuator]
                if selected_actuator < len(left_joints):
                    if selected_actuator == len(left_joints) - 1:
                        joint_name = "Left Gripper"
                    else:
                        joint_name = f"Left Joint {selected_actuator + 1}"
                else:
                    right_idx = selected_actuator - len(left_joints)
                    if right_idx == len(right_joints) - 1:
                        joint_name = "Right Gripper"
                    else:
                        joint_name = f"Right Joint {right_idx + 1}"
                text = [
                    "Manual Robot Control",
                    f"Selected: {joint_name}",
                    f"Value: {data.ctrl[actuator_index]:.4f}",
                    "",
                    "Controls:",
                    "1-5: Select left arm joints",
                    "6-0: Select right arm joints",
                    "UP/DOWN: Large adjustment",
                    "LEFT/RIGHT: Fine adjustment",
                    "R: Reset position",
                    "P: Toggle pause",
                    "D: Debug info",
                    "ESC: Exit"
                ]
                overlay_text = "\n".join(text)
                mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport, overlay_text, "", context)
            glfw.swap_buffers(window)
            glfw.poll_events()
            time.sleep(0.01)
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()
        glfw.terminate()
        print("Simulation terminated")

# 키보드 콜백 함수
def keyboard(window, key, scancode, action, mods):
    global paused, selected_actuator
    if action == glfw.PRESS or action == glfw.REPEAT:
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_1:
            selected_actuator = 0
            print("Left Joint 1 selected (Z-axis rotation)")
        elif key == glfw.KEY_2:
            selected_actuator = 1
            print("Left Joint 2 selected")
        elif key == glfw.KEY_3:
            selected_actuator = 2
            print("Left Joint 3 selected")
        elif key == glfw.KEY_4:
            selected_actuator = 3
            print("Left Joint 4 selected")
        elif key == glfw.KEY_5:
            selected_actuator = 4
            print("Left Gripper selected")
        elif key == glfw.KEY_6:
            selected_actuator = 5
            print("Right Joint 1 selected (Z-axis rotation)")
        elif key == glfw.KEY_7:
            selected_actuator = 6
            print("Right Joint 2 selected")
        elif key == glfw.KEY_8:
            selected_actuator = 7
            print("Right Joint 3 selected")
        elif key == glfw.KEY_9:
            selected_actuator = 8
            print("Right Joint 4 selected")
        elif key == glfw.KEY_0:
            selected_actuator = 9
            print("Right Gripper selected")
        # 관절 값 조정 - 큰 조정
        elif key == glfw.KEY_UP:
            adjust_joint(0.5)
        elif key == glfw.KEY_DOWN:
            adjust_joint(-0.5)
        # 관절 값 조정 - 미세 조정
        elif key == glfw.KEY_RIGHT:
            adjust_joint(0.10)
        elif key == glfw.KEY_LEFT:
            adjust_joint(-0.10)
        elif key == glfw.KEY_R:
            reset_position()
            print("Position reset")
        elif key == glfw.KEY_P:
            paused = not paused
            print(f"Simulation {'paused' if paused else 'resumed'}")
        elif key == glfw.KEY_D:
            print_debug_info()

# 관절 값 조정 함수
def adjust_joint(adjustment):
    global selected_actuator, data, model
    if data is None or model is None:
        print("Error: MuJoCo model or data not initialized")
        return
    # 선택된 액추에이터의 인덱스 계산
    all_actuators = left_joints + right_joints
    if selected_actuator < len(all_actuators):
        actuator_index = all_actuators[selected_actuator]
        # joint1 (Z축 회전) 증폭
        joint_id = model.actuator_trnid[actuator_index, 0]
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name in ["left_joint1", "right_joint1"]:
            adjustment *= 2.0
        # 그리퍼인 경우 더 작은 조정 단위 사용
        if selected_actuator in [4, 9]:
            adjustment *= 0.1
        old_value = data.ctrl[actuator_index]
        data.ctrl[actuator_index] += adjustment
        # 관절 범위 제한
        if actuator_index < model.nu:
            rmin, rmax = model.actuator_ctrlrange[actuator_index]
            data.ctrl[actuator_index] = max(min(data.ctrl[actuator_index], rmax), rmin)
        new_value = data.ctrl[actuator_index]
        print(f"Joint value changed: {old_value:.4f} -> {new_value:.4f} (adjustment: {adjustment:.4f})")
        # 제어 관절 이름 출력
        joint_id = model.actuator_trnid[actuator_index, 0]
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) if joint_id >= 0 else "unknown"
        print(f"Controlling joint: {joint_name}")
        if joint_name in ["left_joint1", "right_joint1"]:
            qpos_addr = model.jnt_qposadr[joint_id]
            print(f"  Joint position (qpos): {data.qpos[qpos_addr]:.4f}")

# 초기 위치 설정
def reset_position():
    global data, left_joints, right_joints
    if data is None:
        print("Error: MuJoCo data not initialized")
        return
    init_angles = [0.0, 0.0, 0.0, 0.0]
    for i, idx in enumerate(left_joints):
        if i < len(init_angles):
            data.ctrl[idx] = init_angles[i]
    for i, idx in enumerate(right_joints):
        if i < len(init_angles):
            data.ctrl[idx] = init_angles[i]
    print("Robot reset to initial position")

if __name__ == "__main__":
    main()
