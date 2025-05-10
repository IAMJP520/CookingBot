#!/usr/bin/env python3
"""
양팔 로봇 수동 조작 코드 - OpenManipulator 스타일
ROS 2 JointState 발행 기능 포함
키보드 입력 처리 완전 수정
"""

import numpy as np
import mujoco
import time
import os
from mujoco.glfw import glfw
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class RobotPublisher(Node):
    def __init__(self, model, data):
        super().__init__('robot_publisher')
        self.model = model
        self.data = data
        
        # 관절 이름 설정
        self.left_joint_names = [f"left_joint{i}" for i in range(1, 5)]
        self.left_joint_names.extend(["left_gripper_left_joint", "left_gripper_right_joint"])
        
        self.right_joint_names = [f"right_joint{i}" for i in range(1, 5)]
        self.right_joint_names.extend(["right_gripper_left_joint", "right_gripper_right_joint"])
        
        # 모든 관절 이름
        self.joint_names = self.left_joint_names + self.right_joint_names
        
        # JointState 메시지 발행자
        self.publisher = self.create_publisher(JointState, '/joint_states', 10)
        self.get_logger().info('Joint state publisher initialized')
    
    def publish_joint_states(self):
        # JointState 메시지 생성
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        
        # 관절 위치 및 속도 설정
        positions = []
        velocities = []
        
        # 왼쪽 및 오른쪽 관절 상태 수집
        for name_list in [self.left_joint_names, self.right_joint_names]:
            for name in name_list:
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
        
        # 메시지 발행
        self.publisher.publish(msg)

# 전역 변수
model = None
data = None
ros_node = None
selected_robot = "left"  # 왼쪽 또는 오른쪽 로봇
selected_joint = 0      # 관절 인덱스 (0=joint1, 1=joint2, ...)
control_step = 0.1      # 기본 제어 단위

# 액추에이터 인덱스
left_actuators = []
right_actuators = []

# 기본 위치
home_position = np.array([0.0, 0.0, 0.0, 0.0, 0.01])
gripper_open = -0.01
gripper_close = 0.019

# 보간 이동을 위한 제너레이터
interpolation = None

def initialize():
    """MuJoCo 및 ROS 2 초기화"""
    global model, data, ros_node, left_actuators, right_actuators
    
    # MuJoCo 모델 로드
    print("현재 작업 디렉토리:", os.getcwd())
    model_path = os.path.join(os.path.dirname(__file__), "dual_arm_robot.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    print("모델 로드 완료")
    
    # 액추에이터 설정
    try:
        left_actuators = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"left_actuator_joint{i}")
            for i in range(1, 5)
        ]
        left_actuators.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_actuator_gripper_joint"))
        
        right_actuators = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"right_actuator_joint{i}")
            for i in range(1, 5)
        ]
        right_actuators.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_actuator_gripper_joint"))
        
        print(f"왼쪽 액추에이터: {left_actuators}")
        print(f"오른쪽 액추에이터: {right_actuators}")
    except Exception as e:
        print(f"액추에이터 설정 오류: {e}")
        # 모든 액추에이터 출력
        for i in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            print(f"  액추에이터 {i}: {name}")
        raise
    
    # ROS 2 초기화
    rclpy.init()
    ros_node = RobotPublisher(model, data)
    
    # 초기 위치 설정
    reset_positions()
    
    return model, data

def reset_positions():
    """로봇을 초기 위치로 리셋"""
    global data, left_actuators, right_actuators, home_position
    
    # 왼쪽 로봇
    for i, idx in enumerate(left_actuators):
        if i < len(home_position):
            data.ctrl[idx] = home_position[i]
    
    # 오른쪽 로봇
    for i, idx in enumerate(right_actuators):
        if i < len(home_position):
            data.ctrl[idx] = home_position[i]
    
    print("로봇을 초기 위치로 리셋했습니다")

def interpolate_to_position(target_pos, actuators):
    """특정 위치로 부드럽게 이동"""
    global data
    steps = 100  # 보간 단계 수
    
    # 현재 위치 가져오기
    current_pos = np.array([data.ctrl[idx] for idx in actuators])
    
    # target_pos가 리스트나 배열이 아니면 단일 값을 위한 처리
    if np.isscalar(target_pos):
        target_pos = np.array([target_pos] * len(actuators))
    
    # 보간 단계
    for step in range(steps + 1):
        alpha = step / steps
        interp_pos = current_pos * (1 - alpha) + target_pos * alpha
        
        for i, idx in enumerate(actuators):
            data.ctrl[idx] = interp_pos[i]
            
        yield  # 제너레이터 일시 중지 (다음 프레임에 계속)

def adjust_joint(actuator_idx, amount):
    """관절 값을 조정"""
    global data, model
    
    # 현재 값 가져오기
    current_value = data.ctrl[actuator_idx]
    
    # 새로운 값 계산
    new_value = current_value + amount
    
    # 범위 제한
    if actuator_idx < model.nu:
        min_range = model.actuator_ctrlrange[actuator_idx, 0]
        max_range = model.actuator_ctrlrange[actuator_idx, 1]
        new_value = max(min(new_value, max_range), min_range)
    
    # 값 설정
    data.ctrl[actuator_idx] = new_value
    
    print(f"관절 {actuator_idx} 값 조정: {current_value:.4f} → {new_value:.4f} (변화량: {amount:.4f})")

def keyboard(window, key, scancode, action, mods):
    """키보드 콜백 함수"""
    global selected_robot, selected_joint, control_step, interpolation
    
    if action != glfw.PRESS and action != glfw.REPEAT:
        return
    
    # 선택된 로봇의 액추에이터 배열
    actuators = left_actuators if selected_robot == "left" else right_actuators
    
    # 로봇 전환 (TAB)
    if key == glfw.KEY_TAB:
        selected_robot = "right" if selected_robot == "left" else "left"
        print(f"선택된 로봇: {selected_robot.upper()}")
    
    # 관절 선택 (1-5 키)
    elif glfw.KEY_1 <= key <= glfw.KEY_5:
        selected_joint = key - glfw.KEY_1
        joint_name = "그리퍼" if selected_joint == 4 else f"관절 {selected_joint+1}"
        print(f"선택된 {selected_robot} 로봇 {joint_name}")
    
    # 민감도 조절
    elif key == glfw.KEY_MINUS:
        control_step = max(0.01, control_step - 0.01)
        print(f"제어 단위 감소: {control_step:.2f}")
    elif key == glfw.KEY_EQUAL:  # + 키 (대부분 키보드에서 =키와 같은 키)
        control_step = min(0.5, control_step + 0.01)
        print(f"제어 단위 증가: {control_step:.2f}")
    
    # 리셋
    elif key == glfw.KEY_R:
        reset_positions()
        interpolation = None
    
    # 홈 위치로 이동
    elif key == glfw.KEY_H:
        interpolation = interpolate_to_position(home_position, actuators)
    
    # 그리퍼 제어
    elif key == glfw.KEY_O:  # 열기
        actuator_idx = actuators[4]  # 그리퍼 액추에이터 인덱스
        data.ctrl[actuator_idx] = gripper_open
        print(f"{selected_robot} 그리퍼를 열었습니다")
    elif key == glfw.KEY_C:  # 닫기
        actuator_idx = actuators[4]  # 그리퍼 액추에이터 인덱스
        data.ctrl[actuator_idx] = gripper_close
        print(f"{selected_robot} 그리퍼를 닫았습니다")
    
    # 관절 조정
    elif key == glfw.KEY_UP or key == glfw.KEY_DOWN:
        if 0 <= selected_joint < len(actuators):
            actuator_idx = actuators[selected_joint]
            direction = 1 if key == glfw.KEY_UP else -1
            
            # joint1(첫 번째 관절)은 Z축 회전이므로 더 큰 값 사용
            boost = 5.0 if selected_joint == 0 else 1.0
            
            # 조정값 계산
            adjustment = control_step * direction * boost
            
            # 관절 조정
            adjust_joint(actuator_idx, adjustment)
    
    # 종료
    elif key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)

def mouse_button(window, button, act, mods):
    """마우스 버튼 콜백"""
    global button_left, button_middle, button_right, lastx, lasty
    
    if button == glfw.MOUSE_BUTTON_LEFT:
        button_left = (act == glfw.PRESS)
    elif button == glfw.MOUSE_BUTTON_MIDDLE:
        button_middle = (act == glfw.PRESS)
    elif button == glfw.MOUSE_BUTTON_RIGHT:
        button_right = (act == glfw.PRESS)
    
    lastx, lasty = glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    """마우스 이동 콜백"""
    global lastx, lasty, cam, button_left, button_middle, button_right
    
    if not (button_left or button_middle or button_right):
        return
    
    dx = xpos - lastx
    dy = ypos - lasty
    
    if button_left:
        cam.azimuth = (cam.azimuth + 0.3 * dx) % 360
        cam.elevation = np.clip(cam.elevation - 0.3 * dy, -90, 90)
    elif button_middle:
        cam.lookat[0] += -0.001 * dx * cam.distance
        cam.lookat[1] += 0.001 * dy * cam.distance
    elif button_right:
        cam.distance = np.clip(cam.distance + 0.01 * dy, 0.1, 5.0)
    
    lastx, lasty = xpos, ypos

def scroll(window, xoffset, yoffset):
    """스크롤 콜백"""
    global cam
    cam.distance = np.clip(cam.distance - 0.1 * yoffset, 0.1, 5.0)

def main():
    """메인 함수"""
    global model, data, ros_node, interpolation
    global button_left, button_middle, button_right, lastx, lasty, cam
    
    # 초기화
    model, data = initialize()
    
    # GLFW 초기화
    if not glfw.init():
        raise RuntimeError("GLFW 초기화 실패")
    
    # 윈도우 생성
    window = glfw.create_window(1200, 900, "양팔 로봇 제어", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("윈도우 생성 실패")
    
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    
    # 콜백 등록
    glfw.set_key_callback(window, keyboard)
    
    # 마우스 변수 초기화
    button_left = button_middle = button_right = False
    lastx = lasty = 0
    
    # 마우스 콜백 등록
    glfw.set_mouse_button_callback(window, mouse_button)
    glfw.set_cursor_pos_callback(window, mouse_move)
    glfw.set_scroll_callback(window, scroll)
    
    # 시각화 설정
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    
    # 카메라 초기 설정
    cam.distance = 1.2
    cam.elevation = -20.0
    cam.azimuth = 90.0
    cam.lookat = np.array([0.0, 0.0, 0.3])
    
    # 메인 루프
    print("=== 양팔 로봇 제어 시작 ===")
    print("TAB: 로봇 전환 (왼쪽/오른쪽)")
    print("1-5: 관절 선택")
    print("↑/↓: 관절 조정")
    print("-/=: 제어 단위 조절")
    print("H: 홈 위치")
    print("O: 그리퍼 열기, C: 그리퍼 닫기")
    print("R: 리셋, ESC: 종료")
    
    try:
        while not glfw.window_should_close(window):
            # ROS 2 콜백 실행
            rclpy.spin_once(ros_node, timeout_sec=0)
            
            # 보간 이동 진행
            if interpolation is not None:
                try:
                    next(interpolation)
                except StopIteration:
                    interpolation = None
            
            # MuJoCo 시뮬레이션 단계 실행
            mujoco.mj_step(model, data)
            
            # JointState 메시지 발행
            ros_node.publish_joint_states()
            
            # 렌더링
            viewport_width, viewport_height = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)
            
            mujoco.mjv_updateScene(model, data, opt, None, cam,
                                mujoco.mjtCatBit.mjCAT_ALL.value, scene)
            mujoco.mjr_render(viewport, scene, context)
            
            # 오버레이 정보 표시
            text = [
                f"선택된 로봇: {selected_robot.upper()}",
                f"선택된 관절: {'그리퍼' if selected_joint == 4 else str(selected_joint+1)}",
                f"제어 단위: {control_step:.2f}",
                "",
                "TAB: 로봇 전환",
                "1-5: 관절 선택",
                "↑/↓: 관절 조정",
                "-/=: 제어 단위 조절",
                "H: 홈 위치",
                "O: 그리퍼 열기",
                "C: 그리퍼 닫기",
                "R: 리셋",
                "ESC: 종료"
            ]
            
            overlay_text = "\n".join(text)
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,
                             mujoco.mjtGridPos.mjGRID_TOPLEFT,
                             viewport, overlay_text, "", context)
            
            # 화면 업데이트
            glfw.swap_buffers(window)
            
            # 이벤트 처리
            glfw.poll_events()
            
            # 프레임 속도 제어
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("사용자에 의해 프로그램이 중단되었습니다.")
    finally:
        # 종료 처리
        ros_node.destroy_node()
        rclpy.shutdown()
        glfw.terminate()
        print("프로그램이 종료되었습니다.")

if __name__ == "__main__":
    main()
