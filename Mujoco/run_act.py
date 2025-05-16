#!/usr/bin/env python3
"""
MuJoCo .act 파일을 재생하는 스크립트
"""

import os
import time
import numpy as np
import mujoco
from mujoco.glfw import glfw

# 설정 파일 경로
model_path = os.path.join(os.path.dirname(__file__), "dual_arm_robot.xml")
act_path = "demo_manual.act"  # 여기를 실제 .act 파일 이름으로 수정하세요

# MuJoCo 모델 로드
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# .act 파일 로드
def load_act_file(filename):
    with open(filename, 'rb') as f:
        # 헤더 읽기
        header = np.fromfile(f, dtype=np.int32, count=4)
        if header[0] != 1:
            raise ValueError(f"지원되지 않는 .act 파일 버전: {header[0]}")
        
        ctrl_size = header[2]  # 제어 차원
        num_frames = header[3]  # 프레임 수
        
        # 시간 및 제어 데이터 읽기
        time_data = np.fromfile(f, dtype=np.float64, count=num_frames)
        ctrl_data = np.fromfile(f, dtype=np.float64, count=num_frames * ctrl_size)
        ctrl_data = ctrl_data.reshape(num_frames, ctrl_size)
        
        return time_data, ctrl_data

# GLFW 초기화
if not glfw.init():
    raise RuntimeError("GLFW 초기화 실패")

# 윈도우 생성
window = glfw.create_window(1200, 900, "MuJoCo 재생", None, None)
if not window:
    glfw.terminate()
    raise RuntimeError("윈도우 생성 실패")

glfw.make_context_current(window)
glfw.swap_interval(1)

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

# 렌더링 함수
def render():
    viewport = mujoco.MjrRect(0, 0, *glfw.get_framebuffer_size(window))
    mujoco.mjv_updateScene(model, data, opt, None, cam, 
                          mujoco.mjtCatBit.mjCAT_ALL.value, scene)
    mujoco.mjr_render(viewport, scene, context)
    
    # 오버레이 텍스트 (재생 정보)
    text = [
        f"시간: {data.time:.2f}",
        "Space: 재생/일시정지",
        "R: 처음부터 다시 재생",
        "Esc: 종료"
    ]
    overlay_text = "\n".join(text)
    mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, 
                      mujoco.mjtGridPos.mjGRID_TOPLEFT,
                      viewport, overlay_text, "", context)
    
    glfw.swap_buffers(window)

# 키보드 콜백
paused = False
def keyboard(window, key, scancode, action, mods):
    global paused
    if action == glfw.PRESS:
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_SPACE:
            paused = not paused
        elif key == glfw.KEY_R:
            # 처음부터 다시 재생
            data.time = 0.0

glfw.set_key_callback(window, keyboard)

# 마우스 콜백 - 카메라 조작
def mouse_button(window, button, act, mods):
    if button == glfw.MOUSE_BUTTON_LEFT:
        if act == glfw.PRESS:
            # 왼쪽 버튼을 누르면 회전 모드
            glfw.set_cursor_pos_callback(window, mouse_move)
        else:
            # 버튼을 떼면 콜백 제거
            glfw.set_cursor_pos_callback(window, None)

def mouse_move(window, xpos, ypos):
    # 마우스 이동에 따라 카메라 회전
    dx = xpos - last_mouse_x
    dy = ypos - last_mouse_y
    
    # 방위각과 고도 조정
    cam.azimuth += dx * 0.3
    cam.elevation = max(min(cam.elevation - dy * 0.3, 90), -90)
    
    # 마우스 위치 업데이트
    last_mouse_x = xpos
    last_mouse_y = ypos

glfw.set_mouse_button_callback(window, mouse_button)
last_mouse_x, last_mouse_y = 0, 0

# .act 파일 로드
try:
    print(f".act 파일 로드 중: {act_path}")
    time_data, ctrl_data = load_act_file(act_path)
    print(f"로드 완료: {len(time_data)}개 프레임, 제어 차원: {ctrl_data.shape[1]}")
except Exception as e:
    print(f"오류: .act 파일 로드 실패 - {e}")
    glfw.terminate()
    raise

# 초기화
frame_idx = 0
last_update_time = time.time()
data.time = 0.0

print("재생 시작...")
print("Space: 재생/일시정지, R: 처음부터 다시 재생, Esc: 종료")

# 메인 루프
while not glfw.window_should_close(window):
    current_time = time.time()
    dt = current_time - last_update_time
    last_update_time = current_time
    
    if not paused:
        # 시간 업데이트
        data.time += dt
        
        # 현재 시간에 맞는 프레임 찾기
        while frame_idx < len(time_data) - 1 and data.time > time_data[frame_idx + 1]:
            frame_idx += 1
        
        if frame_idx < len(time_data):
            # 각 액추에이터에 제어 값 설정
            num_ctrl = min(model.nu, ctrl_data.shape[1])
            for i in range(num_ctrl):
                data.ctrl[i] = ctrl_data[frame_idx, i]
        else:
            # 재생 완료 시 처음부터 다시 재생
            data.time = 0.0
            frame_idx = 0
    
    # 시뮬레이션 단계 실행
    mujoco.mj_step(model, data)
    
    # 렌더링
    render()
    
    # 이벤트 
