#!/usr/bin/env python3
"""
MuJoCo .act 파일을 재생하는 스크립트 (개선된 순서 보존 버전)
사용법: python3 run.py [model_xml_path] [act_file_path]
예시: python3 run.py dual_arm_robot.xml demo_manual.act

개선 사항:
1. 선형 보간을 통한 부드러운 재생
2. 재생 속도 조절 및 UI 개선
3. 재생 진행 상태 표시
4. 최신 MuJoCo API 지원
5. 관절 움직임 순서 보존 강화
"""

import os
import sys
import time
import numpy as np
import mujoco
from mujoco.glfw import glfw

def load_act_file(filename):
    """
    .act 파일을 로드하고 시간 및 제어 데이터를 반환합니다.
    """
    try:
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
            
            return time_data, ctrl_data, ctrl_size, num_frames
    except Exception as e:
        raise IOError(f"파일 로드 중 오류 발생: {e}")

def interpolate_ctrl_data(time_data, ctrl_data, current_time):
    """
    현재 시간에 대한 제어 데이터를 보간합니다.
    """
    # 시간이 범위를 벗어나면 처음이나 끝 프레임 사용
    if current_time <= time_data[0]:
        return ctrl_data[0, :]
    elif current_time >= time_data[-1]:
        return ctrl_data[-1, :]
    
    # 현재 시간이 속한 프레임 찾기
    idx = np.searchsorted(time_data, current_time) - 1
    next_idx = idx + 1
    
    if next_idx >= len(time_data):
        return ctrl_data[idx, :]
    
    # 두 프레임 사이의 비율 계산
    t_ratio = (current_time - time_data[idx]) / (time_data[next_idx] - time_data[idx])
    
    # 선형 보간
    interp_data = ctrl_data[idx, :] + t_ratio * (ctrl_data[next_idx, :] - ctrl_data[idx, :])
    
    return interp_data

def draw_progress_bar(context, viewport, progress, width=200, height=10, x_pos=50, y_pos=50):
    """
    진행 상태 표시줄을 그립니다.
    """
    try:
        # 현재 MuJoCo 버전 확인 - API가 다를 수 있음
        # 배경 (회색)
        bgcolor = [0.3, 0.3, 0.3, 1.0]
        try:
            # 최신 버전의 MuJoCo API (색상을 개별 인수로 사용)
            mujoco.mjr_rectangle(viewport, x_pos, y_pos, width, height, 
                               bgcolor[0], bgcolor[1], bgcolor[2], bgcolor[3])
        except TypeError:
            # 이전 버전의 MuJoCo API (색상을 리스트로 사용)
            mujoco.mjr_rectangle(viewport, x_pos, y_pos, width, height, bgcolor)
        
        # 진행 상태 (초록색)
        bar_width = int(width * progress)
        if bar_width > 0:
            fgcolor = [0.0, 1.0, 0.0, 1.0]
            try:
                # 최신 버전의 MuJoCo API
                mujoco.mjr_rectangle(viewport, x_pos, y_pos, bar_width, height, 
                                   fgcolor[0], fgcolor[1], fgcolor[2], fgcolor[3])
            except TypeError:
                # 이전 버전의 MuJoCo API
                mujoco.mjr_rectangle(viewport, x_pos, y_pos, bar_width, height, fgcolor)
    except Exception as e:
        # 진행 바 그리기에 실패해도 프로그램은 계속 실행
        print(f"진행 바 그리기 오류 (무시됨): {e}")

def main():
    # 명령줄 인수 처리
    if len(sys.argv) < 2:
        model_path = os.path.join(os.path.dirname(__file__), "dual_arm_robot.xml")
        print(f"모델 파일이 지정되지 않아 기본값 사용: {model_path}")
    else:
        model_path = sys.argv[1]
    
    if len(sys.argv) < 3:
        act_path = "demo_manual.act"
        print(f"액트 파일이 지정되지 않아 기본값 사용: {act_path}")
    else:
        act_path = sys.argv[2]
    
    # MuJoCo 모델 로드
    try:
        print(f"모델 로드 중: {model_path}")
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        print("모델 로드 성공")
    except Exception as e:
        print(f"오류: 모델 로드 실패 - {e}")
        sys.exit(1)
    
    # .act 파일 로드
    try:
        print(f".act 파일 로드 중: {act_path}")
        time_data, ctrl_data, ctrl_size, num_frames = load_act_file(act_path)
        total_duration = time_data[-1] - time_data[0]
        print(f"로드 완료: {num_frames}개 프레임, 제어 차원: {ctrl_size}, 총 시간: {total_duration:.2f}초")
    except Exception as e:
        print(f"오류: .act 파일 로드 실패 - {e}")
        sys.exit(1)
    
    # 제어 차원 확인
    if ctrl_size > model.nu:
        print(f"경고: .act 파일의 제어 차원({ctrl_size})이 모델의 액추에이터 수({model.nu})보다 큽니다. 초과 차원은 무시됩니다.")
    elif ctrl_size < model.nu:
        print(f"경고: .act 파일의 제어 차원({ctrl_size})이 모델의 액추에이터 수({model.nu})보다 작습니다. 부족한 차원은 0으로 설정됩니다.")
    
    # GLFW 초기화
    if not glfw.init():
        print("오류: GLFW 초기화 실패")
        sys.exit(1)
    
    # 윈도우 생성
    window = glfw.create_window(1200, 900, "MuJoCo 동작 재생", None, None)
    if not window:
        glfw.terminate()
        print("오류: 윈도우 생성 실패")
        sys.exit(1)
    
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
    
    # 전역 변수
    paused = True  # 처음에는 일시정지 상태로 시작
    reset_next_frame = True  # 시작 시 첫 프레임 위치로 설정
    playback_speed = 1.0
    sim_time = time_data[0]  # 첫 프레임 시간으로 초기화
    last_mouse_x, last_mouse_y = 0, 0
    show_info = True
    loop_playback = True
    
    # 로봇 관절 이름 (MuJoCo XML 파일에 정의된 순서대로)
    joint_names = [
        "왼쪽 관절1", "왼쪽 관절2", "왼쪽 관절3", "왼쪽 관절4", "왼쪽 그리퍼",
        "오른쪽 관절1", "오른쪽 관절2", "오른쪽 관절3", "오른쪽 관절4", "오른쪽 그리퍼"
    ]
    
    # 키보드 콜백
    def keyboard(window, key, scancode, action, mods):
        nonlocal paused, reset_next_frame, playback_speed, sim_time, show_info, loop_playback
        
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            elif key == glfw.KEY_SPACE:
                paused = not paused
                print(f"재생 {'일시정지' if paused else '재개'}")
            elif key == glfw.KEY_R:
                reset_next_frame = True
                print("처음부터 다시 재생")
            elif key == glfw.KEY_UP:
                playback_speed *= 1.5
                print(f"재생 속도 증가: {playback_speed:.2f}x")
            elif key == glfw.KEY_DOWN:
                playback_speed /= 1.5
                playback_speed = max(0.1, playback_speed)
                print(f"재생 속도 감소: {playback_speed:.2f}x")
            elif key == glfw.KEY_I:
                show_info = not show_info
                print(f"정보 표시 {'활성화' if show_info else '비활성화'}")
            elif key == glfw.KEY_L:
                loop_playback = not loop_playback
                print(f"반복 재생 {'활성화' if loop_playback else '비활성화'}")
            elif key == glfw.KEY_0:
                playback_speed = 1.0
                print("재생 속도 초기화: 1.0x")
    
    # 마우스 콜백 - 카메라 조작
    def mouse_button(window, button, act, mods):
        nonlocal last_mouse_x, last_mouse_y
        
        if button == glfw.MOUSE_BUTTON_LEFT:
            if act == glfw.PRESS:
                last_mouse_x, last_mouse_y = glfw.get_cursor_pos(window)
                glfw.set_cursor_pos_callback(window, mouse_move)
            else:
                glfw.set_cursor_pos_callback(window, None)
    
    def mouse_move(window, xpos, ypos):
        nonlocal last_mouse_x, last_mouse_y
        
        # 마우스 이동에 따라 카메라 회전
        dx = xpos - last_mouse_x
        dy = ypos - last_mouse_y
        
        # 방위각과 고도 조정 (천천히 회전하도록 계수 조정)
        cam.azimuth += dx * 0.3
        cam.elevation = max(min(cam.elevation - dy * 0.3, 90), -90)
        
        # 마우스 위치 업데이트
        last_mouse_x = xpos
        last_mouse_y = ypos
    
    # 콜백 설정
    glfw.set_key_callback(window, keyboard)
    glfw.set_mouse_button_callback(window, mouse_button)
    
    # 초기화 - 중요: 첫 프레임 위치로 설정
    initial_ctrl = ctrl_data[0, :]
    data.ctrl[:min(model.nu, ctrl_size)] = initial_ctrl[:min(model.nu, ctrl_size)]
    mujoco.mj_forward(model, data)  # 첫 프레임 상태로 시뮬레이션 초기화
    
    print("\n=== 재생 준비 완료 ===")
    print("Space: 재생/일시정지")
    print("R: 처음부터 다시 재생")
    print("UP/DOWN: 재생 속도 조절")
    print("I: 정보 표시 전환")
    print("L: 반복 재생 전환")
    print("0: 재생 속도 초기화")
    print("ESC: 종료")
    print("\n초기 상태로 설정되었습니다. 스페이스바를 눌러 재생을 시작하세요.")
    
    # 메인 루프
    last_update_time = time.time()
    first_frame = True
    
    try:
        while not glfw.window_should_close(window):
            current_time = time.time()
            dt = (current_time - last_update_time) * playback_speed
            last_update_time = current_time
            
            # 리셋 요청 처리
            if reset_next_frame:
                sim_time = time_data[0]
                # 첫 프레임 데이터 적용
                data.ctrl[:min(model.nu, ctrl_size)] = ctrl_data[0, :min(model.nu, ctrl_size)]
                mujoco.mj_forward(model, data)
                reset_next_frame = False
                first_frame = True
                paused = True  # 리셋 후 일시정지
                print("첫 프레임으로 리셋되었습니다. 스페이스바를 눌러 재생을 시작하세요.")
            
            # 시간 업데이트 (일시정지 상태가 아닐 때만)
            if not paused:
                if first_frame:
                    # 첫 프레임일 경우 시간을 증가시키지 않고 초기 위치만 설정
                    first_frame = False
                else:
                    sim_time += dt
                
                # 재생 완료 처리
                if sim_time > time_data[-1]:
                    if loop_playback:
                        print("재생 완료, 처음부터 다시 시작")
                        sim_time = time_data[0]
                        first_frame = True
                    else:
                        sim_time = time_data[-1]
                        paused = True
                        print("재생 완료, 일시정지됨")
                
                # 현재 시간에 맞는 제어 값 보간
                interp_ctrl = interpolate_ctrl_data(time_data, ctrl_data, sim_time)
                
                # 액추에이터에 제어 값 설정
                num_ctrl = min(model.nu, ctrl_data.shape[1])
                data.ctrl[:num_ctrl] = interp_ctrl[:num_ctrl]
            
            # 시뮬레이션 단계 실행
            mujoco.mj_step(model, data)
            
            # 렌더링
            viewport = mujoco.MjrRect(0, 0, *glfw.get_framebuffer_size(window))
            mujoco.mjv_updateScene(model, data, opt, None, cam, 
                                  mujoco.mjtCatBit.mjCAT_ALL.value, scene)
            mujoco.mjr_render(viewport, scene, context)
            
            # 정보 표시
            if show_info:
                # 진행 상태 계산
                progress = (sim_time - time_data[0]) / (time_data[-1] - time_data[0])
                progress = max(0, min(1, progress))  # 0~1 범위로 제한
                
                window_width = viewport.width
                window_height = viewport.height
                bar_width = int(window_width * 0.6)
                bar_height = 20
                bar_x = (window_width - bar_width) // 2
                bar_y = window_height - 40
                
                # 진행 상태 바 그리기 (오류 무시)
                try:
                    draw_progress_bar(context, viewport, progress, 
                                    width=bar_width, height=bar_height, 
                                    x_pos=bar_x, y_pos=bar_y)
                except Exception as e:
                    # 진행 바 그리기 오류 무시
                    pass
                
                # 오버레이 텍스트
                elapsed_time = sim_time - time_data[0]
                total_time = time_data[-1] - time_data[0]
                
                text = [
                    f"시간: {elapsed_time:.2f} / {total_time:.2f} 초 ({progress*100:.1f}%)",
                    f"재생 속도: {playback_speed:.2f}x",
                    f"상태: {'일시정지' if paused else '재생 중'}",
                    f"반복 재생: {'켜짐' if loop_playback else '꺼짐'}",
                    "",
                    "Space: 재생/일시정지",
                    "R: 처음부터 다시 재생",
                    "UP/DOWN: 재생 속도 조절",
                    "I: 정보 표시 전환",
                    "L: 반복 재생 전환",
                    "0: 재생 속도 초기화",
                    "ESC: 종료"
                ]
                
                overlay_text = "\n".join(text)
                mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, 
                                  mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                  viewport, overlay_text, "", context)
                
                # 현재 관절 값 표시 (하단에)
                joint_text = ""
                num_joints_to_show = min(len(joint_names), model.nu, ctrl_size)
                
                # 관절 정보 구성
                for i in range(num_joints_to_show):
                    name = joint_names[i] if i < len(joint_names) else f"관절 {i}"
                    joint_text += f"{name}: {data.ctrl[i]:.4f}  "
                    if i == 4:  # 5개마다 줄바꿈
                        joint_text += "\n"
                
                # 관절 정보 표시
                mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, 
                                  mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                                  viewport, joint_text, "", context)
            
            # 버퍼 교체 및 이벤트 처리
            glfw.swap_buffers(window)
            glfw.poll_events()
            
            # 프레임 속도 제어 (과도한 CPU 사용 방지)
            time.sleep(0.001)
            
    except Exception as e:
        print(f"실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 종료 처리
        glfw.terminate()
        print("프로그램 종료")

if __name__ == "__main__":
    main()