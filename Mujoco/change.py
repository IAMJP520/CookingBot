#!/usr/bin/env python3
"""
rosbag2 db3 파일을 MuJoCo .act 파일로 변환 (개선된 버전)
사용법: python3 db3_to_act.py [bag_directory] [topic_name] [output_act_file]
예시: python3 db3_to_act.py demo_1_0 /joint_states demo_1.act

개선 사항:
1. 데이터 전처리 및 필터링 추가
2. 일정한 샘플링 레이트로 리샘플링
3. 데이터 보간 기능 추가
4. 디버깅 및 진단 정보 출력
"""

import sys
import numpy as np
import os
import sqlite3
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import array
import matplotlib.pyplot as plt
from scipy import interpolate

def main():
    if len(sys.argv) != 4:
        print("사용법: python3 db3_to_act.py [bag_directory] [topic_name] [output_act_file]")
        print("예시: python3 db3_to_act.py demo_1_0 /joint_states demo_1.act")
        sys.exit(1)
    
    bag_directory = sys.argv[1]
    topic_name = sys.argv[2]
    output_file = sys.argv[3]
    
    # 추가 옵션
    debug_mode = True  # 디버그 정보 출력 여부
    resample_rate = 100  # Hz 단위의 리샘플링 레이트
    smooth_data = True  # 데이터 스무딩 여부
    
    # DB 파일 경로 찾기
    db_file = None
    for file in os.listdir(bag_directory):
        if file.endswith('.db3'):
            db_file = os.path.join(bag_directory, file)
            break
    
    if db_file is None:
        print(f"오류: {bag_directory} 디렉토리에 .db3 파일을 찾을 수 없습니다.")
        sys.exit(1)
    
    print(f"DB 파일: {db_file}")
    print(f"토픽: {topic_name}")
    print(f"출력 파일: {output_file}")
    print(f"리샘플링 레이트: {resample_rate} Hz")
    
    # DB 연결
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # 메시지 타입 확인
    cursor.execute("SELECT id, name, type FROM topics WHERE name = ?", (topic_name,))
    topic_row = cursor.fetchone()
    
    if topic_row is None:
        print(f"오류: 토픽 {topic_name}을 찾을 수 없습니다.")
        cursor.execute("SELECT name FROM topics")
        topics = cursor.fetchall()
        print("사용 가능한 토픽:")
        for topic in topics:
            print(f"  {topic[0]}")
        conn.close()
        sys.exit(1)
    
    topic_id, _, topic_type = topic_row
    msg_type = get_message(topic_type)
    
    # 메시지 데이터 쿼리
    cursor.execute("""
        SELECT data, timestamp 
        FROM messages 
        WHERE topic_id = ? 
        ORDER BY timestamp
    """, (topic_id,))
    
    rows = cursor.fetchall()
    
    if not rows:
        print(f"오류: 토픽 {topic_name}에 메시지가 없습니다.")
        conn.close()
        sys.exit(1)
    
    print(f"{len(rows)}개의 메시지 로드 중...")
    
    # 메시지 디시리얼라이즈
    ctrl_data = []
    timestamps = []
    joint_names = []
    
    for row in rows:
        data, timestamp = row
        msg = deserialize_message(data, msg_type)
        
        # 첫 번째 메시지에서 관절 이름 추출
        if not joint_names and hasattr(msg, 'name') and msg.name:
            joint_names = msg.name
            print(f"관절 이름: {joint_names}")
        
        # JointState 메시지에서 position 데이터 추출
        if hasattr(msg, 'position') and msg.position:
            ctrl_data.append(list(msg.position))
            timestamps.append(timestamp)
    
    if not ctrl_data:
        print("오류: 메시지에서 position 데이터를 찾을 수 없습니다.")
        conn.close()
        sys.exit(1)
    
    # 데이터 형식 변환
    time_data = np.array(timestamps, dtype=np.int64)
    time_data = (time_data - time_data[0]) / 1e9  # 나노초를 초로 변환
    ctrl_data = np.array(ctrl_data, dtype=np.float64)
    
    print(f"원본 시간 데이터 범위: {time_data[0]:.3f}s - {time_data[-1]:.3f}s (총 {time_data[-1]:.3f}초)")
    print(f"원본 데이터 형태: 시간({time_data.shape}), 제어({ctrl_data.shape})")
    
    # 샘플 속도 확인
    if len(time_data) > 1:
        avg_dt = np.mean(np.diff(time_data))
        avg_rate = 1.0 / avg_dt if avg_dt > 0 else 0
        min_dt = np.min(np.diff(time_data))
        max_dt = np.max(np.diff(time_data))
        print(f"원본 샘플링: 평균 {avg_rate:.1f}Hz (dt: 평균 {avg_dt*1000:.1f}ms, 최소 {min_dt*1000:.1f}ms, 최대 {max_dt*1000:.1f}ms)")
    
    # 리샘플링 수행
    if resample_rate > 0:
        # 균일한 시간 간격으로 새 타임스탬프 생성
        duration = time_data[-1] - time_data[0]
        num_samples = int(duration * resample_rate) + 1
        new_time_data = np.linspace(time_data[0], time_data[-1], num_samples)
        
        # 각 관절에 대해 개별적으로 보간 수행
        new_ctrl_data = np.zeros((num_samples, ctrl_data.shape[1]))
        
        for i in range(ctrl_data.shape[1]):
            # 데이터 이상치 확인
            joint_data = ctrl_data[:, i]
            mean_val = np.mean(joint_data)
            std_val = np.std(joint_data)
            
            # 이상치 확인 (평균에서 3 표준편차 이상 벗어난 값)
            outliers = np.abs(joint_data - mean_val) > 3 * std_val
            if np.any(outliers):
                outlier_count = np.sum(outliers)
                print(f"경고: 관절 {i}에서 {outlier_count}개의 이상치 발견. 이상치는 필터링됩니다.")
                
                # 이상치가 아닌 데이터만 사용해 보간
                valid_indices = ~outliers
                valid_times = time_data[valid_indices]
                valid_values = joint_data[valid_indices]
                
                if len(valid_times) < 3:  # 충분한 유효 포인트가 없으면 모든 데이터 사용
                    f = interpolate.interp1d(time_data, joint_data, kind='linear', fill_value="extrapolate")
                else:
                    f = interpolate.interp1d(valid_times, valid_values, kind='linear', fill_value="extrapolate")
            else:
                # 이상치가 없으면 모든 데이터 사용
                f = interpolate.interp1d(time_data, joint_data, kind='linear', fill_value="extrapolate")
            
            # 큐빅 스플라인 보간 (부드러운 곡선)
            if smooth_data and len(time_data) > 3:
                if np.any(outliers):
                    valid_indices = ~outliers
                    valid_times = time_data[valid_indices]
                    valid_values = joint_data[valid_indices]
                    
                    if len(valid_times) > 3:  # 충분한 유효 포인트가 있을 때만 스플라인 사용
                        tck = interpolate.splrep(valid_times, valid_values, s=0.5)  # s는 스무딩 계수
                        new_ctrl_data[:, i] = interpolate.splev(new_time_data, tck)
                    else:
                        new_ctrl_data[:, i] = f(new_time_data)  # 선형 보간으로 대체
                else:
                    tck = interpolate.splrep(time_data, joint_data, s=0.5)
                    new_ctrl_data[:, i] = interpolate.splev(new_time_data, tck)
            else:
                # 선형 보간
                new_ctrl_data[:, i] = f(new_time_data)
        
        print(f"리샘플링 완료: {len(new_time_data)}개 샘플 ({resample_rate}Hz)")
        
        # 원본 데이터를 리샘플링된 데이터로 교체
        time_data = new_time_data
        ctrl_data = new_ctrl_data
    
    # 데이터 시각화 (디버그 모드)
    if debug_mode:
        try:
            # 그래프를 표시할 관절 선택 (최대 6개)
            num_joints = min(6, ctrl_data.shape[1])
            
            plt.figure(figsize=(12, 8))
            for i in range(num_joints):
                joint_name = joint_names[i] if i < len(joint_names) else f"Joint {i}"
                plt.plot(time_data, ctrl_data[:, i], label=joint_name)
            
            plt.xlabel('Time (s)')
            plt.ylabel('Joint Position')
            plt.title('Joint Positions from ROS Bag')
            plt.grid(True)
            plt.legend()
            
            # 그래프 저장
            debug_plot_file = os.path.splitext(output_file)[0] + "_debug_plot.png"
            plt.savefig(debug_plot_file)
            print(f"디버그 그래프 저장됨: {debug_plot_file}")
            
            # 추가 통계 정보
            print("\n관절별 통계 정보:")
            for i in range(min(10, ctrl_data.shape[1])):  # 최대 10개 관절까지만 출력
                joint_name = joint_names[i] if i < len(joint_names) else f"Joint {i}"
                joint_data = ctrl_data[:, i]
                joint_min = np.min(joint_data)
                joint_max = np.max(joint_data)
                joint_range = joint_max - joint_min
                print(f"  {joint_name}: 범위 {joint_min:.4f} ~ {joint_max:.4f} (폭: {joint_range:.4f})")
            
        except Exception as e:
            print(f"디버그 그래프 생성 중 오류 발생: {e}")
    
    # MuJoCo .act 파일로 변환
    
    # 관절 데이터 중 필요한 것만 선택 (MuJoCo 모델의 액추에이터 순서와 일치)
    # XML 파일의 액추에이터 순서를 따라야 함
    actuator_names = [
        "left_actuator_joint1", "left_actuator_joint2", "left_actuator_joint3", "left_actuator_joint4", "left_actuator_gripper_joint",
        "right_actuator_joint1", "right_actuator_joint2", "right_actuator_joint3", "right_actuator_joint4", "right_actuator_gripper_joint"
    ]
    
    # 관절 이름에서 필요한 인덱스 찾기
    required_joints = []  # 필요한 관절들의 인덱스를 저장
    
    # 왼쪽 관절
    left_joints = ['left_joint1', 'left_joint2', 'left_joint3', 'left_joint4', 'left_gripper_left_joint']
    right_joints = ['right_joint1', 'right_joint2', 'right_joint3', 'right_joint4', 'right_gripper_left_joint']
    
    all_required_joints = left_joints + right_joints
    
    # 관절 인덱스 찾기
    joint_indices = []
    for joint_name in all_required_joints:
        if joint_name in joint_names:
            joint_indices.append(joint_names.index(joint_name))
        else:
            print(f"경고: 필요한 관절 '{joint_name}'이 ROS 메시지에 없습니다.")
            # 누락된 관절의 경우 -1을 추가하여 나중에 0으로 채움
            joint_indices.append(-1)
    
    # 필요한 관절 데이터만 추출
    final_ctrl_data = np.zeros((len(time_data), len(all_required_joints)))
    
    for i, idx in enumerate(joint_indices):
        if idx >= 0:  # 유효한 인덱스인 경우
            final_ctrl_data[:, i] = ctrl_data[:, idx]
    
    # 최종 출력 정보
    print(f"\n최종 데이터: {len(time_data)}개 프레임, {final_ctrl_data.shape[1]}개 액추에이터")
    
    # .act 파일 헤더 작성
    act_header = array.array('i', [1, 0])  # 형식 버전, 예약
    act_header.append(final_ctrl_data.shape[1])  # 제어 차원
    act_header.append(final_ctrl_data.shape[0])  # 프레임 수
    
    # .act 파일 저장
    with open(output_file, 'wb') as f:
        act_header.tofile(f)
        time_data.astype(np.float64).tofile(f)
        final_ctrl_data.astype(np.float64).tofile(f)
    
    print(f"변환 완료: {output_file}")
    print(f"파일 크기: {os.path.getsize(output_file) / 1024:.1f} KB")
    
    conn.close()

if __name__ == "__main__":
    main()