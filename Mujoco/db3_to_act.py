#!/usr/bin/env python3
"""
rosbag2 db3 파일을 MuJoCo .act 파일로 변환
사용법: python3 db3_to_act_ros2.py [bag_directory] [topic_name] [output_act_file]
예시: python3 db3_to_act_ros2.py demo_1_0 /joint_states demo_1.act
"""

import sys
import numpy as np
import os
import sqlite3
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import array

def main():
    if len(sys.argv) != 4:
        print("사용법: python3 db3_to_act_ros2.py [bag_directory] [topic_name] [output_act_file]")
        print("예시: python3 db3_to_act_ros2.py demo_1_0 /joint_states demo_1.act")
        sys.exit(1)
    
    bag_directory = sys.argv[1]
    topic_name = sys.argv[2]
    output_file = sys.argv[3]
    
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
    
    for row in rows:
        data, timestamp = row
        msg = deserialize_message(data, msg_type)
        
        # JointState 메시지에서 position 데이터 추출
        if hasattr(msg, 'position') and msg.position:
            ctrl_data.append(msg.position)
            timestamps.append(timestamp)
    
    if not ctrl_data:
        print("오류: 메시지에서 position 데이터를 찾을 수 없습니다.")
        conn.close()
        sys.exit(1)
    
    # 데이터 형식 변환
    time_data = np.array(timestamps, dtype=np.int64)
    time_data = (time_data - time_data[0]) / 1e9  # 나노초를 초로 변환
    ctrl_data = np.array(ctrl_data, dtype=np.float64)
    
    print(f"시간 데이터 형태: {time_data.shape}")
    print(f"제어 데이터 형태: {ctrl_data.shape}")
    
    # .act 파일 형식으로 변환
    act_header = array.array('i', [1, 0])  # 형식 버전, 예약
    act_header.append(ctrl_data.shape[1])  # 제어 차원
    act_header.append(ctrl_data.shape[0])  # 프레임 수
    
    # .act 파일 저장
    with open(output_file, 'wb') as f:
        act_header.tofile(f)
        time_data.tofile(f)
        ctrl_data.tofile(f)
    
    print(f"변환 완료: {output_file}")
    conn.close()

if __name__ == "__main__":
    main()
