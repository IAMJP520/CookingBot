#!/usr/bin/env python3
"""
ros2 bag(.db3) → MuJoCo .act 변환 (dual‐arm)
사용법: python3 db3_to_act_dual.py [bag_directory] [topic_name] [output_act_file]
예시: python3 db3_to_act_dual.py demo_bag /joint_states dual_arm.act
"""

import sys, os, sqlite3, array
import numpy as np
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 db3_to_act_dual.py <bag_directory> <topic_name> <output_act_file>")
        sys.exit(1)

    bag_dir, topic_name, output_file = sys.argv[1], sys.argv[2], sys.argv[3]

    # 1) .db3 파일 찾기
    db_file = None
    for f in os.listdir(bag_dir):
        if f.endswith('.db3'):
            db_file = os.path.join(bag_dir, f)
            break
    if db_file is None:
        print(f"Error: no .db3 in {bag_dir}")
        sys.exit(1)

    # 2) SQLite 연결 및 토픽 ID/타입 조회
    conn   = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT id, type FROM topics WHERE name = ?", (topic_name,))
    row = cursor.fetchone()
    if row is None:
        print(f"Error: topic '{topic_name}' not found in {db_file}")
        conn.close()
        sys.exit(1)
    topic_id, topic_type = row
    msg_type = get_message(topic_type)

    # 3) 메시지 쿼리 (정렬된 timestamp 순)
    cursor.execute("""
        SELECT data, timestamp
          FROM messages
         WHERE topic_id = ?
         ORDER BY timestamp
    """, (topic_id,))
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print(f"Error: no messages on '{topic_name}'")
        sys.exit(1)

    # 4) deserialize & 데이터 수집
    ctrl_list  = []
    timestamps = []
    for blob, ts in rows:
        msg = deserialize_message(blob, msg_type)
        # dual‐arm → position 길이 10이라고 가정
        ctrl_list.append(msg.position)
        timestamps.append(ts)

    # 5) numpy 변환
    time_ns   = np.array(timestamps, dtype=np.int64)
    time_data = (time_ns - time_ns[0]) / 1e9       # 첫 프레임 0초 기준
    ctrl_data = np.array(ctrl_list, dtype=np.float64)  # shape (T,10)

    # 6) .act 헤더 작성 (int32 ×4)
    #    [version, reserved, ctrl_dim, num_frames]
    hdr = array.array('i', [1, 0, ctrl_data.shape[1], ctrl_data.shape[0]])

    # 7) 파일 저장
    with open(output_file, 'wb') as f:
        hdr.tofile(f)                       # int32 헤더 :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
        time_data.tofile(f)                 # float64 시간 데이터 :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}
        ctrl_data.tofile(f)                 # float64 제어 데이터 :contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}

    print(f"✅ Saved {output_file}: shape={ctrl_data.shape}")

if __name__ == '__main__':
    main()
