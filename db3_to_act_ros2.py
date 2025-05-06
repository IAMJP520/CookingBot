#!/usr/bin/env python3
import sys
import numpy as np

import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions

def main():
    if len(sys.argv) < 2:
        print("Usage: db3_to_act_ros2.py <bag_directory> [<topic_name> [<output.act>]]")
        sys.exit(1)

    bag_dir    = sys.argv[1]
    topic_name = sys.argv[2] if len(sys.argv) > 2 else '/joint_states'
    act_out    = sys.argv[3] if len(sys.argv) > 3 else 'demo_0.act'

    # ROS init
    rclpy.init()

    # rosbag2 reader 설정 (디렉토리 단위)
    reader = SequentialReader()
    storage_opts   = StorageOptions(uri=bag_dir, storage_id='sqlite3')
    converter_opts = ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    reader.open(storage_opts, converter_opts)

    # 토픽 타입 매핑
    topic_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic_name not in topic_map:
        print(f"❌ 토픽 '{topic_name}' 이(가) 이 bag에 없습니다.")
        sys.exit(1)

    msg_cls = get_message(topic_map[topic_name])
    ctrl_list = []

    # 메시지 읽어서 배열 수집
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == topic_name:
            msg = deserialize_message(data, msg_cls)
            # position 사용. 필요시 msg.effort 등으로 변경 가능
            ctrl_list.append(np.array(msg.position, dtype=np.float32))

    if not ctrl_list:
        print("⚠️ 메시지가 한 개도 없습니다.")
        sys.exit(1)

    arr = np.vstack(ctrl_list)  # shape: (T, nu)
    with open(act_out, 'wb') as f:
        f.write(arr.tobytes())

    print(f"✅ 변환 완료: {act_out} (shape: {arr.shape})")

if __name__ == '__main__':
    main()

