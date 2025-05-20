#!/usr/bin/env python3
"""
db3_to_hdf5.py
  .db3 (ROS2 bag) → HDF5 변환

Usage:
  python3 db3_to_hdf5.py <bag_directory> <topic_name> <output_h5_file>

Example:
  python3 db3_to_hdf5.py dual_arm_bag /joint_states joint_states.h5
"""
import sys, os, sqlite3
import h5py
import numpy as np
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    bag_dir, topic_name, out_h5 = sys.argv[1:]
    # 1) Find .db3 file
    db3 = None
    for fn in os.listdir(bag_dir):
        if fn.endswith('.db3'):
            db3 = os.path.join(bag_dir, fn)
            break
    if not db3:
        raise FileNotFoundError(f"No .db3 file in {bag_dir}")

    # 2) Connect and lookup topic_id/type
    conn = sqlite3.connect(db3)
    cur  = conn.cursor()
    cur.execute("SELECT id,type FROM topics WHERE name=?", (topic_name,))
    row = cur.fetchone()
    if not row:
        raise RuntimeError(f"Topic {topic_name} not found in {db3}")
    topic_id, topic_type = row
    msg_type = get_message(topic_type)

    # 3) Query all messages sorted by timestamp
    cur.execute("""
        SELECT data, timestamp
          FROM messages
         WHERE topic_id=?
         ORDER BY timestamp
    """, (topic_id,))
    entries = cur.fetchall()
    conn.close()
    if not entries:
        raise RuntimeError(f"No messages on {topic_name}")

    # 4) Deserialize and collect fields
    stamps = []
    positions = []
    velocities = []
    efforts = []
    for blob, ts in entries:
        msg = deserialize_message(blob, msg_type)
        stamps.append(ts / 1e9)  # seconds
        positions.append(msg.position if msg.position else [0.0]*len(msg.name))
        velocities.append(msg.velocity if msg.velocity else [0.0]*len(msg.name))
        efforts.append(msg.effort if msg.effort else [0.0]*len(msg.name))

    stamps = np.array(stamps, dtype=np.float64)       # (T,)
    pos_arr = np.array(positions, dtype=np.float64)   # (T, N)
    vel_arr = np.array(velocities, dtype=np.float64)  # (T, N)
    eff_arr = np.array(efforts, dtype=np.float64)     # (T, N)
    names   = np.array(entries[0][0] and msg.name or [], dtype='S')

    # 5) Write to HDF5
    with h5py.File(out_h5, 'w') as hf:
        grp = hf.create_group(topic_name.strip('/'))
        grp.create_dataset('time',       data=stamps)
        grp.create_dataset('joint_names',data=names)
        grp.create_dataset('position',   data=pos_arr)
        grp.create_dataset('velocity',   data=vel_arr)
        grp.create_dataset('effort',     data=eff_arr)

    print(f"✅ Wrote HDF5: {out_h5}\n  frames: {len(stamps)}, joints: {pos_arr.shape[1]}")


if __name__ == '__main__':
    main()
