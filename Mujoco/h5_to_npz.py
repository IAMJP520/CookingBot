import h5py
import numpy as np

def convert_effort_only_to_npz(h5_path, npz_path):
    """
    Convert .h5 file with joint_states/effort into ACT-compatible .npz file
    with observations and actions both set to 10D effort vectors.
    """
    with h5py.File(h5_path, 'r') as f:
        effort = f['joint_states/effort'][:]  # shape: (T, 10)

    # Use effort as both observations and actions
    obs = effort.astype(np.float32)[None, :, :]  # (1, T, 10)
    act = effort.astype(np.float32)[None, :, :]  # (1, T, 10)

    np.savez(npz_path, observations=obs, actions=act)
    print(f"Saved {npz_path} with shape: obs {obs.shape}, act {act.shape}")

# 예시 사용법
if __name__ == "__main__":
    convert_effort_only_to_npz("joint_states_1.h5", "joint_states_1_effort_only.npz")
    convert_effort_only_to_npz("joint_states_2.h5", "joint_states_2_effort_only.npz")
