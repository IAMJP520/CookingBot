#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import mujoco
from mujoco.glfw import glfw

# Usage: run_act_multiview.py <model_xml> <act_file>
if len(sys.argv) != 3:
    print("Usage: ./run_act_multiview.py dual_arm_robot.xml dual_arm.act")
    sys.exit(1)
model_path = sys.argv[1]
act_path   = sys.argv[2]

# Load MuJoCo model and data
model = mujoco.MjModel.from_xml_path(model_path)
data  = mujoco.MjData(model)


####################쿼터니언 회전##########################3
# ─── 왼쪽 손목 카메라 180° 뒤집기 ─────────────────────────────────────────────┐
# Flip quaternion for left wrist camera so replay matches recording
cam_id_left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'left_wrist_cam')
base_q = np.array(model.cam_quat[cam_id_left], dtype=np.float64)
# 180° rotation about Y-axis quaternion: (w=0, x=0, y=1, z=0)
flip_q = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
# Perform q_flip * q_base
w0, x0, y0, z0 = flip_q
w1, x1, y1, z1 = base_q
new_q = np.array([
    w0*w1 - x0*x1 - y0*y1 - z0*z1,
    w0*x1 + x0*w1 + y0*z1 - z0*y1,
    w0*y1 - x0*z1 + y0*w1 + z0*x1,
    w0*z1 + x0*y1 - y0*x1 + z0*w1
], dtype=np.float64)
model.cam_quat[cam_id_left] = new_q
# ───────────────────────────────────────────────────────────────────────────────┘

# Load .act file
def load_act_file(filename):
    with open(filename, 'rb') as f:
        hdr = np.fromfile(f, dtype=np.int32, count=4)
        if hdr[0] != 1:
            raise ValueError(f"Unsupported .act version: {hdr[0]}")
        ctrl_dim   = int(hdr[2])
        num_frames = int(hdr[3])
        time_data  = np.fromfile(f, dtype=np.float64, count=num_frames)
        ctrl_data  = np.fromfile(f, dtype=np.float64, count=num_frames*ctrl_dim)
        ctrl_data  = ctrl_data.reshape(num_frames, ctrl_dim)
        return time_data, ctrl_data

print(f"Loading .act file: {act_path}")
time_data, ctrl_data = load_act_file(act_path)
num_frames, ctrl_dim = ctrl_data.shape
print(f"Loaded {num_frames} frames, ctrl_dim={ctrl_dim}")

# GLFW init
def init_window(title, width, height, share=None):
    win = glfw.create_window(width, height, title, None, share)
    if not win:
        glfw.terminate()
        raise RuntimeError(f"Failed to create window: {title}")
    return win

if not glfw.init():
    raise RuntimeError("GLFW initialization failed")

# Create three windows: external, left cam, right cam
win_main  = init_window('External View', 1200, 900)
win_left  = init_window('Left Wrist Cam', 640, 480, share=win_main)
win_right = init_window('Right Wrist Cam', 640, 480, share=win_main)
for w in (win_main, win_left, win_right):
    glfw.make_context_current(w)
    glfw.swap_interval(1)

# MJV setup
opt       = mujoco.MjvOption()
scene_main= mujoco.MjvScene(model, maxgeom=10000)
scene_left= mujoco.MjvScene(model, maxgeom=10000)
scene_right=mujoco.MjvScene(model, maxgeom=10000)
ctx_main  = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
ctx_left  = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
ctx_right = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

# Cameras
cam_main = mujoco.MjvCamera()
cam_main.distance  = 1.2
cam_main.elevation = -20.0
cam_main.azimuth   = 0.0
cam_main.lookat    = np.array([0.0,0.0,0.3])

# Fixed wrist cams by name
cam_id_left  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'left_wrist_cam')
cam_id_right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'right_wrist_cam')

cam_left = mujoco.MjvCamera()
cam_left.type       = mujoco.mjtCamera.mjCAMERA_FIXED
cam_left.fixedcamid = cam_id_left

cam_right= mujoco.MjvCamera()
cam_right.type       = mujoco.mjtCamera.mjCAMERA_FIXED
cam_right.fixedcamid = cam_id_right

# Playback state
paused = False
frame_idx = 0
last_update = time.time()
data.time = 0.0

# Keyboard callback
def keyboard(window, key, scancode, action, mods):
    global paused, frame_idx, data
    if action != glfw.PRESS:
        return
    if key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(win_main, True)
        glfw.set_window_should_close(win_left, True)
        glfw.set_window_should_close(win_right, True)
    elif key == glfw.KEY_SPACE:
        paused = not paused
    elif key == glfw.KEY_R:
        data.time = 0.0
        frame_idx = 0

for w in (win_main, win_left, win_right):
    glfw.set_key_callback(w, keyboard)

# Render helper
def render(window, cam, scene, ctx, overlay=False):
    glfw.make_context_current(window)
    w, h = glfw.get_framebuffer_size(window)
    vp = mujoco.MjrRect(0,0,w,h)
    mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene)
    mujoco.mjr_render(vp, scene, ctx)
    if overlay and window == win_main:
        txt = [f"Time: {data.time:.2f}  Frame: {frame_idx}/{num_frames}",
               "Space: pause/play  R: restart  Esc: exit"]
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                           vp, "\n".join(txt), "", ctx)
    glfw.swap_buffers(window)

print("Starting playback...")

# Main loop
while not (glfw.window_should_close(win_main) or glfw.window_should_close(win_left) or glfw.window_should_close(win_right)):
    now = time.time()
    dt = now - last_update
    last_update = now
    if not paused:
        data.time += dt
        while frame_idx < num_frames-1 and data.time > time_data[frame_idx+1]:
            frame_idx += 1
    if frame_idx < num_frames:
        for i in range(min(model.nu, ctrl_dim)):
            data.ctrl[i] = ctrl_data[frame_idx, i]
    mujoco.mj_step(model, data)
    render(win_main, cam_main, scene_main, ctx_main, overlay=True)
    render(win_left, cam_left, scene_left, ctx_left)
    render(win_right, cam_right, scene_right, ctx_right)
    glfw.poll_events()
    time.sleep(0.01)

# Cleanup
glfw.terminate()
print("Playback exited.")
