#!/usr/bin/env python3
import os
import time
import numpy as np
import mujoco
from mujoco.glfw import glfw

# Load MuJoCo model and data
xml_path = os.path.join(os.path.dirname(__file__), 'dual_arm_robot.xml')
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Initialize GLFW for multiview windows
if not glfw.init():
    raise RuntimeError('Failed to initialize GLFW')

# Create windows: external, left wrist, right wrist
window_main = glfw.create_window(1200, 900, 'External View', None, None)
window_left = glfw.create_window(640, 480, 'Left Wrist Cam', None, window_main)
window_right = glfw.create_window(640, 480, 'Right Wrist Cam', None, window_main)
for win in (window_main, window_left, window_right):
    if not win:
        glfw.terminate()
        raise RuntimeError('Failed to create GLFW window')

glfw.make_context_current(window_main)
glfw.swap_interval(1)

# Setup cameras and scenes
# External camera
cam_main = mujoco.MjvCamera()
opt = mujoco.MjvOption()
scene_main = mujoco.MjvScene(model, maxgeom=10000)
context_main = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
cam_main.distance = 1.2
cam_main.elevation = -20.0
cam_main.azimuth = 0.0
cam_main.lookat = np.array([0.0, 0.0, 0.3])

# Left wrist camera
glfw.make_context_current(window_left)
cam_left = mujoco.MjvCamera()
scene_left = mujoco.MjvScene(model, maxgeom=10000)
context_left = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
cam_id_left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'left_wrist_cam')
cam_left.type = mujoco.mjtCamera.mjCAMERA_FIXED
cam_left.fixedcamid = cam_id_left
import numpy as np
#######
# 1) 원본 카메라 쿼터니언(w, x, y, z) 읽어오기
base_q = np.array(model.cam_quat[cam_id_left], dtype=np.float64)

# 2) Y축(위 방향) 기준 180° 회전 쿼터니언
flip_q = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)

# 3) 쿼터니언 곱셈 (base ⊗ flip)
w0, x0, y0, z0 = base_q
w1, x1, y1, z1 = flip_q
new_q = np.array([
    w0*w1 - x0*x1 - y0*y1 - z0*z1,
    w0*x1 + x0*w1 + y0*z1 - z0*y1,
    w0*y1 - x0*z1 + y0*w1 + z0*x1,
    w0*z1 + x0*y1 - y0*x1 + z0*w1,
], dtype=np.float64)

# 4) 모델에 덮어쓰기
model.cam_quat[cam_id_left] = new_q


# Right wrist camera
glfw.make_context_current(window_right)
cam_right = mujoco.MjvCamera()
scene_right = mujoco.MjvScene(model, maxgeom=10000)
context_right = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
cam_id_right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'right_wrist_cam')
cam_right.type = mujoco.mjtCamera.mjCAMERA_FIXED
cam_right.fixedcamid = cam_id_right

# Actuator indices for control
left_joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'left_actuator_joint{i}') for i in range(1,5)]
left_joints.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'left_actuator_gripper_joint'))
right_joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'right_actuator_joint{i}') for i in range(1,5)]
right_joints.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'right_actuator_gripper_joint'))
all_actuators = left_joints + right_joints

# Control parameters
step_size = 0.05  # base step in radians or meters
# Precompute control ranges
ctrlrange = np.array(model.actuator_ctrlrange).reshape(model.nu, 2)
ctrl_lo, ctrl_hi = ctrlrange[:,0], ctrlrange[:,1]

# State variables
selected_actuator = 0
paused = False

# Reset to initial pose
def reset_pose():
    # simple initial angles
    init_left = [0.0, -0.3, 0.6, 0.0, 0.01]
    init_right = [0.0, -0.3, 0.6, 0.0, 0.01]
    for idx, ang in zip(left_joints, init_left):
        data.ctrl[idx] = ang
    for idx, ang in zip(right_joints, init_right):
        data.ctrl[idx] = ang
    print('Pose reset.')

# Clamp helper
def clamp(idx, val):
    return float(np.clip(val, ctrl_lo[idx], ctrl_hi[idx]))

# Adjust selected joint
def adjust_joint(direction):
    global data
    idx = all_actuators[selected_actuator]
    # increase step for joint1 (indices 0 and 5)
    factor = 4 if selected_actuator in (0, 5) else 1
    delta = step_size * direction * factor
    data.ctrl[idx] = clamp(idx, data.ctrl[idx] + delta)
    print(f'Actuator {selected_actuator} -> {data.ctrl[idx]:.3f}')

# Keyboard callback
def keyboard(window, key, scancode, action, mods):
    global selected_actuator, paused
    if action not in (glfw.PRESS, glfw.REPEAT):
        return
    if key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window_main, True)
        glfw.set_window_should_close(window_left, True)
        glfw.set_window_should_close(window_right, True)
    elif key == glfw.KEY_R:
        reset_pose()
    elif key == glfw.KEY_P:
        paused = not paused
        print('Paused' if paused else 'Resumed')
    elif key == glfw.KEY_UP:
        adjust_joint(+1)
    elif key == glfw.KEY_DOWN:
        adjust_joint(-1)
    # Select joints: 1-5 left, 6-0 right
    elif glfw.KEY_1 <= key <= glfw.KEY_5:
        selected_actuator = key - glfw.KEY_1
    elif key == glfw.KEY_6:
        selected_actuator = 5
    elif key == glfw.KEY_7:
        selected_actuator = 6
    elif key == glfw.KEY_8:
        selected_actuator = 7
    elif key == glfw.KEY_9:
        selected_actuator = 8
    elif key == glfw.KEY_0:
        selected_actuator = 9
    print(f'Selected actuator: {selected_actuator}')

# Set callback on main window
glfw.set_key_callback(window_main, keyboard)

# Initialize pose
reset_pose()
print('=== Manual Multiview Control Started ===')
print('Keys: 1-5: left joints, 6-0: right joints; UP/DOWN: adjust; R: reset; P: pause; ESC: exit')

# Render helper
def render(window, cam, scene, context, overlay=False):
    glfw.make_context_current(window)
    w, h = glfw.get_framebuffer_size(window)
    viewport = mujoco.MjrRect(0, 0, w, h)
    mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene)
    mujoco.mjr_render(viewport, scene, context)
    if overlay and window == window_main:
        text = [
            f'Selected: {"L" if selected_actuator<5 else "R"}{(selected_actuator%5)+1}',
            f'Value: {data.ctrl[all_actuators[selected_actuator]]:.3f}'
        ]
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport, "\n".join(text), "", context)
    glfw.swap_buffers(window)

# Main loop
while not (glfw.window_should_close(window_main) or glfw.window_should_close(window_left) or glfw.window_should_close(window_right)):
    if not paused:
        mujoco.mj_step(model, data)
    # Render all views
    render(window_main, cam_main, scene_main, context_main, overlay=True)
    render(window_left, cam_left, scene_left, context_left)
    render(window_right, cam_right, scene_right, context_right)
    glfw.poll_events()
    time.sleep(0.01)

# Cleanup
glfw.terminate()
print('Exited.')
