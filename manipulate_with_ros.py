import mujoco
import numpy as np
import time
from mujoco.glfw import glfw
import os

# ROS2 관련 추가
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

# Initialize ROS2
rclpy.init()
ros_node = rclpy.create_node('mujoco_publisher')
joint_pub = ros_node.create_publisher(JointState, '/joint_states', 10)

# Load MuJoCo model
model_path = os.path.join(os.path.dirname(__file__), "scene.xml")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Initialize GLFW and create window
glfw.init()
window = glfw.create_window(1200, 900, "OpenManipulator X Control", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# MuJoCo visualization
cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()
scene = mujoco.MjvScene(model, maxgeom=10000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

cam.distance = 1.0
cam.elevation = -20.0
cam.azimuth = 90.0
cam.lookat = np.array([0.2, 0.0, 0.2])

# Mouse control
button_left = button_middle = button_right = False
lastx = lasty = 0

# Control states
demo_mode = False
run_once = False
demo_step = 0
joint_idx = 0

# Positions
home_position = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
pick_position = np.array([1.57, 0.5, 0.5, -0.5, 0.0])
place_position = np.array([-1.57, 0.5, 0.5, -0.5, 0.0])
gripper_open_val = -0.01
gripper_close_val = 0.019

def set_joint_targets(targets):
    for i, target in enumerate(targets):
        if i < 5:
            data.ctrl[i] = target

def interpolate_joints(start_pos, end_pos, steps=100):
    for step in range(steps):
        alpha = step / steps
        current_pos = start_pos * (1 - alpha) + end_pos * alpha
        set_joint_targets(current_pos)
        mujoco.mj_step(model, data)
        publish_joint_state()
        render_scene()
        glfw.poll_events()
        if glfw.window_should_close(window):
            return False
    return True

def publish_joint_state():
    msg = JointState()
    msg.header.stamp = ros_node.get_clock().now().to_msg()
    msg.name = [f"joint_{i+1}" for i in range(5)]
    msg.position = data.ctrl[:5].tolist()
    joint_pub.publish(msg)
    rclpy.spin_once(ros_node, timeout_sec=0.001)

def render_scene():
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)
    mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene)
    mujoco.mjr_render(viewport, scene, context)
    mujoco.mjr_overlay(
        mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
        viewport,
        "ESC: Exit\nSPACE: Demo\nENTER: Run Once\nH: Home\nP: Pick\nL: Place\nO: Open\nC: Close\n1-4: Select Joint\n↑↓: Adjust",
        "", context)
    glfw.swap_buffers(window)

def run_demo_step():
    global demo_step
    current_pos = data.ctrl[:5].copy()
    if demo_step == 0:
        if interpolate_joints(current_pos, home_position): demo_step += 1
    elif demo_step == 1:
        current_pos[-1] = gripper_open_val
        if interpolate_joints(data.ctrl[:5], current_pos): demo_step += 1; time.sleep(0.5)
    elif demo_step == 2:
        if interpolate_joints(data.ctrl[:5], pick_position): demo_step += 1; time.sleep(0.5)
    elif demo_step == 3:
        current_pos = data.ctrl[:5].copy(); current_pos[-1] = gripper_close_val
        if interpolate_joints(data.ctrl[:5], current_pos): demo_step += 1; time.sleep(0.5)
    elif demo_step == 4:
        if interpolate_joints(data.ctrl[:5], place_position): demo_step += 1; time.sleep(0.5)
    elif demo_step == 5:
        current_pos = data.ctrl[:5].copy(); current_pos[-1] = gripper_open_val
        if interpolate_joints(data.ctrl[:5], current_pos): demo_step = 0; time.sleep(0.5); return False
    return True

def keyboard(window, key, scancode, act, mods):
    global demo_mode, run_once, demo_step, joint_idx
    if act == glfw.PRESS:
        if key == glfw.KEY_ESCAPE: glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_SPACE: demo_mode = not demo_mode; demo_step = 0
        elif key == glfw.KEY_ENTER: run_once = True; demo_step = 0
        elif key == glfw.KEY_H: set_joint_targets(home_position)
        elif key == glfw.KEY_P: set_joint_targets(pick_position)
        elif key == glfw.KEY_L: set_joint_targets(place_position)
        elif key == glfw.KEY_O: pos = data.ctrl[:5].copy(); pos[-1] = gripper_open_val; set_joint_targets(pos)
        elif key == glfw.KEY_C: pos = data.ctrl[:5].copy(); pos[-1] = gripper_close_val; set_joint_targets(pos)
        elif glfw.KEY_1 <= key <= glfw.KEY_4: joint_idx = key - glfw.KEY_1
        elif key == glfw.KEY_UP: pos = data.ctrl[:5].copy(); pos[joint_idx] += 0.1; set_joint_targets(pos)
        elif key == glfw.KEY_DOWN: pos = data.ctrl[:5].copy(); pos[joint_idx] -= 0.1; set_joint_targets(pos)

def mouse_button(window, button, act, mods):
    global button_left, button_middle, button_right, lastx, lasty
    if button == glfw.MOUSE_BUTTON_LEFT: button_left = (act == glfw.PRESS)
    elif button == glfw.MOUSE_BUTTON_MIDDLE: button_middle = (act == glfw.PRESS)
    elif button == glfw.MOUSE_BUTTON_RIGHT: button_right = (act == glfw.PRESS)
    lastx, lasty = glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    global lastx, lasty
    dx, dy = xpos - lastx, ypos - lasty
    if button_left:
        cam.elevation = np.clip(cam.elevation - 0.1 * dy, -90, 90)
        cam.azimuth = (cam.azimuth + 0.1 * dx) % 360
    elif button_middle:
        cam.lookat[0] += -0.001 * dx * cam.distance
        cam.lookat[1] += 0.001 * dy * cam.distance
    elif button_right:
        cam.distance = np.clip(cam.distance + 0.01 * dy, 0.1, 5.0)
    lastx, lasty = xpos, ypos

def scroll(window, xoffset, yoffset):
    cam.distance = np.clip(cam.distance - 0.1 * yoffset, 0.1, 5.0)

# Register callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_scroll_callback(window, scroll)

try:
    set_joint_targets(home_position)
    print("Use SPACE for demo, H/P/L/O/C to move, ↑/↓ to adjust joint.")
    while not glfw.window_should_close(window):
        mujoco.mj_step(model, data)
        if demo_mode:
            run_demo_step()
        elif run_once:
            if not run_demo_step(): run_once = False
        publish_joint_state()
        render_scene()
        glfw.poll_events()
finally:
    glfw.terminate()
    ros_node.destroy_node()
    rclpy.shutdown()

