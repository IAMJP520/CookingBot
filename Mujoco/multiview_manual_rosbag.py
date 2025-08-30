#!/usr/bin/env python3
import os
import time
import numpy as np
import mujoco
from mujoco.glfw import glfw
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

# Helper to clamp control values
from numpy import clip as _clip
def clamp(val, lo, hi):
    return float(_clip(val, lo, hi))

# Main entry
def main():
    selected_actuator = 0
    paused = False

    # Initialize ROS2
    rclpy.init()
    node = Node('multi_view_manual_pub')
    pub = node.create_publisher(JointState, '/joint_states', 10)

    # Load MuJoCo model and data
    xml_path = os.path.join(os.path.dirname(__file__), 'dual_arm_robot.xml')
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Setup GLFW windows
    if not glfw.init():
        raise RuntimeError('Failed to initialize GLFW')
    window_main = glfw.create_window(1200, 900, 'External View', None, None)
    window_left = glfw.create_window(640, 480, 'Left Wrist Cam', None, window_main)
    window_right = glfw.create_window(640, 480, 'Right Wrist Cam', None, window_main)
    for win in (window_main, window_left, window_right):
        if not win:
            glfw.terminate()
            raise RuntimeError('Failed to create GLFW window')
    glfw.make_context_current(window_main)
    glfw.swap_interval(1)

    # Setup cameras
    cam_main = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scene_main = mujoco.MjvScene(model, maxgeom=10000)
    context_main = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    cam_main.distance, cam_main.elevation, cam_main.azimuth = 1.2, -20.0, 0.0
    cam_main.lookat = np.array([0.0, 0.0, 0.3])

    # Left wrist camera
    cam_id_left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'left_wrist_cam')
    # Correct left camera orientation by flipping 180° about Y axis
    base_q = np.array(model.cam_quat[cam_id_left], dtype=np.float64)
    # flip quaternion represents 180° around Y: [cos(pi/2)=0, axis*sin(pi/2)=(0,1,0)] -> (w,x,y,z)
    flip_q = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    # Perform quaternion multiplication: flip_q ⊗ base_q
    w0, x0, y0, z0 = flip_q
    w1, x1, y1, z1 = base_q
    new_q = np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1
    ], dtype=np.float64)
    model.cam_quat[cam_id_left] = new_q

    glfw.make_context_current(window_left)
    cam_left = mujoco.MjvCamera()
    scene_left = mujoco.MjvScene(model, maxgeom=10000)
    context_left = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    cam_left.type, cam_left.fixedcamid = mujoco.mjtCamera.mjCAMERA_FIXED, cam_id_left

    # Right wrist camera
    cam_id_right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'right_wrist_cam')
    glfw.make_context_current(window_right)
    cam_right = mujoco.MjvCamera()
    scene_right = mujoco.MjvScene(model, maxgeom=10000)
    context_right = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    cam_right.type, cam_right.fixedcamid = mujoco.mjtCamera.mjCAMERA_FIXED, cam_id_right

    # Actuator IDs
    left_joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'left_actuator_joint{i}') for i in range(1,5)]
    left_joints.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'left_actuator_gripper_joint'))
    right_joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'right_actuator_joint{i}') for i in range(1,5)]
    right_joints.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'right_actuator_gripper_joint'))
    all_actuators = left_joints + right_joints

    # Control parameters
    ctrlrange = np.array(model.actuator_ctrlrange).reshape(model.nu, 2)
    ctrl_lo, ctrl_hi = ctrlrange[:, 0], ctrlrange[:, 1]
    step_size = 0.05

    # Define control functions
    def reset_pose():
        init_left = [0.0, -0.3, 0.6, 0.0, 0.01]
        init_right = [0.0, -0.3, 0.6, 0.0, 0.01]
        for idx, ang in zip(left_joints, init_left): data.ctrl[idx] = ang
        for idx, ang in zip(right_joints, init_right): data.ctrl[idx] = ang
        node.get_logger().info('Pose reset.')

    def adjust_joint(direction):
        idx = all_actuators[selected_actuator]
        factor = 4 if selected_actuator in (0,5) else 1
        delta = step_size * direction * factor
        data.ctrl[idx] = clamp(data.ctrl[idx] + delta, ctrl_lo[idx], ctrl_hi[idx])
        node.get_logger().info(f'Actuator {selected_actuator} -> {data.ctrl[idx]:.3f}')

    # Keyboard callback
    def keyboard(window, key, scancode, action, mods):
        nonlocal selected_actuator, paused
        if action not in (glfw.PRESS, glfw.REPEAT): return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window_main, True)
        elif key == glfw.KEY_R:
            reset_pose()
        elif key == glfw.KEY_P:
            paused = not paused; node.get_logger().info('Paused' if paused else 'Resumed')
        elif key == glfw.KEY_UP:
            adjust_joint(+1)
        elif key == glfw.KEY_DOWN:
            adjust_joint(-1)
        elif glfw.KEY_1 <= key <= glfw.KEY_5:
            selected_actuator = key - glfw.KEY_1
        elif key == glfw.KEY_6: selected_actuator = 5
        elif key == glfw.KEY_7: selected_actuator = 6
        elif key == glfw.KEY_8: selected_actuator = 7
        elif key == glfw.KEY_9: selected_actuator = 8
        elif key == glfw.KEY_0: selected_actuator = 9
        node.get_logger().info(f'Selected actuator: {selected_actuator}')
    glfw.set_key_callback(window_main, keyboard)

    # Initialize and run main loop
    reset_pose()
    node.get_logger().info('=== Manual Multiview Control with ROS2 ===')

    def render(window, cam, scene, context, overlay=False):
        glfw.make_context_current(window)
        w, h = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, w, h)
        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(viewport, scene, context)
        if overlay and window == window_main:
            txt = [f'Sel: {{"L" if selected_actuator<5 else "R"}}{{(selected_actuator%5)+1}}',
                   f'Val: {{data.ctrl[all_actuators[selected_actuator]]:.3f}}']
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                viewport, "\n".join(txt), "", context)
        glfw.swap_buffers(window)

    while not glfw.window_should_close(window_main):
        if not paused: mujoco.mj_step(model, data)
        # Publish JointState
        js = JointState()
        js.header.stamp = node.get_clock().now().to_msg()
        js.name = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) for aid in all_actuators]
        js.position = [data.ctrl[aid] for aid in all_actuators]
        pub.publish(js)
        rclpy.spin_once(node, timeout_sec=0)
        # Render views
        render(window_main, cam_main, scene_main, context_main, overlay=True)
        render(window_left, cam_left, scene_left, context_left)
        render(window_right, cam_right, scene_right, context_right)
        glfw.poll_events(); time.sleep(0.01)

    node.destroy_node(); rclpy.shutdown(); glfw.terminate(); print('Exited.')

if __name__ == '__main__':
    main()