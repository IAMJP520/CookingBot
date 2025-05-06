import numpy as np
import mujoco
import time
import os
from mujoco.glfw import glfw

# Debug: current working directory
print(f"Current working directory: {os.getcwd()}")

# Load MuJoCo model
model_path = os.path.join(os.path.dirname(__file__), "dual_arm_robot.xml")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
print("Model loaded successfully!")

# Initialize GLFW
if not glfw.init():
    raise RuntimeError("Failed to initialize GLFW")

# Create windows for external and wrist cameras
window_main = glfw.create_window(1200, 900, "External View", None, None)
window_left = glfw.create_window(640, 480, "Left Wrist Cam", None, window_main)
window_right = glfw.create_window(640, 480, "Right Wrist Cam", None, window_main)

# External view context and camera
glfw.make_context_current(window_main)
glfw.swap_interval(1)
cam_main = mujoco.MjvCamera()
opt = mujoco.MjvOption()
scene_main = mujoco.MjvScene(model, maxgeom=10000)
context_main = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
# Default external camera settings
cam_main.distance = 1.2
cam_main.elevation = -20.0
cam_main.azimuth = 0.0
cam_main.lookat = np.array([0.0, 0.0, 0.3])

# Left wrist camera context
glfw.make_context_current(window_left)
cam_left = mujoco.MjvCamera()
scene_left = mujoco.MjvScene(model, maxgeom=10000)
context_left = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
cam_id_left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "left_wrist_cam")
cam_left.type = mujoco.mjtCamera.mjCAMERA_FIXED
cam_left.fixedcamid = cam_id_left

#######
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

#######

# Right wrist camera context
glfw.make_context_current(window_right)
cam_right = mujoco.MjvCamera()
scene_right = mujoco.MjvScene(model, maxgeom=10000)
context_right = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
cam_id_right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "right_wrist_cam")
cam_right.type = mujoco.mjtCamera.mjCAMERA_FIXED
cam_right.fixedcamid = cam_id_right

# Shared input callback on main window
def keyboard(window, key, scancode, action, mods):
    if action == glfw.PRESS:
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window_main, True)
            glfw.set_window_should_close(window_left, True)
            glfw.set_window_should_close(window_right, True)
        elif key == glfw.KEY_R:
            reset_position()
        elif key == glfw.KEY_H:
            ready_position()
        elif key == glfw.KEY_SPACE:
            box_lifting_task()

glfw.set_key_callback(window_main, keyboard)

# Render helper
def render(window, cam, scene, context, overlay=False):
    glfw.make_context_current(window)
    width, height = glfw.get_framebuffer_size(window)
    viewport = mujoco.MjrRect(0, 0, width, height)
    mujoco.mjv_updateScene(model, data, opt, None, cam,
                          mujoco.mjtCatBit.mjCAT_ALL.value, scene)
    mujoco.mjr_render(viewport, scene, context)
    if overlay:
        text = ["ESC: Exit", "SPACE: Start", "R: Reset", "H: Ready"]
        overlay_text = "\n".join(text)
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,
                           mujoco.mjtGridPos.mjGRID_TOPLEFT,
                           viewport, overlay_text, "", context)
    glfw.swap_buffers(window)

# Actuator indices
left_joints = [model.actuator(f"left_actuator_joint{i}").id for i in range(1,5)]
left_joints.append(model.actuator("left_actuator_gripper_joint").id)
right_joints = [model.actuator(f"right_actuator_joint{i}").id for i in range(1,5)]
right_joints.append(model.actuator("right_actuator_gripper_joint").id)

# Reset and ready poses
def reset_position():
    for idx, angle in zip(left_joints, [0.0, -0.3, 0.6, 0.0, 0.01]):
        data.ctrl[idx] = angle
    for idx, angle in zip(right_joints, [0.0, -0.3, 0.6, 0.0, 0.01]):
        data.ctrl[idx] = angle

def ready_position():
    for idx, angle in zip(left_joints, [0.0, -0.4, 0.8, 0.2, 0.015]):
        data.ctrl[idx] = angle
    for idx, angle in zip(right_joints, [0.0, -0.4, 0.8, 0.2, 0.015]):
        data.ctrl[idx] = angle

# Simulation step and box task
def simulate(duration):
    start_time = time.time()
    while time.time() - start_time < duration:
        mujoco.mj_step(model, data)
        render(window_main, cam_main, scene_main, context_main, overlay=False)
        render(window_left, cam_left, scene_left, context_left)
        render(window_right, cam_right, scene_right, context_right)
        glfw.poll_events()
        if glfw.window_should_close(window_main):
            break
        time.sleep(0.01)

def box_lifting_task():
    # 1. Move to box
    data.ctrl[left_joints[0]] = 0.0
    data.ctrl[left_joints[1]] = -0.6
    data.ctrl[left_joints[2]] = 1.0
    data.ctrl[left_joints[3]] = 0.3
    data.ctrl[left_joints[4]] = 0.015
    data.ctrl[right_joints[0]] = 0.0
    data.ctrl[right_joints[1]] = -0.6
    data.ctrl[right_joints[2]] = 1.0
    data.ctrl[right_joints[3]] = 0.3
    data.ctrl[right_joints[4]] = 0.015
    simulate(1.5)
    # 2. Grip
    data.ctrl[left_joints[4]] = -0.005
    data.ctrl[right_joints[4]] = -0.005
    simulate(1.0)
    # 3. Lift
    for i in range(15):
        data.ctrl[left_joints[1]] = -0.6 + i * 0.04
        data.ctrl[right_joints[1]] = -0.6 + i * 0.04
        simulate(0.1)
    simulate(1.5)
    # 4. Hand-off
    data.ctrl[left_joints[4]] = 0.015
    simulate(1.0)
    # 5. Left away
    data.ctrl[left_joints[2]] = 0.5
    data.ctrl[left_joints[1]] = -0.3
    simulate(1.0)
    # 6. Lower right
    for i in range(15):
        data.ctrl[right_joints[1]] = - i * 0.04
        simulate(0.1)
    # 7. Release
    data.ctrl[right_joints[4]] = 0.015
    simulate(1.0)
    # 8. Reset
    reset_position()
    simulate(2.0)

# Main loop
print("=== Starting Multi-View Simulation ===")
reset_position()
while not (glfw.window_should_close(window_main) 
           or glfw.window_should_close(window_left) 
           or glfw.window_should_close(window_right)):
    mujoco.mj_step(model, data)
    render(window_main, cam_main, scene_main, context_main, overlay=True)
    render(window_left, cam_left, scene_left, context_left)
    render(window_right, cam_right, scene_right, context_right)
    glfw.poll_events()
    time.sleep(0.01)

# Cleanup
glfw.terminate()