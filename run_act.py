#!/usr/bin/env python3
import numpy as np
import mujoco
from mujoco.glfw import glfw
import os

# 파일 경로
model_path = "scene.xml"
act_path   = "demo_#.act"

# 모델 로드
model = mujoco.MjModel.from_xml_path(model_path)
data  = mujoco.MjData(model)

# 액션 로드
ctrl = np.fromfile(act_path, dtype=np.float32).reshape(-1, model.nu)

# GLFW 초기화
glfw.init()
win = glfw.create_window(800, 600, "Replay", None, None)
glfw.make_context_current(win)
glfw.swap_interval(1)

# 렌더링 세팅
cam = mujoco.MjvCamera(); opt = mujoco.MjvOption()
scene = mujoco.MjvScene(model, maxgeom=10000)
ctx   = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
cam.distance, cam.elevation, cam.azimuth = 1.0, -20.0, 90.0
cam.lookat = np.array([0.2,0,0.2])

# 재생 루프
for u in ctrl:
    if glfw.window_should_close(win):
        break
    data.ctrl[:] = u
    mujoco.mj_step(model, data)
    w, h = glfw.get_framebuffer_size(win)
    vp = mujoco.MjrRect(0,0,w,h)
    mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene)
    mujoco.mjr_render(vp, scene, ctx)
    glfw.swap_buffers(win)
    glfw.poll_events()

glfw.terminate()
