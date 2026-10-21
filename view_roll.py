import mujoco
import mujoco.viewer
import numpy as np

XML_PATH = "playground/open_duck_mini_v2/xmls/scene_flat_terrain_roll.xml"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

paused = False

# ===== 可调参数 =====
BEND_ANGLE   = 1.4    # 弯腰角度（髋关节），越大弯得越深
ANKLE_COMP   = -1.0   # 踝关节补偿，让脚掌保持贴地（符号要试）
NECK_PITCH   = -0.5   # 脖子低头
HEAD_PITCH   = -1.0   # 头再低
HEIGHT       = 0.22   # 身体高度（腿直立时的站立高度，需按模型调）

def set_bend_pose():
    """腿直立，从髋关节弯腰，低头。"""
    data.qpos[:] = 0        # 先归零，避免残留
    data.qpos[2] = HEIGHT   # 高度

    # ---- 腿：膝盖锁 0，保持直立 ----
    # 左腿
    data.qpos[9]  = BEND_ANGLE   # left_hip_pitch  -> 弯腰主力
    data.qpos[10] = 0.0          # left_knee       -> 直立
    data.qpos[11] = ANKLE_COMP   # left_ankle      -> 脚掌贴地补偿
    # 右腿
    data.qpos[18] = BEND_ANGLE   # right_hip_pitch
    data.qpos[19] = 0.0          # right_knee
    data.qpos[20] = ANKLE_COMP   # right_ankle

    # ---- 头颈：低头 ----
    data.qpos[12] = NECK_PITCH
    data.qpos[13] = HEAD_PITCH

    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    # 打印头部世界坐标，方便判断是否触地
    head_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head")
    if head_id >= 0:
        print("head pos:", data.xpos[head_id])

def key_callback(keycode):
    global paused
    if keycode == 32:      # 空格
        paused = not paused
        print(f"paused = {paused}")
    elif keycode == 84:    # 't'
        set_bend_pose()
        paused = True
    elif keycode == 265:   # 上箭头：加大弯腰
        global BEND_ANGLE
        BEND_ANGLE += 0.1
        print("BEND_ANGLE =", BEND_ANGLE)
        set_bend_pose()
    elif keycode == 264:   # 下箭头：减小弯腰
        BEND_ANGLE -= 0.1
        print("BEND_ANGLE =", BEND_ANGLE)
        set_bend_pose()

with mujoco.viewer.launch_passive(
    model, data, key_callback=key_callback,
    show_left_ui=True, show_right_ui=True,
) as viewer:
    while viewer.is_running():
        if not paused:
            mujoco.mj_step(model, data)
        viewer.sync()
