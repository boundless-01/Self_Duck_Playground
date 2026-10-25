import mujoco
import mujoco.viewer
import numpy as np

XML_PATH = "playground/open_duck_mini_v2/xmls/scene_flat_terrain_roll.xml"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

paused = False

# ===== 可调参数 =====
BEND_ANGLE   = 1.2    # 正值：左髋向前折；右髋取相反数
KNEE_ANGLE   = 1.5    # 正值：收腿团身
ANKLE_COMP   = -1.2   # 团身时勾脚，避免脚掌把身体撑开
NECK_PITCH   = -0.3   # 负值：下巴向胸口低头
HEAD_PITCH   = -0.7   # 负值：头部继续向前下折
HEIGHT       = 0.22   # 身体高度（腿直立时的站立高度，需按模型调）

def set_bend_pose():
    """腿直立，从髋关节弯腰，低头。"""
    data.qpos[:] = model.keyframe("home").qpos  # 保留有效的基座四元数
    data.qpos[2] = HEIGHT                       # 高度

    # ---- 腿：按左右镜像的关节方向团身 ----
    # 左腿
    left_hip_addr = model.jnt_qposadr[model.joint("left_hip_pitch").id]
    left_knee_addr = model.jnt_qposadr[model.joint("left_knee").id]
    left_ankle_addr = model.jnt_qposadr[model.joint("left_ankle").id]
    right_hip_addr = model.jnt_qposadr[model.joint("right_hip_pitch").id]
    right_knee_addr = model.jnt_qposadr[model.joint("right_knee").id]
    right_ankle_addr = model.jnt_qposadr[model.joint("right_ankle").id]
    neck_addr = model.jnt_qposadr[model.joint("neck_pitch").id]
    head_addr = model.jnt_qposadr[model.joint("head_pitch").id]

    data.qpos[left_hip_addr] = BEND_ANGLE
    data.qpos[left_knee_addr] = KNEE_ANGLE
    data.qpos[left_ankle_addr] = ANKLE_COMP
    # 右腿
    data.qpos[right_hip_addr] = -BEND_ANGLE
    data.qpos[right_knee_addr] = KNEE_ANGLE
    data.qpos[right_ankle_addr] = ANKLE_COMP

    # ---- 头颈：低头 ----
    data.qpos[neck_addr] = NECK_PITCH
    data.qpos[head_addr] = HEAD_PITCH

    # Hold the displayed pose when simulation is resumed.
    data.ctrl[:] = model.keyframe("home").ctrl
    ctrl = lambda name: model.actuator(name).id
    data.ctrl[ctrl("left_hip_pitch")] = BEND_ANGLE
    data.ctrl[ctrl("left_knee")] = KNEE_ANGLE
    data.ctrl[ctrl("left_ankle")] = ANKLE_COMP
    data.ctrl[ctrl("neck_pitch")] = NECK_PITCH
    data.ctrl[ctrl("head_pitch")] = HEAD_PITCH
    data.ctrl[ctrl("right_hip_pitch")] = -BEND_ANGLE
    data.ctrl[ctrl("right_knee")] = KNEE_ANGLE
    data.ctrl[ctrl("right_ankle")] = ANKLE_COMP

    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    # 打印头部世界坐标，方便判断是否触地
    head_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "head_collision"
    )
    if head_id >= 0:
        print("head collision center:", data.geom_xpos[head_id])

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
