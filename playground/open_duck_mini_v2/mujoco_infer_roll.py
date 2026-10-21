# playground/open_duck_mini_v2/mujoco_infer_roll.py
"""
MuJoCo viewer / inference environment for the Open Duck Mini V2 "roll" task.

Mirrors playground/open_duck_mini_v2/roll.py (the MJX training env).

NOTE: the 52.onnx policy expects a 92-dim observation, i.e. the FULL
14-actuator model (including the 4 head joints). We therefore use all 14
actuators directly, matching roll.py's `self._actuators = self.mjx_model.nu`.
"""

import argparse
import pickle
import time

import numpy as np
import mujoco
import mujoco.viewer

from playground.common.onnx_infer import OnnxInfer
from playground.open_duck_mini_v2.mujoco_infer_base import MJInferBase


USE_MOTOR_SPEED_LIMITS = True

# Must match roll.py
ROLL_AXIS = 1  # 0 = x (side roll), 1 = y (forward roll)


class MjInferRoll(MJInferBase):
    """Replay a trained roll policy inside the MuJoCo viewer."""

    def __init__(
        self,
        model_path: str,
        onnx_model_path: str,
    ):
        super().__init__(model_path)

        # ------------------------------------------------------------------
        # Params (must match roll.py's default_config)
        # ------------------------------------------------------------------
        self.action_scale = 0.25
        self.dof_vel_scale = 0.05
        self.max_motor_velocity = 5.24  # rad/s

        # 14 actuators (including head)
        self.num_dofs = self.model.nu  # 14
        print(f"[roll] using {self.num_dofs} actuators: {self.actuator_names}")

        # ------------------------------------------------------------------
        # Policy
        # ------------------------------------------------------------------
        self.policy = OnnxInfer(onnx_model_path, awd=True)

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------
        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)

        self.motor_targets = np.asarray(self.default_actuator).copy()
        self.prev_motor_targets = np.asarray(self.default_actuator).copy()

        self.saved_obs = []
        self.max_roll = 0.0

    # ----------------------------------------------------------------------
    # Observation  (mirrors roll.py::_get_obs "state" vector, 92 dims)
    # ----------------------------------------------------------------------
    def get_obs(self, data) -> np.ndarray:
        gyro = self.get_gyro(data)                          # 3
        accelerometer = self.get_accelerometer(data)        # 3

        joint_angles = self.get_actuator_joints_qpos(data.qpos)  # 14
        joint_vel = self.get_actuator_joints_qvel(data.qvel)     # 14
        contacts = np.asarray(self.get_feet_contacts(data), dtype=np.float32)  # 2

        obs = np.concatenate(
            [
                gyro,                                          # 3
                accelerometer,                                 # 3
                joint_angles - self.default_actuator,          # 14
                joint_vel * self.dof_vel_scale,                # 14
                self.last_action,                              # 14
                self.last_last_action,                         # 14
                self.last_last_last_action,                    # 14
                self.motor_targets,                            # 14
                contacts,                                      # 2
            ]
        ).astype(np.float32)

        assert obs.shape == (92,), f"obs shape {obs.shape} != (92,)"
        return obs

    # ----------------------------------------------------------------------
    # Key callback
    # ----------------------------------------------------------------------
    def key_callback(self, keycode):
        print(f"key: {keycode}")
        if keycode == 82:  # r : reset roll bookkeeping
            self.max_roll = 0.0
            print("reset roll bookkeeping")

    def get_gravity(self, data):
        """直接从 imu site 的旋转矩阵算 z 轴，绕过传感器布局问题。"""
        imu_site_id = self.model.site("imu").id
        return data.site_xmat[imu_site_id].reshape(3, 3)[:, 2]
    
    # ----------------------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------------------
    def run(self):
        try:
            with mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
                key_callback=self.key_callback,
            ) as viewer:
                counter = 0
                while True:
                    step_start = time.time()

                    mujoco.mj_step(self.model, self.data)
                    counter += 1

                    if counter % self.decimation == 0:
                        obs = self.get_obs(self.data)
                        self.saved_obs.append(obs)

                        action = np.asarray(self.policy.infer(obs)).reshape(
                            self.num_dofs
                        )

                        if counter % (self.decimation * 50) == 0:
                            gravity = self.get_gravity(self.data)
                            gyro = self.get_gyro(self.data)
                            qpos_base = self.data.qpos[:3]
                            roll_progress = abs(float(gravity[ROLL_AXIS]))

                            floor_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
                            ncon = self.data.ncon
                            floor_contacts = 0
                            for i in range(ncon):
                                c = self.data.contact[i]
                                if c.geom1 == floor_gid or c.geom2 == floor_gid:
                                    floor_contacts += 1

                            print(
                                f"[{counter:6d}] "
                                f"base_z={qpos_base[2]:+.3f} "
                                f"gravity={np.round(gravity, 2)} "
                                f"gyro={np.round(gyro, 2)} "
                                f"roll_prog={roll_progress:.2f} "
                                f"max_roll={self.max_roll:.2f} "
                                f"ncon={ncon} floor_con={floor_contacts} "
                                f"|act|={np.linalg.norm(action):.2f}"
                            )

                            # ---- Debug: imu site 旋转矩阵 + base 四元数 ------------------
                            imu_site_id = self.model.site("imu").id
                            imu_xmat = self.data.site_xmat[imu_site_id].reshape(3, 3)
                            base_quat = self.data.qpos[3:7]
                            quat_norm = np.linalg.norm(base_quat)

                            # 从旋转矩阵提取 z 轴（应该等于 upvector）
                            imu_zaxis = imu_xmat[:, 2]

                            print(
                                f"    imu_xmat z-axis: {np.round(imu_zaxis, 3)} "
                                f"(norm={np.linalg.norm(imu_zaxis):.3f})"
                            )
                            print(
                                f"    base_quat: {np.round(base_quat, 3)} "
                                f"(norm={quat_norm:.4f})"
                            )

                            print(f"gravity_id: {self.gravity_id}, dim: {self.gravity_dimensions}")

                            # 看看 upvector 实际在哪个地址
                            upvector_id = self.model.sensor("upvector").id
                            print(f"upvector sensor id: {upvector_id}")
                            print(f"upvector sensor adr: {self.model.sensor_adr[upvector_id]}")
                            print(f"upvector sensor dim: {self.model.sensor_dim[upvector_id]}")

                        # Shift action history
                        self.last_last_last_action = self.last_last_action.copy()
                        self.last_last_action = self.last_action.copy()
                        self.last_action = action.copy()

                        self.motor_targets = (
                            self.default_actuator + action * self.action_scale
                        )

                        if USE_MOTOR_SPEED_LIMITS:
                            dt = self.sim_dt * self.decimation
                            self.motor_targets = np.clip(
                                self.motor_targets,
                                self.prev_motor_targets - self.max_motor_velocity * dt,
                                self.prev_motor_targets + self.max_motor_velocity * dt,
                            )
                            self.prev_motor_targets = self.motor_targets.copy()

                        self.data.ctrl[:] = self.motor_targets

                        # ---- Debug: roll progress ------------------------
                        gravity = self.get_gravity(self.data)
                        roll_progress = abs(float(gravity[ROLL_AXIS]))
                        if roll_progress > self.max_roll:
                            self.max_roll = roll_progress

                    viewer.sync()

                    time_until_next_step = self.model.opt.timestep - (
                        time.time() - step_start
                    )
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)

        except KeyboardInterrupt:
            pickle.dump(
                self.saved_obs, open("mujoco_saved_obs_roll.pkl", "wb")
            )
            print(f"\nmax roll progress: {self.max_roll:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
    parser.add_argument(
        "--model_path",
        type=str,
        default="playground/open_duck_mini_v2/xmls/scene_flat_terrain_roll.xml",
    )

    args = parser.parse_args()

    mjinfer = MjInferRoll(args.model_path, args.onnx_model_path)
    mjinfer.run()