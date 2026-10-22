# playground/open_duck_mini_v2/mujoco_infer_roll.py
"""
MuJoCo viewer / inference environment for the Open Duck Mini V2 "roll" task.

Mirrors playground/open_duck_mini_v2/roll.py (the MJX training env).

The policy expects a 99-dim observation: 95 proprioceptive values plus roll
phase, episode time, and head/back ground contact.
"""

import argparse
import pickle
import time

import numpy as np
import mujoco
import mujoco.viewer

from playground.common.onnx_infer import OnnxInfer
from playground.open_duck_mini_v2.mujoco_infer_base import MJInferBase
from playground.open_duck_mini_v2.roll import default_config as roll_default_config


USE_MOTOR_SPEED_LIMITS = True

# Must match roll.py
ROLL_AXIS = 1  # 0 = x (side roll), 1 = y (forward roll)


class MjInferRoll(MJInferBase):
    """Replay a trained roll policy inside the MuJoCo viewer."""

    def __init__(
        self,
        model_path: str,
        onnx_model_path: str,
        roll_phase: int = 0,
    ):
        super().__init__(model_path)

        # ------------------------------------------------------------------
        # Params (must match roll.py's default_config)
        # ------------------------------------------------------------------
        roll_config = roll_default_config()
        self.action_scale = roll_config.action_scale
        self.dof_vel_scale = roll_config.dof_vel_scale
        self.max_motor_velocity = roll_config.max_motor_velocity
        self.control_dt = roll_config.ctrl_dt
        self.episode_length = roll_config.episode_length
        self.decimation = int(round(self.control_dt / self.sim_dt))
        self.obs_size = 99
        self.roll_phase = roll_phase
        controlled_joints = {
            "left_hip_pitch",
            "left_knee",
            "left_ankle",
            "right_hip_pitch",
            "right_knee",
            "right_ankle",
            "neck_pitch",
            "head_pitch",
        }
        self.action_mask = np.array(
            [name in controlled_joints for name in self.actuator_names],
            dtype=bool,
        )

        # 14 actuators (including head)
        self.num_dofs = self.model.nu  # 14
        print(
            f"[roll] phase={self.roll_phase}, "
            f"using {self.num_dofs} actuators: {self.actuator_names}"
        )

        # ------------------------------------------------------------------
        # Policy
        # ------------------------------------------------------------------
        self.policy = OnnxInfer(onnx_model_path, awd=True)
        policy_input_shape = self.policy.ort_session.get_inputs()[0].shape
        if policy_input_shape[-1] != self.obs_size:
            raise ValueError(
                "This roll policy expects "
                f"{policy_input_shape[-1]} observations, but the updated roll "
                f"environment uses {self.obs_size}. Retrain/export the policy "
                "before running the viewer."
            )

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------
        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)

        self.motor_targets = np.asarray(self.default_actuator).copy()
        self.prev_motor_targets = np.asarray(self.default_actuator).copy()

        self.saved_obs = []
        self.reset_episode()

    def reset_episode(self):
        """Reset simulation and all policy history to the training start."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.model.keyframe("home").qpos
        self.data.qvel[:] = 0
        if self.roll_phase == 0:
            self.data.qvel[2] = 0.0
            self.data.qvel[4] = 3.0
        self.data.ctrl[:] = self.default_actuator
        mujoco.mj_forward(self.model, self.data)

        self.last_action[:] = 0.0
        self.last_last_action[:] = 0.0
        self.last_last_last_action[:] = 0.0
        self.motor_targets[:] = self.default_actuator
        self.prev_motor_targets[:] = self.default_actuator
        self.max_roll = 0.0
        imu_site_id = self.model.site("imu").id
        self.initial_forward = self.data.site_xmat[imu_site_id].reshape(3, 3)[:, 0]
        self.last_roll_angle = 0.0
        self.accumulated_roll = 0.0
        self.control_step = 0
        self.head_contacted = False
        self.back_contacted = False

    # ----------------------------------------------------------------------
    # Observation  (mirrors roll.py::_get_obs "state" vector, 99 dims)
    # ----------------------------------------------------------------------
    def get_obs(self, data) -> np.ndarray:
        gyro = self.get_gyro(data)                          # 3
        accelerometer = self.get_accelerometer(data)        # 3
        gravity = self.get_gravity(data)                    # 3

        joint_angles = self.get_actuator_joints_qpos(data.qpos)  # 14
        joint_vel = self.get_actuator_joints_qvel(data.qvel)     # 14
        contacts = np.asarray(self.get_feet_contacts(data), dtype=np.float32)  # 2
        head_contact = np.float32(self.check_contact(data, "head_assembly", "floor"))
        self.head_contacted = self.head_contacted or bool(head_contact)
        back_contact = np.float32(self.check_contact(data, "trunk_assembly", "floor"))
        self.back_contacted = self.back_contacted or bool(back_contact)

        obs = np.concatenate(
            [
                gyro,                                          # 3
                accelerometer,                                 # 3
                gravity,                                       # 3
                joint_angles - self.default_actuator,          # 14
                joint_vel * self.dof_vel_scale,                # 14
                self.last_action,                              # 14
                self.last_last_action,                         # 14
                self.last_last_last_action,                    # 14
                self.motor_targets,                            # 14
                contacts,                                      # 2
                np.array(
                    [min(self.accumulated_roll / (2.0 * np.pi), 2.0)],
                    dtype=np.float32,
                ),                                             # 1
                np.array(
                    [min(self.control_step / self.episode_length, 1.0)],
                    dtype=np.float32,
                ),                                             # 1
                np.array([head_contact], dtype=np.float32),     # 1
                np.array([back_contact], dtype=np.float32),     # 1
            ]
        ).astype(np.float32)

        assert obs.shape == (self.obs_size,), (
            f"obs shape {obs.shape} != ({self.obs_size},)"
        )
        return obs

    # ----------------------------------------------------------------------
    # Key callback
    # ----------------------------------------------------------------------
    def key_callback(self, keycode):
        print(f"key: {keycode}")
        if keycode == 82:  # r : reset the episode
            self.reset_episode()
            print("reset roll bookkeeping")

    def get_gravity(self, data):
        """Gravity direction expressed in the IMU/body frame."""
        imu_site_id = self.model.site("imu").id
        rotation = data.site_xmat[imu_site_id].reshape(3, 3)
        return rotation.T @ np.array([0.0, 0.0, -1.0])

    def update_roll_progress(self, data) -> float:
        """Update the unwrapped forward inclination and return its fraction."""
        imu_site_id = self.model.site("imu").id
        up_world = data.site_xmat[imu_site_id].reshape(3, 3)[:, 2]
        roll_angle = np.arctan2(
            np.dot(up_world, self.initial_forward),
            up_world[2],
        )
        angle_delta = np.arctan2(
            np.sin(roll_angle - self.last_roll_angle),
            np.cos(roll_angle - self.last_roll_angle),
        )
        self.last_roll_angle = roll_angle
        self.accumulated_roll = max(self.accumulated_roll + angle_delta, 0.0)
        return self.accumulated_roll / (2.0 * np.pi)
    
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
                    if counter % self.decimation == 0:
                        if self.control_step >= self.episode_length:
                            self.reset_episode()
                        roll_progress = self.update_roll_progress(self.data)
                        obs = self.get_obs(self.data)
                        self.saved_obs.append(obs)

                        action = np.asarray(self.policy.infer(obs)).reshape(
                            self.num_dofs
                        )

                        if roll_progress > self.max_roll:
                            self.max_roll = roll_progress

                        if counter % (self.decimation * 50) == 0:
                            gravity = self.get_gravity(self.data)
                            gyro = self.get_gyro(self.data)
                            qpos_base = self.data.qpos[:3]

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
                                f"roll_angle={np.degrees(self.accumulated_roll):.0f}deg "
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
                        self.motor_targets = np.where(
                            self.action_mask,
                            self.motor_targets,
                            self.default_actuator,
                        )
                        neck_i = self.actuator_names.index("neck_pitch")
                        head_i = self.actuator_names.index("head_pitch")
                        self.motor_targets[neck_i] = np.clip(
                            self.motor_targets[neck_i], -0.349, 0.0
                        )
                        self.motor_targets[head_i] = np.clip(
                            self.motor_targets[head_i], -0.785, 0.0
                        )

                        if USE_MOTOR_SPEED_LIMITS:
                            dt = self.control_dt
                            self.motor_targets = np.clip(
                                self.motor_targets,
                                self.prev_motor_targets - self.max_motor_velocity * dt,
                                self.prev_motor_targets + self.max_motor_velocity * dt,
                            )
                            self.prev_motor_targets = self.motor_targets.copy()

                        self.data.ctrl[:] = self.motor_targets
                        self.control_step += 1

                    mujoco.mj_step(self.model, self.data)
                    counter += 1
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
    parser.add_argument(
        "--roll_phase",
        type=int,
        choices=(0, 1),
        default=0,
        help="Use 0 for assisted phase-0 checkpoints; 1 for from-rest models.",
    )

    args = parser.parse_args()

    mjinfer = MjInferRoll(
        args.model_path,
        args.onnx_model_path,
        roll_phase=args.roll_phase,
    )
    mjinfer.run()
