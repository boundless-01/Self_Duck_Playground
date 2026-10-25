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
from playground.open_duck_mini_v2.roll import (
    FLIP_DIRECTION,
    default_config as roll_default_config,
)


USE_MOTOR_SPEED_LIMITS = True

# Must match roll.py
ROLL_AXIS = 1  # 0 = x (side roll), 1 = y (forward roll)


class MjInferRoll(MJInferBase):
    """Replay a trained roll policy inside the MuJoCo viewer."""

    def __init__(
        self,
        model_path: str,
        onnx_model_path: str | None,
        roll_phase: int = 0,
        scripted_only: bool = False,
    ):
        super().__init__(model_path)

        # ------------------------------------------------------------------
        # Params (must match roll.py's default_config)
        # ------------------------------------------------------------------
        roll_config = roll_default_config()
        self.action_scale = roll_config.action_scale
        self.pre_head_action_scale = roll_config.pre_head_action_scale
        self.dof_vel_scale = roll_config.dof_vel_scale
        self.max_motor_velocity = roll_config.max_motor_velocity
        self.control_dt = roll_config.ctrl_dt
        self.episode_length = roll_config.episode_length
        self.head_pitch_lower = np.array(
            [
                self.model.jnt_range[
                    self.actuator_joint_dict["neck_pitch"]
                ][0],
                self.model.jnt_range[
                    self.actuator_joint_dict["head_pitch"]
                ][0],
            ]
        )
        self.release_com_tolerance = (
            roll_config.reward_config.head_release_com_tolerance
        )
        self.release_up_z = roll_config.reward_config.head_release_up_z
        self.release_knee = roll_config.reward_config.head_release_knee
        self.heel_support_steps = roll_config.reward_config.heel_support_steps
        self.recovery_roll_fraction = (
            roll_config.reward_config.recovery_roll_fraction
        )
        self.deep_fold_neck = roll_config.reward_config.deep_fold_neck
        self.deep_fold_head = roll_config.reward_config.deep_fold_head
        self.decimation = int(round(self.control_dt / self.sim_dt))
        self.obs_size = 99
        self.roll_phase = roll_phase
        self.scripted_only = scripted_only
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
        self.roll_action_mask = np.array(
            [name in controlled_joints for name in self.actuator_names],
            dtype=bool,
        )
        recovery_joints = controlled_joints | {
            "left_hip_roll",
            "right_hip_roll",
            "left_hip_yaw",
            "right_hip_yaw",
        }
        self.recovery_action_mask = np.array(
            [name in recovery_joints for name in self.actuator_names],
            dtype=bool,
        )
        self.launch_crouch_duration = 0.10
        self.launch_push_end = 0.30
        self.launch_crouch_targets = np.asarray(self.default_actuator).copy()
        self.launch_crouch_targets[
            [self.actuator_names.index("left_hip_pitch"),
             self.actuator_names.index("right_hip_pitch")]
        ] = [1.20, -1.20]
        self.launch_crouch_targets[
            [self.actuator_names.index("left_knee"),
             self.actuator_names.index("right_knee")]
        ] = [0.80, 0.80]
        self.launch_crouch_targets[
            [self.actuator_names.index("left_ankle"),
             self.actuator_names.index("right_ankle")]
        ] = [0.0, 0.0]
        self.launch_push_targets = np.asarray(self.default_actuator).copy()
        self.launch_push_targets[
            [self.actuator_names.index("left_hip_pitch"),
             self.actuator_names.index("right_hip_pitch")]
        ] = [1.20, -1.20]
        self.launch_push_targets[
            [self.actuator_names.index("left_knee"),
             self.actuator_names.index("right_knee")]
        ] = [0.0, 0.0]
        self.launch_push_targets[
            [self.actuator_names.index("left_ankle"),
             self.actuator_names.index("right_ankle")]
        ] = [-1.20, -1.20]

        # 14 actuators (including head)
        self.num_dofs = self.model.nu  # 14
        print(
            f"[roll] phase={self.roll_phase}, "
            f"using {self.num_dofs} actuators: {self.actuator_names}"
        )

        # ------------------------------------------------------------------
        # Policy
        # ------------------------------------------------------------------
        self.policy = None
        if not self.scripted_only:
            if onnx_model_path is None:
                raise ValueError("--onnx_model_path is required unless using --scripted_only")
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
            self.data.qvel[2] = 1.5
            self.data.qvel[4] = 8.0
        self.data.ctrl[:] = self.default_actuator
        mujoco.mj_forward(self.model, self.data)

        self.last_action[:] = 0.0
        self.last_last_action[:] = 0.0
        self.last_last_last_action[:] = 0.0
        self.motor_targets[:] = self.default_actuator
        self.prev_motor_targets[:] = self.default_actuator
        self.max_roll = 0.0
        imu_site_id = self.model.site("imu").id
        # site_xmat is a view into MjData. Copy it at reset, otherwise the
        # "initial" forward direction rotates with the robot and both roll
        # progress and the COM/head test become meaningless.
        self.initial_forward = np.array(
            self.data.site_xmat[imu_site_id].reshape(3, 3)[:, 0],
            copy=True,
        )
        self.forward_xy = self.initial_forward.copy()
        self.forward_xy[2] = 0.0
        self.forward_xy /= max(np.linalg.norm(self.forward_xy), 1e-6)
        self.last_roll_angle = 0.0
        self.accumulated_roll = 0.0
        self.control_step = 0
        self.head_contacted = False
        self.head_contact_steps = 0
        self.release_ready = False
        self.knees_popped = False
        self.hips_released = False
        self.inverted = False
        self.recovery_ready = False
        self.back_contacted = False
        self.heel_contact_steps = 0
        self.heel_planted = False

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
        self.current_contacts = contacts
        head_contact = np.float32(self.check_contact(data, "head_assembly", "floor"))
        q = joint_angles
        neck_i = self.actuator_names.index("neck_pitch")
        head_i = self.actuator_names.index("head_pitch")
        left_hip_i = self.actuator_names.index("left_hip_pitch")
        right_hip_i = self.actuator_names.index("right_hip_pitch")
        left_knee_i = self.actuator_names.index("left_knee")
        right_knee_i = self.actuator_names.index("right_knee")
        was_head_contacted = self.head_contacted
        was_release_ready = self.release_ready
        was_knees_popped = self.knees_popped
        was_hips_released = self.hips_released
        deep_fold = (
            q[neck_i] <= self.deep_fold_neck
            and q[head_i] <= self.deep_fold_head
            and q[left_hip_i] >= 0.60
            and q[right_hip_i] <= -0.60
        )
        deep_head_contact = bool(head_contact) and deep_fold
        if deep_head_contact:
            self.head_contact_steps += 1
        self.head_contacted = self.head_contacted or deep_head_contact
        head_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "head_collision"
        )
        trunk_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk_assembly"
        )
        com_offset = data.subtree_com[trunk_body_id] - data.geom_xpos[head_geom_id]
        com_offset[2] = 0.0
        com_forward_of_head = float(np.dot(com_offset, self.forward_xy))
        imu_site_id = self.model.site("imu").id
        up_z = float(data.site_xmat[imu_site_id].reshape(3, 3)[2, 2])
        release_ready_now = self.head_contacted and (
            com_forward_of_head >= self.release_com_tolerance
            and up_z <= self.release_up_z
            and q[left_knee_i] >= self.release_knee
            and q[right_knee_i] >= self.release_knee
        )
        self.release_ready = self.release_ready or release_ready_now
        if (
            self.release_ready
            and q[left_knee_i] <= 0.2
            and q[right_knee_i] <= 0.2
        ):
            self.knees_popped = True
        if (
            self.knees_popped
            and abs(q[left_hip_i]) <= 0.2
            and abs(q[right_hip_i]) <= 0.2
        ):
            self.hips_released = True
        heel_support_now = (
            self.hips_released
            and self.back_contacted
            and bool(np.any(contacts))
        )
        self.heel_contact_steps = (
            min(self.heel_contact_steps + 1, 100)
            if heel_support_now
            else 0
        )
        if self.heel_contact_steps >= self.heel_support_steps:
            self.heel_planted = True
        if self.head_contacted and not was_head_contacted:
            print(
                f"[roll event] head contact t={self.control_step * self.control_dt:.2f}s "
                f"COM-head={com_forward_of_head:+.3f}m"
            )
        if self.release_ready and not was_release_ready:
            print(
                f"[roll event] release t={self.control_step * self.control_dt:.2f}s "
                f"COM-head={com_forward_of_head:+.3f}m up_z={up_z:+.3f}"
            )
        if self.knees_popped and not was_knees_popped:
            print(
                f"[roll event] knees extended t={self.control_step * self.control_dt:.2f}s"
            )
        if self.hips_released and not was_hips_released:
            print(
                f"[roll event] hips released t={self.control_step * self.control_dt:.2f}s"
            )
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
        self.accumulated_roll = max(
            self.accumulated_roll + FLIP_DIRECTION * angle_delta, 0.0
        )
        if self.accumulated_roll >= self.recovery_roll_fraction * 2.0 * np.pi:
            if not self.recovery_ready:
                print(
                    f"[roll event] recovery enabled t="
                    f"{self.control_step * self.control_dt:.2f}s"
                )
            self.recovery_ready = True
        if self.head_contacted and self.release_ready and up_world[2] < -0.5:
            if not self.inverted:
                print(
                    f"[roll event] inverted t={self.control_step * self.control_dt:.2f}s"
                )
            self.inverted = True
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

                        if self.scripted_only:
                            action = np.zeros(self.num_dofs, dtype=np.float32)
                        else:
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

                        reference_targets = np.asarray(
                            self.default_actuator
                        ).copy()
                        neck_i = self.actuator_names.index("neck_pitch")
                        head_i = self.actuator_names.index("head_pitch")
                        left_hip_i = self.actuator_names.index("left_hip_pitch")
                        right_hip_i = self.actuator_names.index("right_hip_pitch")
                        left_knee_i = self.actuator_names.index("left_knee")
                        right_knee_i = self.actuator_names.index("right_knee")
                        left_ankle_i = self.actuator_names.index("left_ankle")
                        right_ankle_i = self.actuator_names.index("right_ankle")
                        if self.recovery_ready:
                            reference_targets = np.asarray(
                                self.default_actuator
                            ).copy()
                            if not np.all(self.current_contacts):
                                reference_targets[
                                    [left_hip_i, right_hip_i, left_knee_i,
                                     right_knee_i, left_ankle_i, right_ankle_i]
                                ] = [-0.8, 0.8, 0.3, 0.3, 1.0, 1.0]
                        elif not self.head_contacted:
                            launch_time = self.control_step * self.control_dt
                            if launch_time < self.launch_crouch_duration:
                                reference_targets = self.launch_crouch_targets.copy()
                            elif launch_time < self.launch_push_end:
                                reference_targets = self.launch_push_targets.copy()
                            else:
                                reference_targets[
                                    [neck_i, head_i, left_hip_i, right_hip_i,
                                     left_knee_i, right_knee_i,
                                     left_ankle_i, right_ankle_i]
                                ] = [-0.349, -0.785, 1.22, -1.22,
                                     1.50, 1.50, -1.20, -1.20]
                        elif not self.release_ready:
                            # Keep the compact tuck while the head is the pivot.
                            reference_targets[
                                [neck_i, head_i, left_hip_i, right_hip_i,
                                 left_knee_i, right_knee_i,
                                 left_ankle_i, right_ankle_i]
                            ] = [-0.349, -0.785, 1.22, -1.22,
                                 1.50, 1.50, -1.20, -1.20]
                        elif not self.knees_popped:
                            # Extend knees toward zero while holding the hip fold.
                            reference_targets[
                                [left_hip_i, right_hip_i, left_knee_i,
                                 right_knee_i, left_ankle_i, right_ankle_i]
                            ] = [1.22, -1.22, 0.0, 0.0, -1.2, -1.2]
                        elif not self.hips_released:
                            reference_targets[
                                [left_hip_i, right_hip_i, left_knee_i,
                                 right_knee_i, left_ankle_i, right_ankle_i]
                            ] = [0.0, 0.0, 0.0, 0.0, -1.2, -1.2]
                        elif not self.heel_planted:
                            # Swing the legs back so the heels can find the floor.
                            reference_targets[
                                [left_hip_i, right_hip_i, left_knee_i,
                                 right_knee_i, left_ankle_i, right_ankle_i]
                            ] = [-0.80, 0.80, 0.30, 0.30, 1.0, 1.0]
                        else:
                            # Recover to the home stance after the heels support
                            # the body.
                            reference_targets = np.asarray(
                                self.default_actuator
                            ).copy()

                        residual_scale = (
                            self.action_scale
                            if self.head_contacted
                            else self.pre_head_action_scale
                        )
                        active_action_mask = (
                            self.recovery_action_mask
                            if self.recovery_ready
                            else self.roll_action_mask
                        )
                        self.motor_targets = reference_targets + (
                            action * residual_scale * active_action_mask
                        )
                        self.motor_targets = np.where(
                            active_action_mask,
                            self.motor_targets,
                            self.default_actuator,
                        )
                        self.motor_targets[neck_i] = np.clip(
                            self.motor_targets[neck_i],
                            self.head_pitch_lower[0],
                            0.0,
                        )
                        self.motor_targets[head_i] = np.clip(
                            self.motor_targets[head_i],
                            self.head_pitch_lower[1],
                            0.0,
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
    parser.add_argument("-o", "--onnx_model_path", type=str)
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
    parser.add_argument(
        "--scripted_only",
        action="store_true",
        help="Run the phase reference without loading or applying an ONNX policy.",
    )

    args = parser.parse_args()

    mjinfer = MjInferRoll(
        args.model_path,
        args.onnx_model_path,
        roll_phase=args.roll_phase,
        scripted_only=args.scripted_only,
    )
    mjinfer.run()
