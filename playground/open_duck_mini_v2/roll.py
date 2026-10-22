# Copyright 2025 DeepMind Technologies Limited
# Copyright 2025 Antoine Pirrone - Steve Nguyen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Roll (forward flip) task for Open Duck Mini V2.

This is a standalone task. It does NOT use the imitation reward or the
reference motion generator. The policy is expected to discover a roll purely
from reward shaping.

Training is meant to be done in two phases (controlled by ROLL_PHASE):
  phase 0: just get the body to rotate 360 deg, no termination on orientation.
  phase 1: add landing / recovery terms and re-enable fall termination.
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math

from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding

from . import constants
from . import base as open_duck_mini_v2_base
from playground.common.rewards import (
    cost_non_roll_rotation,
    cost_roll_base_height,
    cost_torques,
    cost_action_rate,
    reward_alive,
)


# ---------------------------------------------------------------------------
# Phase switch. 0 = learn to flip, 1 = learn to land.
# Flip this to 1 only AFTER the policy reliably completes a 360 deg rotation.
# ---------------------------------------------------------------------------
ROLL_PHASE = 0

# Which body axis to roll around. 0 = x (side roll), 1 = y (forward roll).
ROLL_AXIS = 1
# Right-handed robot frame: +X forward, +Y left, +Z up.
# A positive rotation about +Y pitches the +X/front direction downward.
FLIP_DIRECTION = 1.0
ROLL_SUCCESS_ANGLE = 2.0 * np.pi

USE_MOTOR_SPEED_LIMITS = True


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        # A roll plus landing should fit in 10s. Long episodes make repeated
        # ground contact jitter a viable local optimum.
        episode_length=500,
        action_repeat=1,
        # The policy output is bounded to [-1, 1]. A 0.25 scale only permits
        # +/-0.25 rad around home and cannot reach the tuck pose.
        action_scale=1.8,
        dof_vel_scale=0.05,
        history_len=0,
        soft_joint_pos_limit_factor=0.95,
        max_motor_velocity=5.24,  # rad/s
        noise_config=config_dict.create(
            # Start without injected observation noise or delays. Re-enable
            # gradually after the basic roll is learned.
            level=0.0,
            action_min_delay=0,
            action_max_delay=1,  # randint upper bound is exclusive: delay=0 only
            imu_min_delay=0,
            imu_max_delay=1,     # delay=0 only
            scales=config_dict.create(
                hip_pos=0.03,
                knee_pos=0.05,
                ankle_pos=0.08,
                joint_vel=2.5,
                gravity=0.1,
                linvel=0.1,
                gyro=0.1,
                accelerometer=0.05,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                # Progress reward is based on the *change* in normalized
                # forward-roll angle, not the current angle every timestep.
                roll_velocity=1.0,
                non_roll_rotation=-0.5,
                roll_progress=0.0,       # Disable gravity-based half-flip proxy.
                roll_progress_delta=8.0,
                roll_success=20.0,       # One-time terminal achievement bonus.
                base_height=-0.1,
                torques=-1.0e-4,
                action_rate=-0.05,
                alive=0.05,
                stagnation=0.0,
                roll_accumulated=0.0,    # Disable repeated reward for current total.
                side_roll_penalty=-0.5,
                tuck_pose=0.5,
                # Shaping only after one full turn; also useful in phase 0.
                landing=2.0,
            ),
            roll_target_vel=6.0,
            roll_sigma=4.0,
            tuck_height=0.12,  # metres, crouched base height
        ),
        push_config=config_dict.create(
            enable=False,
            interval_range=[5.0, 10.0],
            magnitude_range=[0.1, 1.0],
        ),
    )


class Roll(open_duck_mini_v2_base.OpenDuckMiniV2Env):
    """Learn to roll over and (optionally) recover."""

    def __init__(
        self,
        task: str = "flat_terrain_roll",
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            xml_path=constants.task_to_xml(task).as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init()

    def _post_init(self) -> None:
        self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
        self._default_actuator = self._mj_model.keyframe("home").ctrl

        self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
        c = (self._lowers + self._uppers) / 2
        r = self._uppers - self._lowers
        self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
        self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

        self._njoints = self._mj_model.njnt
        self._actuators = self._mj_model.nu

        self._torso_body_id = self._mj_model.body(constants.ROOT_BODY).id
        self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
        self._site_id = self._mj_model.site("imu").id

        self._feet_site_id = np.array(
            [self._mj_model.site(name).id for name in constants.FEET_SITES]
        )
        self._floor_geom_id = self._mj_model.geom("floor").id
        self._feet_geom_id = np.array(
            [self._mj_model.geom(name).id for name in constants.FEET_GEOMS]
        )

        foot_linvel_sensor_adr = []
        for site in constants.FEET_SITES:
            sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
            sensor_adr = self._mj_model.sensor_adr[sensor_id]
            sensor_dim = self._mj_model.sensor_dim[sensor_id]
            foot_linvel_sensor_adr.append(
                list(range(sensor_adr, sensor_adr + sensor_dim))
            )
        self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

        qpos_noise_scale = np.zeros(self._actuators)
        for i, joint_name in enumerate(self.actuator_names):
            if "_hip_" in joint_name:
                qpos_noise_scale[i] = self._config.noise_config.scales.hip_pos
            elif "_knee" in joint_name:
                qpos_noise_scale[i] = self._config.noise_config.scales.knee_pos
            elif "_ankle" in joint_name:
                qpos_noise_scale[i] = self._config.noise_config.scales.ankle_pos
        self._qpos_noise_scale = jp.array(qpos_noise_scale)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)

        # Place robot near origin with small noise.
        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.05, maxval=0.05)

        base_qpos = self.get_floating_base_qpos(qpos)
        base_qpos = base_qpos.at[0:2].set(
            qpos[self._floating_base_qpos_addr : self._floating_base_qpos_addr + 2]
            + dxy
        )

        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
        quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
        new_quat = math.quat_mul(
            qpos[
                self._floating_base_qpos_addr + 3 : self._floating_base_qpos_addr + 7
            ],
            quat,
        )
        base_qpos = base_qpos.at[3:7].set(new_quat)
        qpos = self.set_floating_base_qpos(base_qpos, qpos)

        # Start close to the model's home pose. Large multiplicative joint
        # randomization and random tuck biases make the initial task needlessly
        # inconsistent, so use only small additive joint noise while debugging.
        rng, key = jax.random.split(rng)
        joint_qpos = self.get_actuator_joints_qpos(qpos)
        joint_noise = jax.random.uniform(
            key, (self._actuators,), minval=-0.02, maxval=0.02
        )
        qpos = self.set_actuator_joints_qpos(joint_qpos + joint_noise, qpos)

        # Zero initial velocity: the policy must create the roll itself.
        qvel = jp.zeros(self.mjx_model.nv)

        ctrl = self.get_actuator_joints_qpos(qpos)
        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl)
        initial_forward = data.site_xmat[self._site_id][:, 0]

        info = {
            "rng": rng,
            "step": 0,
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "last_last_last_act": jp.zeros(self.mjx_model.nu),
            "motor_targets": self._default_actuator,
            "feet_air_time": jp.zeros(2),
            "last_contact": jp.zeros(2, dtype=bool),
            "swing_peak": jp.zeros(2),
            "push": jp.array([0.0, 0.0]),
            "push_step": 0,
            "push_interval_steps": jp.array(1, dtype=jp.int32),
            "action_history": jp.zeros(
                self._config.noise_config.action_max_delay * self._actuators
            ),
            "imu_history": jp.zeros(self._config.noise_config.imu_max_delay * 3),
            # Roll bookkeeping: how far around the roll axis we have gone.
            "accumulated_roll": jp.array(0.0),
            "max_roll": jp.array(0.0),
            "roll_progress_delta": jp.array(0.0),
            "roll_rate": jp.array(0.0),
            "last_roll_angle": jp.array(0.0),
            "roll_forward": initial_forward,
            "stalled_steps": jp.array(0, dtype=jp.int32),
            "roll_success": jp.array(False),
            "roll_success_now": jp.array(0.0),
        }

        metrics = {}
        for k, v in self._config.reward_config.scales.items():
            if v != 0:
                if v > 0:
                    metrics[f"reward/{k}"] = jp.zeros(())
                else:
                    metrics[f"cost/{k}"] = jp.zeros(())
        metrics["swing_peak"] = jp.zeros(())
        metrics["max_roll"] = jp.zeros(())
        metrics["roll_fraction"] = jp.zeros(())
        metrics["accumulated_roll_rad"] = jp.zeros(())
        metrics["roll_success"] = jp.zeros(())

        contact = jp.array(
            [
                geoms_colliding(data, geom_id, self._floor_geom_id)
                for geom_id in self._feet_geom_id
            ]
        )
        obs = self._get_obs(data, info, contact)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        state.info["rng"], action_delay_rng = jax.random.split(state.info["rng"], 2)

        # Action delay (same mechanism as joystick.py).
        action_history = (
            jp.roll(state.info["action_history"], self._actuators)
            .at[: self._actuators]
            .set(action)
        )
        state.info["action_history"] = action_history
        action_idx = jax.random.randint(
            action_delay_rng,
            (1,),
            minval=self._config.noise_config.action_min_delay,
            maxval=self._config.noise_config.action_max_delay,
        )
        action_w_delay = action_history.reshape((-1, self._actuators))[action_idx[0]]

        motor_targets = (
            self._default_actuator + action_w_delay * self._config.action_scale
        )

        if USE_MOTOR_SPEED_LIMITS:
            prev_motor_targets = state.info["motor_targets"]
            motor_targets = jp.clip(
                motor_targets,
                prev_motor_targets - self._config.max_motor_velocity * self.dt,
                prev_motor_targets + self._config.max_motor_velocity * self.dt,
            )

        data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)
        state.info["motor_targets"] = motor_targets

        contact = jp.array(
            [
                geoms_colliding(data, geom_id, self._floor_geom_id)
                for geom_id in self._feet_geom_id
            ]
        )
        contact_filt = contact | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
        state.info["feet_air_time"] += self.dt
        p_f = data.site_xpos[self._feet_site_id]
        p_fz = p_f[..., -1]
        state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

        # Integrate the actual body inclination instead of raw angular
        # velocity. Small back-and-forth jitter then cancels out and cannot be
        # mistaken for completing a full forward roll.
        gravity = self._get_body_gravity(data)
        up_world = data.site_xmat[self._site_id][:, 2]
        roll_forward = state.info["roll_forward"]
        roll_angle = jp.arctan2(
            jp.dot(up_world, roll_forward),
            jp.dot(up_world, jp.array([0.0, 0.0, 1.0])),
        )
        angle_delta = jp.arctan2(
            jp.sin(roll_angle - state.info["last_roll_angle"]),
            jp.cos(roll_angle - state.info["last_roll_angle"]),
        )
        state.info["last_roll_angle"] = roll_angle
        state.info["roll_rate"] = FLIP_DIRECTION * angle_delta / self.dt

        old_roll = state.info["accumulated_roll"]
        roll_delta = FLIP_DIRECTION * angle_delta

        # Keep up to two turns of angular bookkeeping so a policy can recover
        # its upright posture even if it overshoots the first revolution.
        new_roll = jp.clip(
            old_roll + roll_delta, 0.0, 2.0 * ROLL_SUCCESS_ANGLE
        )
        old_progress = jp.clip(old_roll / ROLL_SUCCESS_ANGLE, 0.0, 1.0)
        new_progress = jp.clip(new_roll / ROLL_SUCCESS_ANGLE, 0.0, 1.0)
        state.info["accumulated_roll"] = new_roll

        # _get_reward is a reward-rate function and step() multiplies it by dt.
        # Divide by dt here so the integrated progress bonus is scale * delta
        # progress, independent of control frequency.
        state.info["roll_progress_delta"] = (
            new_progress - old_progress
        ) / self.dt
        state.info["max_roll"] = jp.maximum(
            state.info["max_roll"], new_progress
        )

        base_height_now = data.qpos[self._floating_base_qpos_addr + 2]
        made_progress = (new_roll - old_roll) > 0.005
        state.info["stalled_steps"] = jp.where(
            (base_height_now < 0.09) & (~made_progress),
            state.info["stalled_steps"] + 1,
            0,
        )
        upright_now = gravity[-1] < -0.7
        feet_touching_now = jp.any(contact)
        roll_complete_now = new_roll >= ROLL_SUCCESS_ANGLE
        success_condition = (
            roll_complete_now
            & upright_now
            & (base_height_now > 0.10)
            & feet_touching_now
        )
        success_now = success_condition & (~state.info["roll_success"])
        state.info["roll_success_now"] = success_now.astype(jp.float32)
        state.info["roll_success"] = state.info["roll_success"] | success_now

        obs = self._get_obs(data, state.info, contact)
        done = self._get_termination(data, state.info)
        done = done | (state.info["roll_success_now"] > 0.5)

        rewards = self._get_reward(
            data, action, state.info, state.metrics, done, first_contact, contact
        )
        rewards = {
            k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
        }
        reward = sum(rewards.values()) * self.dt
        reward = jp.clip(reward, -100.0, 10000.0)

        state.info["step"] += 1
        state.info["last_last_last_act"] = state.info["last_last_act"]
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        state.info["feet_air_time"] *= ~contact
        state.info["last_contact"] = contact
        state.info["swing_peak"] *= ~contact
        
        for k, v in rewards.items():
            rew_scale = self._config.reward_config.scales[k]
            if rew_scale != 0:
                if rew_scale > 0:
                    state.metrics[f"reward/{k}"] = v
                else:
                    state.metrics[f"cost/{k}"] = -v
        state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])
        state.metrics["max_roll"] = state.info["max_roll"]
        state.metrics["roll_fraction"] = jp.clip(
            state.info["accumulated_roll"] / ROLL_SUCCESS_ANGLE, 0.0, 1.0
        )
        state.metrics["accumulated_roll_rad"] = state.info["accumulated_roll"]
        state.metrics["roll_success"] = state.info["roll_success"].astype(jp.float32)

        done = done.astype(reward.dtype)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)
        return state

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------
    def _get_termination(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        # Numerical safety always applies.
        nan_term = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

        if ROLL_PHASE == 0:
            # Give the policy room to fall and explore, but reset a fallen
            # robot after it has become stationary instead of letting it farm
            # a long episode by twitching on the ground.
            settled = info["stalled_steps"] > 25
            incomplete = (
                info["accumulated_roll"] < ROLL_SUCCESS_ANGLE - 0.5
            )
            return nan_term | (settled & incomplete)

        # Body-frame gravity is -Z upright and +Z upside down.
        upside_down = self._get_body_gravity(data)[-1] > 0.5
        return nan_term | upside_down

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_obs(
        self, data: mjx.Data, info: dict[str, Any], contact: jax.Array
    ) -> mjx_env.Observation:
        gyro = self.get_gyro(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gyro = (
            gyro
            + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gyro
        )

        accelerometer = self.get_accelerometer(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_accelerometer = (
            accelerometer
            + (2 * jax.random.uniform(noise_rng, shape=accelerometer.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.accelerometer
        )

        gravity = data.site_xmat[self._site_id].T @ jp.array([0, 0, -1])
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gravity = (
            gravity
            + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gravity
        )

        imu_history = jp.roll(info["imu_history"], 3).at[:3].set(noisy_gravity)
        info["imu_history"] = imu_history
        imu_idx = jax.random.randint(
            noise_rng,
            (1,),
            minval=self._config.noise_config.imu_min_delay,
            maxval=self._config.noise_config.imu_max_delay,
        )
        noisy_gravity = imu_history.reshape((-1, 3))[imu_idx[0]]

        joint_angles = self.get_actuator_joints_qpos(data.qpos)
        joint_backlash = self.get_actuator_backlash_qpos(data.qpos)
        for i in self.backlash_idx_to_add:
            joint_backlash = jp.insert(joint_backlash, i, 0)
        joint_angles = joint_angles + joint_backlash

        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_angles = (
            joint_angles
            + (2.0 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1.0)
            * self._config.noise_config.level
            * self._qpos_noise_scale
        )

        joint_vel = self.get_actuator_joints_qvel(data.qvel)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_vel = (
            joint_vel
            + (2.0 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1.0)
            * self._config.noise_config.level
            * self._config.noise_config.scales.joint_vel
        )

        linvel = self.get_local_linvel(data)

        # NOTE: no command, no imitation phase. Roll task only needs
        # proprioception + last actions.
        state = jp.hstack(
            [
                noisy_gyro,  # 3
                noisy_accelerometer,  # 3
                noisy_gravity,  # 3
                noisy_joint_angles - self._default_actuator,  # 14
                noisy_joint_vel * self._config.dof_vel_scale,  # 14
                info["last_act"],  # 14
                info["last_last_act"],  # 14
                info["last_last_last_act"],  # 14
                info["motor_targets"],  # 14
                contact,  # 2
            ]
        )

        accelerometer = self.get_accelerometer(data)
        global_angvel = self.get_global_angvel(data)
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
        root_height = data.qpos[self._floating_base_qpos_addr + 2]

        privileged_state = jp.hstack(
            [
                state,
                gyro,
                accelerometer,
                gravity,
                linvel,
                global_angvel,
                joint_angles - self._default_actuator,
                joint_vel,
                root_height,
                data.actuator_force,
                contact,
                feet_vel,
                info["feet_air_time"],
            ]
        )

        return {
            "state": state,
            "privileged_state": privileged_state,
        }

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
        done: jax.Array,
        first_contact: jax.Array,
        contact: jax.Array,
    ) -> dict[str, jax.Array]:
        del metrics, done, first_contact

        gyro = self.get_gyro(data)
        gravity = self._get_body_gravity(data)
        base_height = data.qpos[self._floating_base_qpos_addr + 2]
        signed_roll_velocity = info["roll_rate"]

        # Actuator joint order:
        # left_hip_pitch, left_knee, right_hip_pitch, right_knee.
        # action_scale must allow the home pose to reach these limits.
        tuck_target = jp.array([1.0, 1.5, -1.0, 1.5])
        joint_angles = self.get_actuator_joints_qpos(data.qpos)
        tuck_angles = joint_angles[jp.array([2, 3, 11, 12])]

        tuck_error = jp.mean(jp.square(tuck_angles - tuck_target))
        tuck_reward = jp.exp(-tuck_error / 0.25)

        # Reward only motion in the intended direction; cap the rate reward
        # so spinning as fast as possible cannot dominate completion.
        target_vel = self._config.reward_config.roll_target_vel
        forward_velocity_reward = jp.clip(
            signed_roll_velocity / jp.maximum(target_vel, 1e-6), 0.0, 1.0
        )

        # Once one turn has been accumulated, encourage the robot to become
        # upright again. This shaping is zero before the first full turn.
        has_completed_rotation = (
            info["accumulated_roll"] >= ROLL_SUCCESS_ANGLE
        )
        upright_shaping = jp.where(
            has_completed_rotation,
            jp.clip(-gravity[-1], 0.0, 1.0),
            0.0,
        )

        ret = {
            "roll_velocity": forward_velocity_reward,
            "non_roll_rotation": cost_non_roll_rotation(gyro, axis=ROLL_AXIS),
            # Kept for config compatibility; deliberately disabled in scales.
            "roll_progress": jp.array(0.0),
            "roll_progress_delta": info["roll_progress_delta"],
            "base_height": cost_roll_base_height(
                base_height, self._config.reward_config.tuck_height
            ),
            "torques": cost_torques(data.actuator_force),
            "action_rate": cost_action_rate(action, info["last_act"]),
            "alive": reward_alive(),
            # Deliberately zero: repeatedly rewarding accumulated angle lets
            # the policy camp at a partially completed pose.
            "roll_accumulated": jp.array(0.0),
            "side_roll_penalty": jp.abs(gyro[0]),
            # Disabled: paying for angular speed while stationary on the
            # ground rewards twitching.
            "stagnation": jp.array(0.0),
            "tuck_pose": tuck_reward,
            "landing": upright_shaping,
            # Divide by dt because step() multiplies the summed reward by dt.
            # Thus a success bonus of 20 is paid once, not every frame.
            "roll_success": info["roll_success_now"] / self.dt,
        }

        return ret

    def _get_body_gravity(self, data: mjx.Data) -> jax.Array:
        """Gravity direction in the IMU/body frame."""
        return data.site_xmat[self._site_id].T @ jp.array([0.0, 0.0, -1.0])
