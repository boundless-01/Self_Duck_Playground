"""Training wrappers for the roll task.

The installed mujoco_playground package keeps environment ``info`` across
episode resets. Roll state cannot do that because the accumulated angle, action
history, and success flags would leak into the next episode.
"""

import contextlib
from typing import Any, Callable, Optional, Tuple

from brax.envs.wrappers import training as brax_training
import jax
import jax.numpy as jp
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src import wrapper as playground_wrapper


class FullResetWrapper(playground_wrapper.Wrapper):
    """Auto-reset the full environment state while preserving eval metrics."""

    def __init__(self, env: Any):
        super().__init__(env)
        self._info_key = "RollAutoResetWrapper"

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng_key = jax.vmap(jax.random.split)(rng)
        rng, reset_key = rng_key[..., 0], rng_key[..., 1]
        state = self.env.reset(reset_key)
        state.info[f"{self._info_key}_first_data"] = state.data
        state.info[f"{self._info_key}_first_obs"] = state.obs
        state.info[f"{self._info_key}_rng"] = rng
        state.info[f"{self._info_key}_done_count"] = jp.zeros(
            reset_key.shape[:-1], dtype=jp.int32
        )
        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        rng_key = jax.vmap(jax.random.split)(
            state.info[f"{self._info_key}_rng"]
        )
        reset_rng, reset_key = rng_key[..., 0], rng_key[..., 1]
        reset_state = self.reset(reset_key)

        if "steps" in state.info:
            steps = jp.where(
                state.done, jp.zeros_like(state.info["steps"]), state.info["steps"]
            )
            state.info.update(steps=steps)

        state = state.replace(done=jp.zeros_like(state.done))
        state = self.env.step(state, action)

        def where_done(x: jax.Array, y: jax.Array) -> jax.Array:
            done = state.done
            if done.shape and done.shape[0] != x.shape[0]:
                return y
            if done.shape:
                done = jp.reshape(
                    done, [x.shape[0]] + [1] * (len(x.shape) - 1)
                )
            return jp.where(done, x, y)

        data = jax.tree.map(where_done, reset_state.data, state.data)
        obs = jax.tree.map(where_done, reset_state.obs, state.obs)
        next_info = jax.tree.map(where_done, reset_state.info, state.info)

        # These fields are consumed after the terminal transition for logging
        # and must survive until EpisodeWrapper advances to the next episode.
        for key in ("episode_metrics", "episode_done", "truncation", "steps"):
            if key in state.info:
                next_info[key] = state.info[key]

        done_count_key = f"{self._info_key}_done_count"
        next_info[done_count_key] = (
            state.info[done_count_key] + state.done.astype(jp.int32)
        )
        next_info[f"{self._info_key}_rng"] = reset_rng
        return state.replace(data=data, obs=obs, info=next_info)


class DomainRandomizationVmapWrapper(playground_wrapper.Wrapper):
    """Vectorized domain randomization with temporary model replacement."""

    def __init__(
        self,
        env: mjx_env.MjxEnv,
        randomization_fn: Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]],
    ):
        super().__init__(env)
        self._mjx_model_v, self._in_axes = randomization_fn(self.mjx_model)

    @contextlib.contextmanager
    def _v_env_fn(self, mjx_model: mjx.Model):
        env = self.env.unwrapped
        old_mjx_model = env._mjx_model
        try:
            env._mjx_model = mjx_model
            yield env
        finally:
            env._mjx_model = old_mjx_model

    def reset(self, rng: jax.Array) -> mjx_env.State:
        def reset(mjx_model: mjx.Model, key: jax.Array):
            with self._v_env_fn(mjx_model) as env:
                return env.reset(key)

        return jax.vmap(reset, in_axes=[self._in_axes, 0])(
            self._mjx_model_v, rng
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        def step(mjx_model: mjx.Model, s: mjx_env.State, a: jax.Array):
            with self._v_env_fn(mjx_model) as env:
                return env.step(s, a)

        return jax.vmap(step, in_axes=[self._in_axes, 0, 0])(
            self._mjx_model_v, state, action
        )


def wrap_for_roll_training(
    env: mjx_env.MjxEnv,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[
        Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]
    ] = None,
) -> playground_wrapper.Wrapper:
    if randomization_fn is None:
        env = brax_training.VmapWrapper(env)
    else:
        env = DomainRandomizationVmapWrapper(env, randomization_fn)
    env = brax_training.EpisodeWrapper(env, episode_length, action_repeat)
    return FullResetWrapper(env)
