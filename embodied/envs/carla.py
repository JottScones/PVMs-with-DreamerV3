import embodied
import elements
import functools
import numpy as np
from vigen.wrappers.carla_wrapper import carla_make


class Carla(embodied.Env):
    def __init__(self, task, repeat=2, size=(128, 128), obs_key="image", act_key="action", seed=None, **kwargs):
        self.env = carla_make(repeat)
        self.size = size

        self._obs_key = obs_key
        self._act_key = act_key
        self._done = True

    @property
    def obs_space(self):
        return {
            self._obs_key: elements.Space(np.uint8, (*self.size, 3)),
            'reward': elements.Space(np.float32),
            'is_first': elements.Space(bool),
            'is_last': elements.Space(bool),
            'is_terminal': elements.Space(bool),
        }

    def _obs(self, obs, reward, is_first=False, is_last=False, is_terminal=False):
        obs = {
            self._obs_key: obs,
            "reward": np.float32(reward),
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal
        }
        obs = {k: np.asarray(v) for k, v in obs.items()}
        return obs

    @functools.cached_property
    def act_space(self):
        spaces = {self._act_key: self.env.action_space}

        spaces = {k: self._convert(v) for k, v in spaces.items()}
        spaces['reset'] = elements.Space(bool)
        return spaces

    @functools.cached_property
    def act_space(self):
        spaces = {self._act_key: self.env.action_space}

        spaces = {k: self._convert(v) for k, v in spaces.items()}
        spaces['reset'] = elements.Space(bool)
        return spaces

    def step(self, action):
        if action['reset'] or self._done:
            self._done = False
            ts = self.env.reset()
            return self._obs(ts.observation, 0.0, is_first=True)

        action = action[self._act_key]

        ts, info = self.env.step(action)
        obs, reward, done = ts.observation, ts.reward, ts.last()

        is_terminal = done and info.get("reason_episode_ended") != "success"
        self._done = done
        return self._obs(
            obs, reward,
            is_last=bool(self._done),
            is_terminal=is_terminal)

    def _convert(self, space):
        if hasattr(space, 'n'):
            return elements.Space(np.int32, (), 0, space.n)
        return elements.Space(space.dtype, space.shape, space.low, space.high)


"""
>>> ts, info = env.step([0.5,0.2])
>>> ts
ExtendedTimeStep(step_type=<StepType.MID: 1>, reward=-0.499999999952938, discount=1.0, observation=array([[[ 90,  90,  90, ...,  90,  90,  88],
        [ 91,  90,  90, ...,  91,  91,  91],
        [ 91,  91,  91, ...,  91,  91,  90],
        ...,
        [210, 213, 212, ..., 208, 210, 210],
        [207, 210, 209, ..., 207, 209, 208],
        [206, 209, 208, ..., 207, 209, 208]],

       [[138, 138, 139, ..., 139, 138, 138],
        [139, 139, 139, ..., 139, 139, 139],
        [139, 138, 139, ..., 139, 139, 138],
        ...,
        [208, 210, 210, ..., 203, 206, 207],
        [204, 208, 205, ..., 202, 205, 205],
        [201, 203, 202, ..., 203, 205, 205]],

       [[180, 180, 179, ..., 181, 180, 180],
        [180, 181, 181, ..., 181, 181, 180],
        [180, 180, 181, ..., 181, 181, 180],
        ...,
        [205, 210, 209, ..., 205, 208, 209],
        [202, 207, 203, ..., 202, 205, 205],
        [202, 203, 201, ..., 202, 205, 204]],

       ...,

       [[ 90,  90,  90, ...,  90,  90,  88],
        [ 90,  90,  90, ...,  90,  88,  88],
        [ 90,  90,  90, ...,  90,  90,  90],
        ...,
        [210, 211, 210, ..., 207, 208, 208],
        [207, 211, 210, ..., 206, 208, 208],
        [204, 208, 207, ..., 206, 206, 206]],

       [[137, 137, 137, ..., 137, 137, 136],
        [138, 137, 137, ..., 138, 137, 137],
        [137, 138, 138, ..., 138, 138, 138],
        ...,
        [208, 209, 209, ..., 203, 205, 205],
        [205, 209, 207, ..., 201, 203, 205],
        [200, 203, 202, ..., 202, 203, 203]],

       [[178, 178, 179, ..., 179, 178, 178],
        [178, 178, 179, ..., 178, 178, 178],
        [178, 178, 179, ..., 179, 178, 178],
        ...,
        [204, 209, 206, ..., 204, 206, 206],
        [201, 208, 205, ..., 201, 205, 206],
        [200, 202, 200, ..., 201, 203, 202]]], dtype=uint8), action=array([0.5, 0.2], dtype=float32))
>>> info
{'reason_episode_ended': '', 'crash_intensity': 0, 'steer': 0.5, 'brake': 0.0, 'distance': 4.467559097613891e-10}
"""
