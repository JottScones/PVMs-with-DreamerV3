import embodied
import traceback
import time
import random

import elements
import functools
import numpy as np
import carla
from vigen.wrappers.carla_wrapper import carla_make, carla_make_eval


class Carla(embodied.Env):
    def __init__(self, task, repeat=2, size=(128, 128), obs_key="image", act_key="action", seed=None, make=True, **kwargs):
        if make:
            self.env = carla_make(repeat)
            self.set_seeds(seed)

        self.size = size

        self._obs_key = obs_key
        self._act_key = act_key
        self._done = True

    def set_seeds(self, seed):
        # Basic Randomness
        np.random.seed(seed)
        random.seed(seed)

        # Carla Randomness
        self.env.tm.set_random_device_seed(seed)

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
            self._obs_key: obs.transpose([1, 2, 0]),
            "reward": np.float32(reward),
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal
        }
        obs = {k: np.asarray(v) for k, v in obs.items()}
        return obs

    @functools.cached_property
    def act_space(self):
        spaces = {self._act_key: elements.Space(
            np.float32, (2,), low=-1.0, high=1.0)}
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
