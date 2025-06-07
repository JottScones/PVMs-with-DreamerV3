import elements
import embodied
import functools
import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper, FlattenActionSpaceWrapper
import gymnasium as gym
import numpy as np

class ManiSkill(embodied.Env):
  def __init__(self, task, size=(64, 64), obs_key="image", act_key="action", **kwargs):
    kwargs['sensor_configs'] = dict(width=size[0], height=size[1])

    self.env = gym.make(task, **kwargs)
    self.env = FlattenRGBDObservationWrapper(self.env, rgb=True, depth=False, state=False)
    if isinstance(self.env.action_space, gym.spaces.Dict):
        self.env = FlattenActionSpaceWrapper(self.env)
    
    self.size = size
    self._obs_key = obs_key
    self._act_key = act_key
    self._done = True
    self._info = None
    
  @property
  def info(self):
    return self._info

  @property
  def obs_space(self):
    return {
        self._obs_key: elements.Space(np.uint8, (*self.size, 3)),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }

  def _obs(self, raw_obs, reward, is_first=False, is_last=False, is_terminal=False):
    image = raw_obs["rgb"].cpu().reshape((*self.size, 3))
    obs = {
        self._obs_key: image,
        "reward": np.float32(reward),
        "is_first": is_first,
        "is_last": is_last,
        "is_terminal": is_terminal
    }
    obs = {k: np.asarray(v) for k, v in obs.items()}
    return obs

  @functools.cached_property
  def act_space(self):
      if self._act_dict:
       spaces = self._flatten(self.env.action_space.spaces)
      else:
        spaces = {self._act_key: self.env.action_space}

      spaces = {k: self._convert(v) for k, v in spaces.items()}
      spaces['reset'] = elements.Space(bool)
      return spaces

  def step(self, action):
    if action['reset'] or self._done:
      self._done = False
      obs, self._info = self.env.reset()
      return self._obs(obs, 0.0, is_first=True)

    action = action[self._act_key]

    obs, reward, terminated, truncated, self._info = self.env.step(action)
    reward, terminated, truncated = reward.cpu(), terminated.cpu(), truncated.cpu()

    self._done = terminated or truncated
    return self._obs(
        obs, reward,
        is_last=bool(self._done),
        is_terminal=terminated)

  def _convert(self, space):
    if hasattr(space, 'n'):
      return elements.Space(np.int32, (), 0, space.n)
    return elements.Space(space.dtype, space.shape, space.low, space.high)
