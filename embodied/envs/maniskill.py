import elements
import embodied
import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper, FlattenActionSpaceWrapper
import gymnasium as gym
import numpy as np

class ManiSkill(embodied.Env):
  def __init__(self, task, size=(64, 64), resize='pillow', **kwargs):
    assert resize in ('opencv', 'pillow'), resize
    from . import from_gym
    self.size = size
    self.resize = resize
    kwargs['sensor_configs'] = dict(width=size[0], height=size[1])

    env = gym.make(task, **kwargs)
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=False)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)

    self.env = from_gym.FromGym(env)


  @property
  def obs_space(self):
    spaces = self.env.obs_space.copy()
    spaces['image'] = elements.Space(np.uint8, (*self.size, 3))
    del spaces['rgb']
    return spaces

  @property
  def act_space(self):
    return self.env.act_space

  def step(self, action):
    obs = self.env.step(action)
    return obs
  
  def _obs(self, obs, reward, is_first=False, is_last=False, is_terminal=False):
    if not (self._obs_dict and isinstance(obs, dict)):
      obs = {self._obs_key: obs['rgb']}

    del obs['rgb']
    obs = self._flatten(obs)
    obs = {k: np.asarray(v) for k, v in obs.items()}
    obs.update(
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal)
    return obs

