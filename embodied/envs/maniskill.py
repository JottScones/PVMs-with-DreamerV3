
import embodied
import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper, FlattenActionSpaceWrapper
import gymnasium as gym

class ManiSkill(embodied.Env):
  def __init__(self, task, size=(64, 64), resize='pillow', **kwargs):
    assert resize in ('opencv', 'pillow'), resize
    from . import from_gym
    self.size = size
    self.resize = resize
    kwargs['sensor_configs'] = dict(width=size[0], height=size[1])

    env = from_gym.FromGym(task, **kwargs)
    self.env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=False)
    if isinstance(env.action_space, gym.spaces.Dict):
        self.env = FlattenActionSpaceWrapper(env)

  @property
  def obs_space(self):
    spaces = self.env.obs_space.copy()
    spaces['image'] = spaces['rgb']
    del spaces['rgb']
    return spaces

  @property
  def act_space(self):
    return self.env.act_space

  def step(self, action):
    obs = self.env.step(action)
    obs['image'] = obs['rgb']
    del obs['rgb']
    return obs
