import elements
import json
import torch
import embodied
import functools
import sapien
import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper, FlattenActionSpaceWrapper
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.utils.structs.pose import Pose
import gymnasium as gym
import numpy as np

from mani_skill.envs.tasks.tabletop import PickSingleYCBEnv
from mani_skill.utils.registration import register_env

@register_env("PickSingleYCBWrist-v1", max_episode_steps=50, asset_download_ids=["ycb"])
class PickSingleYCBWristEnv(PickSingleYCBEnv):
  def __init__(
    self,
    *args,
    robot_uids="panda_wristcam",
    robot_init_qpos_noise=0.02,
    num_envs=1,
    reconfiguration_freq=None,
    in_distribution=True,
    rand_obj_idx=0,
    **kwargs,
):
    """
    Copy of PickSingleYCBEnv init but sourcing model ids from a randomised 80/20 split of the available
    objects. 
    We skip over the parent init and initialise BaseEnv.
    """
    self.robot_init_qpos_noise = robot_init_qpos_noise
    self.model_id = None
    self.in_distribution = in_distribution

    with open('random_object_split.json', 'r') as f:
      object_splits = json.load(f)
    
    eval_mode = "train" if self.in_distribution else "test"
    train_model_ids = object_splits[str(rand_obj_idx)][eval_mode]
    
    self.all_model_ids = np.array(train_model_ids)

    if reconfiguration_freq is None:
      if num_envs == 1:
        reconfiguration_freq = 1
      else:
        reconfiguration_freq = 0

    # Skip PickSingle init and call parent
    super(PickSingleYCBEnv, self).__init__(
        *args,
        robot_uids=robot_uids,
        reconfiguration_freq=reconfiguration_freq,
        num_envs=num_envs,
        **kwargs,
    )

  def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
    """
    Copy of method from PickSingleYCBEnv but instead ensuring that when training we use 80% of the available positions, centrally.
    When evaluating OOD mode then we use the outer ring equivalent to 20% area.
    """
    L = 0.08944

    with torch.device(self.device):
      b = len(env_idx)
      self.table_scene.initialize(env_idx)

      # ------ Set Object Position ------
      obj_xyz = torch.zeros((b, 3))
      if self.in_distribution:
        obj_xyz[:, :2] = torch.rand((b, 2)) * (2 * L) - L

      else:
        xy_rand = torch.zeros((b, 2))
        is_valid = torch.zeros(b, dtype=torch.bool)
        while not torch.all(is_valid):
          # Sample from the entire area [-0.1, 0.1] x [-0.1, 0.1] for the invalid points
          num_invalid = (~is_valid).sum()
          new_samples = torch.rand((num_invalid, 2)) * 0.2 - 0.1
          
          # Check if the new samples are outside the central square
          newly_valid = torch.any(torch.abs(new_samples) > L, dim=1)

          # Update the full batch tensors
          xy_rand[~is_valid] = new_samples
          is_valid[~is_valid] = newly_valid
        obj_xyz[:, :2] = xy_rand

      obj_xyz[:, 2] = self.object_zs[env_idx]
      qs = random_quaternions(b, lock_x=True, lock_y=True)
      self.obj.set_pose(Pose.create_from_pq(p=obj_xyz, q=qs))

      # ----- Set Goal Position -----
      goal_xyz = torch.zeros((b, 3))
      if self.in_distribution:
        xy_rand_goal = torch.rand((b, 2)) * (2 * L) - L
        goal_xyz[:, :2] = xy_rand_goal

      else: 
        xy_rand_goal = torch.zeros((b, 2))
        is_valid = torch.zeros(b, dtype=torch.bool)
        while not torch.all(is_valid):
          # Sample from the entire area [-0.1, 0.1] x [-0.1, 0.1] for the invalid points
          num_invalid = (~is_valid).sum()
          new_samples = torch.rand((num_invalid, 2)) * 0.2 - 0.1

          # Check if the new samples are outside the central square
          newly_valid = torch.any(torch.abs(new_samples) > L, dim=1)

          # Update the full batch tensors
          xy_rand_goal[~is_valid] = new_samples
          is_valid[~is_valid] = newly_valid
        goal_xyz[:, :2] = xy_rand_goal

      goal_xyz[:, 2] = torch.rand((b)) * 0.3 + obj_xyz[:, 2]
      self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

      # ------ Set Robot Arm Position ------
      # Initialize robot arm to a higher position above the table than the default typically used for other table top tasks
      if self.robot_uids == "panda" or self.robot_uids == "panda_wristcam":
        # fmt: off
        qpos = np.array(
            [0.0, 0, 0, -np.pi * 2 / 3, 0, np.pi * 2 / 3, np.pi / 4, 0.04, 0.04]
        )
        # fmt: on
        qpos[:-2] += self._episode_rng.normal(
            0, self.robot_init_qpos_noise, len(qpos) - 2
        )
        self.agent.reset(qpos)
        self.agent.robot.set_root_pose(sapien.Pose([-0.615, 0, 0]))
      else:
        raise NotImplementedError(self.robot_uids)

  @property
  def _default_sensor_configs(self):
    # Remove other camera view
    return []
  
  def _load_scene(self, options):
    super()._load_scene(options)
    # Unhide goal circle from rendering
    self._hidden_objects = [o for o in self._hidden_objects if o != self.goal_site]

class ManiSkill(embodied.Env):
  def __init__(self, task, size=(64, 64), obs_key="image", act_key="action", seed=None, **kwargs):
    kwargs['sensor_configs'] = dict(width=size[0], height=size[1], shader_pack="default")

    self.env = gym.make(task, **kwargs)
    self.env = FlattenRGBDObservationWrapper(self.env, rgb=True, depth=False, state=False)
    if isinstance(self.env.action_space, gym.spaces.Dict):
        self.env = FlattenActionSpaceWrapper(self.env)
    
    self.size = size
    self._obs_key = obs_key
    self._act_key = act_key
    self._done = True
    self._info = None
    self._random = np.random.RandomState(seed)
    
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
      spaces = {self._act_key: self.env.action_space}

      spaces = {k: self._convert(v) for k, v in spaces.items()}
      spaces['reset'] = elements.Space(bool)
      return spaces

  def step(self, action):
    if action['reset'] or self._done:
      self._done = False
      obs, self._info = self.env.reset(seed=self._random.randint(0, 2 ** 31 - 1))
      return self._obs(obs, 0.0, is_first=True)

    action = action[self._act_key]

    obs, reward, terminated, truncated, self._info = self.env.step(action)
    reward, terminated, truncated = reward.cpu(), terminated.cpu(), truncated.cpu()

    self._done = terminated.item() or truncated.item()
    return self._obs(
        obs, reward.item(),
        is_last=bool(self._done),
        is_terminal=terminated.item())

  def _convert(self, space):
    if hasattr(space, 'n'):
      return elements.Space(np.int32, (), 0, space.n)
    return elements.Space(space.dtype, space.shape, space.low, space.high)
