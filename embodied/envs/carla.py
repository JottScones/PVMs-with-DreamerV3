import embodied
import time
import elements
import functools
import numpy as np
import carla
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

    def _convert(self, space):
        if hasattr(space, 'n'):
            return elements.Space(np.int32, (), 0, space.n)
        return elements.Space(space.dtype, space.shape, space.low, space.high)

    def close(self):
        import time
        import carla


def close(self):
    print("Closing CARLA environment...")

    # -------------------------------------------------------------
    # 1) Stop every listening sensor *before* we touch sync mode.
    # -------------------------------------------------------------
    sensors = self.env.world.get_actors().filter("*sensor*")
    vehicles = self.env.world.get_actors().filter("*vehicle*")
    for s in sensors:
        print(
            f"  stopping sensor {s.id} ({s.type_id})  listening={s.is_listening()}")
        if s.is_listening():
            s.stop()

    self.env.world.tick()     # make sure STOP reaches the server
    time.sleep(0.02)          # safety margin on busy machines

    # -------------------------------------------------------------
    # 2) Now we can drop CarlaSyncMode safely.
    # -------------------------------------------------------------
    if getattr(self.env, "sync_mode", None):
        print("Sync Mode begone")
        self.env.sync_mode.__exit__(None, None, None)
        self.env.sync_mode = None

    # -------------------------------------------------------------
    # 3) Destroy all actors in one synchronous batch.
    # -------------------------------------------------------------
    commands = [carla.command.DestroyActor(x)
                for x in list(sensors) + list(vehicles)]
    if commands:
        self.env.client.apply_batch_sync(commands)  # True = also tick

    # -------------------------------------------------------------
    # 4) Verify nothing is left.
    # -------------------------------------------------------------
    leftovers = self.env.world.get_actors().filter("*sensor*")
    if leftovers:
        print("WARNING:", len(leftovers), "sensor(s) still alive – investigate")
    else:
        print("All sensors destroyed successfully.")


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

(carla) [swj24@gpu-q-18 dreamerv3]$ python dreamerv3/main.py --configs carla --logdir ./logdir/carla_baseline
---  ___                           __   ______ ---
--- |   \ _ _ ___ __ _ _ __  ___ _ \ \ / /__ / ---
--- | |) | '_/ -_) _` | '  \/ -_) '/\ V / |_ \ ---
--- |___/|_| \___\__,_|_|_|_\___|_|  \_/ |___/ ---
Replica: 0 / 1
Logdir: logdir/carla_baseline
Run script: train
Observations
  image            Space(uint8, shape=(64, 64, 3), low=0, high=255)
  vector           Space(float32, shape=(7,), low=-inf, high=inf)
  token            Space(int32, shape=(), low=0, high=256)
  count            Space(float32, shape=(), low=0, high=100)
  float2d          Space(float32, shape=(4, 5), low=-inf, high=inf)
  int2d            Space(int32, shape=(2, 3), low=0, high=4)
  reward           Space(float32, shape=(), low=-inf, high=inf)
  is_first         Space(bool, shape=(), low=False, high=True)
  is_last          Space(bool, shape=(), low=False, high=True)
  is_terminal      Space(bool, shape=(), low=False, high=True)
Actions
  act_disc         Space(int32, shape=(), low=0, high=5)
  act_cont         Space(float32, shape=(6,), low=-1.0, high=1.0)
Extras
  consec           Space(int32, shape=(), low=-2147483648, high=2147483647)
  stepid           Space(uint8, shape=(20,), low=0, high=255)
  dyn/deter        Space(float32, shape=(8192,), low=-inf, high=inf)
  dyn/stoch        Space(float32, shape=(32, 64), low=-inf, high=inf)
JAX devices (1): [cuda:0]
Policy devices: cuda:0
Train devices:  cuda:0
Initializing parameters...
Exception happened inside Ninjax scope 'enc'.
jax.errors.SimplifiedTraceback: For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set JAX_TRACEBACK_FILTERING=off to include these.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3/dreamerv3/main.py", line 282, in <module>
    main()
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3/dreamerv3/main.py", line 69, in main
    embodied.run.train(
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3/embodied/run/train.py", line 11, in train
    agent = make_agent()
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3/dreamerv3/main.py", line 138, in make_agent
    return Agent(obs_space, act_space, elements.Config(
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3/embodied/jax/agent.py", line 47, in __new__
    outer.__init__(model, obs_space, act_space, config, jaxcfg)
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3/embodied/jax/agent.py", line 113, in __init__
    self.params, self.train_params_sharding = self._init_params()
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3/embodied/jax/agent.py", line 437, in _init_params
    params, params_sharding = transform.init(
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3/embodied/jax/transform.py", line 50, in init
    params_shapes = fn.eval_shape(*dummy_inputs)
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3/embodied/jax/transform.py", line 44, in fn
    params, _ = inner(params, *args, seed=seed)
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3/embodied/jax/transform.py", line 34, in wrapper
    state, out = fun(*args, create=True, modify=True, ignore=True, **kwargs)
  File "/home/swj24/miniforge3/envs/carla/lib/python3.10/site-packages/ninjax/ninjax.py", line 41, in hidewrapper
    raise e
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3/dreamerv3/agent.py", line 159, in train
    metrics, (carry, entries, outs, mets) = self.opt(
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3/embodied/jax/opt.py", line 43, in __call__
    loss, params, grads, aux = nj.grad(
  File "/home/swj24/miniforge3/envs/carla/lib/python3.10/contextlib.py", line 79, in inner
    return func(*args, **kwds)
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3/embodied/jax/opt.py", line 35, in lossfn2
    outs = lossfn(*args, **kwargs)
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3/dreamerv3/agent.py", line 187, in loss
    enc_carry, enc_entries, tokens = self.enc(
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3/dreamerv3/rssm.py", line 317, in __call__
    assert 3 <= x.shape[-3] <= 16, x.shape
AssertionError: (64, 2, 2, 256)
"""
