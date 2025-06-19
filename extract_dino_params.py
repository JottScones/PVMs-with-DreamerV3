import elements
from dreamerv3.agent import Agent
import ruamel.yaml as yaml
import jax
import numpy as np
from dreamerv3.main import make_agent
from custom_models.dino import CheckpointableFlaxDinov2Model
from pathlib import Path

hf_model = CheckpointableFlaxDinov2Model.from_pretrained(
    "facebook/dinov2-small",
    dtype=jax.numpy.bfloat16,
)
path = "logdir/DINOFT_pick_ycb_train_ID/ckpt/20250615T225839F198992"
configs = elements.Path('dreamerv3/configs.yaml').read()
configs = yaml.YAML(typ='safe').load(configs)
config = elements.Config(configs['defaults'])
config = config.update(configs['maniskillview'])
agent = make_agent(config)
cp = elements.Checkpoint()
cp.agent = agent
cp.load(path, keys=['agent'])
dino_da = agent.params['enc/dino_enc/pretrained_vision_params/value']
dino_np = jax.device_get(dino_da)
dino_np = jax.tree_util.tree_map(lambda x: np.asarray(x), dino_da)

hf_model.params = dino_np
out_dir = Path("my-dino-flax-checkpoint")
hf_model.save_pretrained(out_dir)
