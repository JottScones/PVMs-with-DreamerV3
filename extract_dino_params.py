import elements
from dreamerv3.agent import Agent
import ruamel.yaml as yaml
import jax
import numpy as np
from dreamerv3.main import make_agent
from transformers import FlaxCLIPVisionModel
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

###################################

hf_model = FlaxCLIPVisionModel.from_pretrained(
    "openai/clip-vit-base-patch32", dtype=jax.numpy.bfloat16)
path = "logdir/PICK_YCB/CLIP/CLIPFT_pick_ycb_train_ID/ckpt/20250728T232902F103693"
configs = elements.Path('dreamerv3/configs.yaml').read()
configs = yaml.YAML(typ='safe').load(configs)
config = elements.Config(configs['defaults'])
config = config.update(configs['maniskillview'])
agent = make_agent(config)
cp = elements.Checkpoint()
cp.agent = agent
cp.load(path, keys=['agent'])
clip_da = agent.params['enc/clip_enc/pretrained_vision_params/value']
clip_np = jax.device_get(clip_da)
clip_np = jax.tree_util.tree_map(lambda x: np.asarray(x), clip_da)

hf_model.params = clip_np
out_dir = Path("clip-ft-checkpoint")
hf_model.save_pretrained(out_dir)
