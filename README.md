# Pre-trained Visual Representations Generalise Where it Matters in Model-Based Reinforcement Learning
Fork of the original [DreamerV3](https://github.com/danijar/dreamerv3/tree/main) repo. In this work, we modify DreamerV3 such that we can load and fine-tune pre-trained vision models. We also integrate two new environments:
* [ManiSkill](https://maniskill.readthedocs.io/en/latest/index.html)
* [RL-ViGen](https://github.com/gemcollector/RL-ViGen)

While any Flax implementation can be integrated into this codebase, the two pre-trained models currently supported are:
* [CLIP](https://huggingface.co/docs/transformers/model_doc/clip)
* [DINOv2](https://huggingface.co/docs/transformers/model_doc/dinov2)

# Instructions

The code has been tested on Linux and Mac and requires Python 3.10+.

## Manual

Install dependencies:

```sh
pip install -U -r requirements.txt
```

Training script:

```sh
python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/{timestamp} \
  --configs crafter \
  --run.train_ratio 32
```

To reproduce results, train on the desired task using the corresponding config,
such as `--configs atari --task atari_pong`.

View results:

```sh
pip install -U scope
python -m scope.viewer --basedir ~/logdir --port 8000
```

To change the vision encoder, use `--agent.enc.typ [simple | dino | clip]`.
To select encoder finetuning layers, use `--agent.enc.finetune_layers <int>`, where the argument represents the index we start finetuning from. Say we provide the number 8, then layers index 8+ will be updated during training.

An example script fully fine-tuning DINOv2 with ManiSkill:

```sh
python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/DINO_FT \
  --configs maniskill \
  --run.train_ratio 32 \
  --agent.enc.typ: dino \
  --agent.enc.freeze: False \
  --agent.enc.finetune_layers: 0 \
```

Although this is not necessary usually to run it in this way, as all configs for each run described in this thesis are defined in the `dreamer/configs.yaml` file.

### CARLA
To get CARLA working, you need the [CARLA server](https://carla.readthedocs.io/en/0.8.4/carla_server/). It must be running in the background before you start training:
```sh
vigen/third_party/CARLA_0.9.15/CarlaUE4.sh -RenderOffScreen -nosound -fps 20 --carla-port=2018 -carla-streaming-port=0 -prefernvidia &
```
