import random
import itertools
from datasets import load_from_disk
from transformers import FlaxDinov2Model, FlaxCLIPVisionModel, AutoImageProcessor
from sklearn.neighbors import KNeighborsClassifier
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

tot = 20_000
ds_local = load_from_disk("./imagenet-1k-20k")

# This shuffles in Arrow and returns a new Dataset:
ds_shuffled = ds_local.shuffle(seed=0)

train_lim = int(tot * 0.8)
train_examples = ds_shuffled.select(range(train_lim))
eval_examples = ds_shuffled.select(range(train_lim, tot))

models_to_test = [
    "carla-clip-ft-checkpoint-1",
    "carla-clip-ft-checkpoint-2",
    "carla-clip-partial-checkpoint-1",
    "carla-clip-partial-checkpoint-2",
    "carla-dino-ft-checkpoint-1",
    "carla-dino-ft-checkpoint-2",
    "carla-dino-partial-checkpoint-1",
    "carla-dino-partial-checkpoint-2",
    "clip-ft-checkpoint-1",
    "clip-ft-checkpoint-2",
    "clip-partial-checkpoint-1",
    "clip-partial-checkpoint-2",
    "dino-ft-checkpoint-1",
    "dino-ft-checkpoint-2",
    "dino-partial-checkpoint-1",
    "dino-partial-checkpoint-2"
]

for model_name in models_to_test:
    if "clip" in model_name:
        model = FlaxCLIPVisionModel.from_pretrained(
            model_name,
            dtype=jax.numpy.bfloat16,
        )
        processor = AutoImageProcessor.from_pretrained(
            "openai/clip-vit-base-patch32")

    elif "dino" in model_name:
        model = FlaxDinov2Model.from_pretrained(
            model_name,
            dtype=jax.numpy.bfloat16,
        )
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")

    @partial(jax.jit, static_argnums=(0,))
    def extract_features(model, pixel_values):
        outs = model(pixel_values=pixel_values, train=False)
        # use the CLS token
        cls = outs.last_hidden_state[:, 0]
        # Return JAX array, convert to numpy outside
        return cls.astype(jnp.float32)

    # 4) Build feature / label arrays

    def build_split(exs):
        feats, labels = [], []
        for i, ex in enumerate(exs):
            if i % 1000 == 0:
                print(f"Processing example {i}/{len(exs)}")
            # Convert PIL image to ensure proper format
            image = ex["image"]
            if hasattr(image, 'convert'):
                image = image.convert('RGB')
            # Process image and extract pixel values
            processed = processor(images=image, return_tensors="np")
            pix = processed["pixel_values"]
            f = extract_features(model, pix)
            # Convert to numpy here, outside the JIT function
            feats.append(np.asarray(f[0]))
            labels.append(ex["label"])
        return np.stack(feats), np.array(labels)

    X_train, y_train = build_split(train_examples)
    X_eval,  y_eval = build_split(eval_examples)

    print("Running k-NN on DINO features...")
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X_train, y_train)
    acc = knn.score(X_eval, y_eval)
    print(f"k-NN accuracy: {acc*100:.2f}%")

    with open('imagenet_results.txt', 'a+') as f:
        f.write(f'{model_name}: {acc*100}')
