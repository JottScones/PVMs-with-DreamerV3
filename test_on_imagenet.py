import random
import itertools
from datasets import load_dataset
from transformers import FlaxDinov2Model, AutoImageProcessor
from sklearn.neighbors import KNeighborsClassifier
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

# 1) Load with streaming=True so nothing is cached
ds_stream = load_dataset("imagenet-1k", split="train", streaming=True)

# 2) Pull out exactly the 2000+500 examples you need
#    (shuffle in-memory, but only keep what you need)
rng = random.Random(0)
buffer = []
# grab first 10k for shuffle buffer
for example in itertools.islice(ds_stream, 10000):
    buffer.append(example)
rng.shuffle(buffer)
train_examples = buffer[:2000]
eval_examples = buffer[2000:2500]

# 3) Load your model & processor once
model = FlaxDinov2Model.from_pretrained(
    "dino-ft-checkpoint",
    dtype=jax.numpy.bfloat16,
    _do_init=False,
)
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")


@partial(jax.jit, static_argnums=(0,))
def extract_features(params, pixel_values):
    outs = model(pixel_values=pixel_values, params=params, train=False)
    # use the CLS token
    cls = outs.last_hidden_state[:, 0]
    return np.asarray(cls.astype(jnp.float32))

# 4) Build feature / label arrays


def build_split(exs):
    feats, labels = [], []
    for ex in exs:
        pix = processor(images=ex["image"], return_tensors="np")[
            "pixel_values"]
        f = extract_features(model.params, pix)
        feats.append(f[0])
        labels.append(ex["label"])
    return np.stack(feats), np.array(labels)


X_train, y_train = build_split(train_examples)
X_eval,  y_eval = build_split(eval_examples)

# 5) k-NN
print("Running k-NN on DINO features...")
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn.fit(X_train, y_train)
acc = knn.score(X_eval, y_eval)
print(f"k-NN accuracy: {acc*100:.2f}%")
