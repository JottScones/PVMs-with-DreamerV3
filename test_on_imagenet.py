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
ds_stream = load_dataset("imagenet-1k", split="train", streaming=True, cache_dir='./datasets')

# 2) Pull out exactly the 2000+500 examples you need
#    (shuffle in-memory, but only keep what you need)
rng = random.Random(0)
buffer = []
# grab first 10k for shuffle buffer
print("downloading")
tot = 20000
for i, example in enumerate(itertools.islice(ds_stream, tot)):
    if i % 1000 == 0:
        print(f"{i}/{tot}")
    buffer.append(example)
rng.shuffle(buffer)

train_lim = int(tot * 0.8)
train_examples = buffer[:train_lim]
eval_examples = buffer[train_lim:]

# 3) Load your model & processor once
model = FlaxDinov2Model.from_pretrained(
    "dino-ft-checkpoint",
    dtype=jax.numpy.bfloat16,
)
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")

@partial(jax.jit, static_argnums=(0,))
def extract_features(model, pixel_values):
    outs = model(pixel_values=pixel_values, train=False)
    # use the CLS token
    cls = outs.last_hidden_state[:, 0]
    return cls.astype(jnp.float32)  # Return JAX array, convert to numpy outside

# 4) Build feature / label arrays
def build_split(exs):
    feats, labels = [], []
    for ex in exs:
        # Convert PIL image to ensure proper format
        image = ex["image"]
        if hasattr(image, 'convert'):
            image = image.convert('RGB')

        # Process image and extract pixel values
        processed = processor(images=image, return_tensors="np")
        pix = processed["pixel_values"]

        f = extract_features(model, pix)
        feats.append(np.asarray(f[0]))  # Convert to numpy here, outside the JIT function
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
