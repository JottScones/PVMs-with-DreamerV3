import jax
import jax.numpy as jnp
import numpy as np
from datasets import load_dataset
from transformers import FlaxDinov2Model, AutoImageProcessor
from sklearn.neighbors import KNeighborsClassifier
from functools import partial


def main():
    # ----------------------------------
    # 1) Load your fine-tuned Flax DINOv2
    # ----------------------------------
    model = FlaxDinov2Model.from_pretrained(
        "dino-ft-checkpoint",
        dtype=jax.numpy.bfloat16,
        _do_init=False,            # skip random init
    )

    # ----------------------------------
    # 2) Load the image processor
    # ----------------------------------
    processor = AutoImageProcessor.from_pretrained("dino-ft-checkpoint")

    # ----------------------------------
    # 3) JIT-compile a feature extractor
    # ----------------------------------
    # We'll take the [CLS] token (index 0) as our vector.
    @partial(jax.jit, static_argnums=(0,))
    def extract_features(params, pixel_values):
        outputs = model(pixel_values=pixel_values, params=params, train=False)
        cls_feat = outputs.last_hidden_state[:, 0]       # (batch, hidden)
        # Cast to float32 on host for sklearn
        return np.asarray(cls_feat.astype(jnp.float32))

    # ----------------------------------
    # 4) Load a small ImageNet split
    # ----------------------------------
    # You can adjust "imagenet-1k" to your actual HF dataset name or use tfds.
    ds = load_dataset("imagenet-1k", split="train[:5%]")

    # ----------------------------------
    # 5) Preprocess into pixel_values + label
    # ----------------------------------
    def preprocess(batch):
        imgs = batch["image"]
        proc = processor(images=imgs, return_tensors="np")
        return {"pixel_values": proc["pixel_values"], "label": batch["label"]}

    ds = ds.map(preprocess, batched=True, batch_size=32)

    # ----------------------------------
    # 6) Split into train / eval subsets
    # ----------------------------------
    train_ds = ds.shuffle(seed=0).select(range(2000))
    eval_ds = ds.shuffle(seed=0).select(range(2000, 2500))

    # ----------------------------------
    # 7) Build feature matrices
    # ----------------------------------
    def build_features(split):
        feats = []
        labels = []
        for ex in split:
            # shape (1, C, H, W) or (batch, C, H, W)
            pv = ex["pixel_values"]
            # Ensure we pass a batch dimension
            pv_batch = pv if pv.ndim == 4 else pv[None, ...]
            f = extract_features(model.params, pv_batch)
            feats.append(f[0])
            labels.append(ex["label"])
        return np.stack(feats), np.array(labels)

    X_train, y_train = build_features(train_ds)
    X_eval,  y_eval = build_features(eval_ds)

    # ----------------------------------
    # 8) Train & evaluate k-NN
    # ----------------------------------
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    print("Training k-NN on ImageNet subset...")
    knn.fit(X_train, y_train)
    acc = knn.score(X_eval, y_eval)

    print(f"✅ k-NN accuracy on ImageNet subset: {acc * 100:.2f}%")
    print("Higher accuracy → less forgetting of ImageNet semantics.")


if __name__ == "__main__":
    main()
