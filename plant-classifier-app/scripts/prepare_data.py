"""
Download a PlantVillage subset from HuggingFace, compute CLIP embeddings,
train an MLP classifier with hyperparameter search, and build a FAISS retrieval index.

Run once before starting the app:
    python scripts/prepare_data.py

Changes from v1:
- Backbone upgraded to ViT-L-14 (768-dim embeddings, better separation)
- SAMPLES_PER_CLASS raised to 500 for more signal
- Stratified 80/20 train/test split — reports honest test accuracy
- Training set balanced via oversampling before fitting
- Linear probe replaced with MLPClassifier + GridSearchCV over hidden sizes and alpha
"""
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import open_clip
import torch
from datasets import load_dataset
from PIL import Image
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample

SELECTED_LABELS = [
    "Apple___Apple_scab",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
]

SAMPLES_PER_CLASS = 500   # used for classifier training
GALLERY_PER_CLASS = 100   # images saved to disk and indexed in FAISS
MODEL_NAME = "ViT-L-14"
PRETRAINED = "openai"
MODEL_DIR = Path("models")
STATIC_DIR = Path("static/sample_images")


def balance_classes(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Oversample every class up to the size of the largest class."""
    classes, counts = np.unique(y, return_counts=True)
    max_count = int(counts.max())
    X_parts, y_parts = [], []
    for cls in classes:
        mask = y == cls
        X_cls, y_cls = resample(X[mask], y[mask], n_samples=max_count, random_state=42)
        X_parts.append(X_cls)
        y_parts.append(y_cls)
    return np.vstack(X_parts), np.concatenate(y_parts)


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading CLIP {MODEL_NAME}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model = model.to(device)
    model.eval()

    print("Downloading BrandonFors/Plant-Diseases-PlantVillage-Dataset...")
    ds = load_dataset("BrandonFors/Plant-Diseases-PlantVillage-Dataset", split="train")
    label_names: list[str] = ds.features["label"].names
    print(f"Dataset has {len(label_names)} classes and {len(ds)} images.")

    for label in SELECTED_LABELS:
        if label not in label_names:
            print(f"WARNING: '{label}' not found in dataset. Available: {label_names}")
            sys.exit(1)

    ordered_labels = sorted(SELECTED_LABELS)
    label_to_idx = {name: i for i, name in enumerate(ordered_labels)}
    selected_int_set = {label_names.index(n) for n in SELECTED_LABELS}

    print(f"Collecting up to {SAMPLES_PER_CLASS} images/class ({SAMPLES_PER_CLASS * len(SELECTED_LABELS)} total)...")
    ds_filtered = ds.filter(lambda x: x["label"] in selected_int_set)

    per_class_count: dict[str, int] = defaultdict(int)
    all_embeddings: list[np.ndarray] = []
    all_labels: list[int] = []
    faiss_embeddings: list[np.ndarray] = []
    image_paths: list[str] = []

    for item in ds_filtered:
        label_name: str = label_names[item["label"]]
        if per_class_count[label_name] >= SAMPLES_PER_CLASS:
            continue

        img: Image.Image = item["image"].convert("RGB")
        idx = per_class_count[label_name]

        tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        emb = feat.cpu().numpy().astype("float32")[0]

        all_embeddings.append(emb)
        all_labels.append(label_to_idx[label_name])

        if idx < GALLERY_PER_CLASS:
            save_path = STATIC_DIR / f"{label_name}_{idx}.jpg"
            img.save(save_path, quality=85)
            image_paths.append(f"static/sample_images/{label_name}_{idx}.jpg")
            faiss_embeddings.append(emb)

        per_class_count[label_name] += 1

        total = sum(per_class_count.values())
        if total % 200 == 0:
            print(f"  {total} / {SAMPLES_PER_CLASS * len(SELECTED_LABELS)}", end="\r")

        if all(v >= SAMPLES_PER_CLASS for v in per_class_count.values()):
            break

    print(f"\nCollected {len(all_embeddings)} embeddings total.")

    X = np.array(all_embeddings)
    y = np.array(all_labels)

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Split — train: {len(X_train)}, test: {len(X_test)}")

    # Balance training set via oversampling
    X_train_bal, y_train_bal = balance_classes(X_train, y_train)
    print(f"Balanced train set: {len(X_train_bal)} samples")

    # MLP with grid search
    print("Running GridSearchCV over MLP hyperparameters (3-fold CV)...")
    param_grid = {
        "hidden_layer_sizes": [(256,), (256, 128), (512, 256)],
        "alpha": [1e-4, 1e-3, 1e-2],
    }
    search = GridSearchCV(
        MLPClassifier(max_iter=500, early_stopping=True, random_state=42),
        param_grid,
        cv=3,
        n_jobs=-1,
        verbose=1,
        scoring="accuracy",
    )
    search.fit(X_train_bal, y_train_bal)

    print(f"Best params:    {search.best_params_}")
    print(f"Best CV acc:    {search.best_score_:.3f}")
    print(f"Test accuracy:  {search.best_estimator_.score(X_test, y_test):.3f}")

    with open(MODEL_DIR / "linear_probe.pkl", "wb") as f:
        pickle.dump(search.best_estimator_, f)

    with open(MODEL_DIR / "labels.json", "w") as f:
        json.dump(ordered_labels, f, indent=2)

    print("Building FAISS index on gallery embeddings...")
    X_faiss = np.array(faiss_embeddings)
    dim = X_faiss.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X_faiss)
    faiss.write_index(index, str(MODEL_DIR / "faiss.index"))

    with open(MODEL_DIR / "faiss_meta.json", "w") as f:
        json.dump(image_paths, f)

    print(f"Done. Models saved to {MODEL_DIR}/")
    print("  linear_probe.pkl  — MLP classifier (best from grid search)")
    print("  labels.json       — class name mapping")
    print("  faiss.index       — FAISS nearest-neighbour index ({dim}-dim)")
    print("  faiss_meta.json   — image path metadata")


if __name__ == "__main__":
    main()
