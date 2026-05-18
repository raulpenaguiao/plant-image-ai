import json
import pickle
from pathlib import Path

import numpy as np


class LinearProbeClassifier:
    def __init__(
        self,
        model_path: str = "models/linear_probe.pkl",
        labels_path: str = "models/labels.json",
    ):
        with open(model_path, "rb") as f:
            self.clf = pickle.load(f)
        with open(labels_path) as f:
            self.labels = json.load(f)

    def predict(self, embedding: np.ndarray, top_k: int = 5) -> dict:
        proba = self.clf.predict_proba(embedding)[0]
        top_indices = np.argsort(proba)[::-1][:top_k]
        return {
            "predicted_class": self.labels[top_indices[0]],
            "confidence": float(proba[top_indices[0]]),
            "top_k": [
                {"label": self.labels[i], "score": float(proba[i])}
                for i in top_indices
            ],
        }
