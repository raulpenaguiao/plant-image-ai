import json

import faiss
import numpy as np


class FAISSRetrieval:
    def __init__(
        self,
        index_path: str = "models/faiss.index",
        meta_path: str = "models/faiss_meta.json",
    ):
        self.index = faiss.read_index(index_path)
        with open(meta_path) as f:
            self.meta: list[str] = json.load(f)

    def query(self, embedding: np.ndarray, k: int = 5) -> list[str]:
        _, indices = self.index.search(embedding, k)
        return [self.meta[i] for i in indices[0] if i >= 0]
