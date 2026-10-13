import json
from pathlib import Path

import numpy as np

from .config import EMBEDDING_MODEL
from .metrics import get_embedder


class EmbeddingIndex:
    def __init__(self, model_name=EMBEDDING_MODEL):
        self.model_name = model_name
        self.embeddings = None
        self.payloads = []

    def encode(self, texts, batch_size=64):
        model = get_embedder(self.model_name)
        return model.encode(list(texts), normalize_embeddings=True, batch_size=batch_size,
                            show_progress_bar=False).astype(np.float32)

    def build(self, texts, payloads):
        self.embeddings = self.encode(texts)
        self.payloads = list(payloads)
        return self

    def search(self, queries, k=3):
        q = self.encode(queries)
        scores = q @ self.embeddings.T
        top = np.argsort(-scores, axis=1)[:, :k]
        return [[(float(scores[i, j]), self.payloads[j]) for j in row] for i, row in enumerate(top)]

    def max_similarity(self, queries):
        return (self.encode(queries) @ self.embeddings.T).max(axis=1)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.embeddings)
        with open(directory / "payloads.jsonl", "w") as f:
            for p in self.payloads:
                f.write(json.dumps(p) + "\n")
        (directory / "meta.json").write_text(json.dumps({"model_name": self.model_name}))

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text())
        index = cls(meta["model_name"])
        index.embeddings = np.load(directory / "embeddings.npy")
        with open(directory / "payloads.jsonl") as f:
            index.payloads = [json.loads(line) for line in f]
        return index


def build_knowledge_index(train_df):
    from .data import knowledge_text

    records = train_df.to_dict("records")
    texts = [r["question"] for r in records]
    payloads = [{"sample_id": r["sample_id"], "text": knowledge_text(r)} for r in records]
    return EmbeddingIndex().build(texts, payloads)


def retrieve_references(index, questions, k=3, exclude_ids=None):
    results = index.search(questions, k=k + 1)
    refs = []
    for i, hits in enumerate(results):
        skip = exclude_ids[i] if exclude_ids else None
        refs.append([p["text"] for _, p in hits if p["sample_id"] != skip][:k])
    return refs
