from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


@dataclass
class RecModel:
    item_ids: np.ndarray
    item_vectors: np.ndarray
    popularity: np.ndarray

    def recommend(self, user_vec: np.ndarray, k: int = 5) -> list[int]:
        scores = cosine_similarity(user_vec.reshape(1, -1), self.item_vectors)[0]
        scores = scores + 0.05 * self.popularity
        top = np.argsort(scores)[::-1][:k]
        return [int(self.item_ids[i]) for i in top]


def fit_model(
    n_users: int = 80,
    n_items: int = 40,
    n_factors: int = 8,
    seed: int = 7,
) -> tuple[RecModel, np.ndarray]:
    rng = np.random.default_rng(seed)
    user_f = rng.normal(scale=0.4, size=(n_users, n_factors))
    item_f = rng.normal(scale=0.4, size=(n_items, n_factors))
    truth = user_f @ item_f.T
    clicks = (truth > np.quantile(truth, 0.7)).astype(np.float64)
    item_vectors = normalize(item_f)
    popularity = clicks.mean(axis=0)
    model = RecModel(
        item_ids=np.arange(n_items),
        item_vectors=item_vectors,
        popularity=popularity,
    )
    return model, clicks


def precision_at_k(model: RecModel, clicks: np.ndarray, k: int = 5) -> float:
    hits = 0.0
    n_users = clicks.shape[0]
    for u in range(n_users):
        liked = set(np.where(clicks[u] > 0)[0].tolist())
        if not liked:
            continue
        user_vec = clicks[u] @ model.item_vectors
        recs = model.recommend(user_vec, k=k)
        hits += len(liked.intersection(recs)) / float(k)
    return hits / float(n_users)


def hit_rate_at_k(model: RecModel, clicks: np.ndarray, k: int = 5) -> float:
    hits = 0.0
    n_users = clicks.shape[0]
    for u in range(n_users):
        liked = set(np.where(clicks[u] > 0)[0].tolist())
        if not liked:
            continue
        user_vec = clicks[u] @ model.item_vectors
        recs = model.recommend(user_vec, k=k)
        hits += 1.0 if liked.intersection(recs) else 0.0
    return hits / float(n_users)
