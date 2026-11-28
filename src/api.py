from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.recommend import fit_model

app = FastAPI(title="ecommerce-support-agent")
MODEL, _CLICKS = fit_model()


class PredictIn(BaseModel):
    user_clicks: list[int] = Field(default_factory=list)
    k: int = 5


class PredictOut(BaseModel):
    items: list[int]


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "n_items": int(len(MODEL.item_ids))}


@app.post("/predict", response_model=PredictOut)
def predict(body: PredictIn) -> PredictOut:
    vec = np.zeros(MODEL.item_vectors.shape[1], dtype=float)
    for item in body.user_clicks:
        if 0 <= item < len(MODEL.item_ids):
            vec = vec + MODEL.item_vectors[item]
    if not np.any(vec):
        vec = MODEL.popularity @ MODEL.item_vectors
    recs = MODEL.recommend(vec, k=body.k)
    return PredictOut(items=recs)
