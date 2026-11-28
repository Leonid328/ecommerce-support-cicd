from __future__ import annotations

import sys

import numpy as np

from src.recommend import fit_model


def ks_mean_shift(a: np.ndarray, b: np.ndarray, max_shift: float = 0.35) -> float:
    shift = float(np.abs(a.mean() - b.mean()))
    if shift > max_shift:
        raise SystemExit(f"drift detected: mean shift {shift:.3f} > {max_shift}")
    return shift


def main() -> None:
    _, clicks = fit_model()
    n = clicks.shape[0]
    ref = clicks[: n // 2]
    cur = clicks[n // 2 :]
    shift = ks_mean_shift(ref, cur)
    print(f"drift check ok, mean shift={shift:.4f}")


if __name__ == "__main__":
    sys.exit(main())
