from src.recommend import fit_model, hit_rate_at_k, precision_at_k

MIN_PRECISION = 0.20
MIN_HIT_RATE = 0.50


def test_precision_and_hit_rate_gate() -> None:
    model, clicks = fit_model()
    precision = precision_at_k(model, clicks, k=5)
    hit_rate = hit_rate_at_k(model, clicks, k=5)
    assert precision >= MIN_PRECISION, f"precision@5 {precision:.3f} < {MIN_PRECISION}"
    assert hit_rate >= MIN_HIT_RATE, f"hit-rate@5 {hit_rate:.3f} < {MIN_HIT_RATE}"
