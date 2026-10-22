def _smoothstep(step: int, start: int, end: int) -> float:
    x = min(max((step - start) / (end - start), 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def semantic_speed_range_v5_3(step: int) -> tuple[float, float]:
    if step < 0:
        raise ValueError("step must be non-negative")
    if step >= 16_000:
        return 0.15, 0.40
    progress = _smoothstep(step, 0, 16_000)
    return 0.10 + 0.05 * progress, 0.20 + 0.20 * progress
