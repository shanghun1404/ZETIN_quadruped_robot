def _smoothstep(step: int, start: int, end: int) -> float:
    x = min(max((step - start) / (end - start), 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def semantic_speed_range(step: int) -> tuple[float, float]:
    if step <= 16_000:
        pilot = _smoothstep(step, 0, 16_000)
        return 0.03 + 0.05 * pilot, 0.06 + 0.09 * pilot
    if step >= 80_000:
        return 0.15, 0.40
    full = _smoothstep(step, 16_000, 80_000)
    return 0.08 + 0.07 * full, 0.15 + 0.25 * full
