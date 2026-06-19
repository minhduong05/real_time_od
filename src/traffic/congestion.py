"""Congestion warning rules."""


def is_congested(density: float, duration_seconds: float, threshold: float = 0.8) -> bool:
    """Return True when congestion conditions are met."""
    return density > threshold and duration_seconds >= 30

