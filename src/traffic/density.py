"""Traffic density estimation."""

from __future__ import annotations

import numpy as np


def estimate_density(vehicle_boxes, roi_area: float) -> float:
    """Estimate density as total box area divided by the region area."""
    if roi_area <= 0 or vehicle_boxes is None:
        return 0.0

    boxes = np.asarray(vehicle_boxes, dtype=float)
    if boxes.size == 0:
        return 0.0

    widths = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
    heights = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return float(np.clip((widths * heights).sum() / roi_area, 0.0, 1.0))
