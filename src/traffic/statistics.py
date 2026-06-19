"""Vehicle type statistics."""

from __future__ import annotations

from collections import Counter


def count_by_class(class_ids, names: dict[int, str] | None = None) -> dict[str, int]:
    """Count detections by class id, optionally mapping ids to names."""
    counts = Counter(int(class_id) for class_id in class_ids)
    if not names:
        return {str(class_id): count for class_id, count in sorted(counts.items())}
    return {names.get(class_id, str(class_id)): count for class_id, count in sorted(counts.items())}
