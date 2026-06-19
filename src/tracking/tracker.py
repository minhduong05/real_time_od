"""Tracking wrapper for assigning stable IDs to detected vehicles."""


class VehicleTracker:
    """Vehicle tracking interface."""

    def update(self, detections):
        """Update tracks from detections."""
        raise NotImplementedError

