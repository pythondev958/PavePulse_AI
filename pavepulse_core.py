"""Framework-independent road damage analytics and temporal review segmentation.

Important: A review segment is a contiguous time interval in which a class was
seen, NOT a deduplicated physical road defect. A dashcam moves, so repeat
observations must not be represented as unique potholes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

CATEGORY_ORDER = ("pothole", "longitudinal_crack", "transverse_crack", "alligator_crack")
CLASS_TITLES = {
    "pothole": "POTHOLE",
    "longitudinal_crack": "LONGITUDINAL",
    "transverse_crack": "TRANSVERSE",
    "alligator_crack": "ALLIGATOR",
}
CLASS_SHORT = {
    "pothole": "POTHOLE",
    "longitudinal_crack": "LONG CRACK",
    "transverse_crack": "CROSS CRACK",
    "alligator_crack": "WEB CRACK",
}
# BGR palette for OpenCV. Pothole coral / long-crack amber / transverse teal / alligator lilac.
CLASS_COLORS = {
    "pothole": (101, 111, 251),
    "longitudinal_crack": (61, 194, 245),
    "transverse_crack": (206, 202, 55),
    "alligator_crack": (228, 129, 181),
}

ALIASES = {
    "d00": "longitudinal_crack",
    "d10": "transverse_crack",
    "d20": "alligator_crack",
    "d40": "pothole",
    "longitudinal_crack": "longitudinal_crack",
    "longitudinal crack": "longitudinal_crack",
    "transverse_crack": "transverse_crack",
    "transverse crack": "transverse_crack",
    "alligator_crack": "alligator_crack",
    "alligator crack": "alligator_crack",
    "pothole": "pothole",
    "potholes": "pothole",
}


def normalize_class(name: str) -> Optional[str]:
    key = str(name).strip().lower().replace("-", "_")
    return ALIASES.get(key)


def safe_bbox(xyxy, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    """Constrain a prediction box to image dimensions; reject empty boxes."""
    if width <= 0 or height <= 0 or len(xyxy) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(n) for n in xyxy)
    except (TypeError, ValueError):
        return None
    import math
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    x1 = max(0, min(width - 1, round(x1)))
    y1 = max(0, min(height - 1, round(y1)))
    x2 = max(0, min(width, round(x2)))
    y2 = max(0, min(height, round(y2)))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


@dataclass(frozen=True)
class Detection:
    cls: str
    confidence: float
    bbox: Tuple[int, int, int, int]


@dataclass(frozen=True)
class ReviewSegment:
    cls: str
    start_frame: int
    end_frame: int
    observed_frames: int
    best_confidence: float


class ReviewSegmenter:
    """Group frame-level class presence into time intervals for human review.

    One segment can contain several defects; one defect can produce several
    segments. Do not use segment counts as unique damage counts.
    """

    def __init__(self, max_gap_frames: int = 4, min_observed_frames: int = 2):
        if max_gap_frames < 0 or min_observed_frames < 1:
            raise ValueError("Invalid segmenter configuration")
        self.max_gap_frames = max_gap_frames
        self.min_observed_frames = min_observed_frames
        self._active: Dict[str, dict] = {}
        self._segments: List[ReviewSegment] = []

    def update(self, frame_id: int, detections: List[Detection]) -> None:
        observed: Dict[str, float] = {}
        for d in detections:
            if d.cls in CATEGORY_ORDER:
                observed[d.cls] = max(observed.get(d.cls, 0.0), float(d.confidence))

        for cls, active in list(self._active.items()):
            if cls not in observed and frame_id - active["last"] > self.max_gap_frames:
                self._finish(cls)

        for cls, conf in observed.items():
            if cls not in self._active:
                self._active[cls] = {
                    "start": frame_id,
                    "last": frame_id,
                    "frames": 0,
                    "best": 0.0,
                }
            active = self._active[cls]
            # Assumes update() is called exactly once per consecutive frame.
            active["last"] = frame_id
            active["frames"] += 1
            active["best"] = max(active["best"], conf)

    def _finish(self, cls: str) -> None:
        active = self._active.pop(cls)
        if active["frames"] >= self.min_observed_frames:
            self._segments.append(ReviewSegment(
                cls=cls,
                start_frame=active["start"],
                end_frame=active["last"],
                observed_frames=active["frames"],
                best_confidence=active["best"],
            ))

    def finalize(self) -> List[ReviewSegment]:
        for cls in list(self._active):
            self._finish(cls)
        return sorted(self._segments, key=lambda item: (item.start_frame, item.cls))

    @property
    def completed_count(self) -> int:
        return len(self._segments)


def summary_of_timeline(timeline: List[dict], segments: List[ReviewSegment]) -> dict:
    total = len(timeline)
    return {
        "frames_processed": total,
        "frames_with_detections": sum(row["visible_detections"] > 0 for row in timeline),
        "max_simultaneous_detections": max((row["visible_detections"] for row in timeline), default=0),
        "mean_simultaneous_detections": round(
            sum(row["visible_detections"] for row in timeline) / total, 3
        ) if total else 0,
        "review_segments": len(segments),
    }
