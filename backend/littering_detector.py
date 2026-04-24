"""
LitteringDetector v2 — Hybrid state machine for temporal illegal-dumping detection.

Algorithm overview
──────────────────
Two complementary detection modes run simultaneously:

  MODE A — Zone-based (original, robust):
    1. Track trash objects across frames (ByteTrack persistent track_ids).
    2. Track persons across frames (yolov8n person detector).
    3. When a person leaves the frame, start a MONITORING window (10s).
    4. If a NEW trash track_id appears inside the area where the person was → EVENT.

  MODE B — Distance-based (new):
    1. When a trash object appears near a person (centre within person bbox),
       mark it as ATTACHED to that person_id.
    2. Track distance between trash centre and person centre each frame.
    3. When trash becomes static (jitter < JITTER_THRESHOLD for STATIC_FRAMES)
       AND person distance increases beyond SEPARATION_DIST_M → SEPARATING.
    4. When distance > ABANDON_DIST_M OR person lost > LOST_TIMEOUT_S → EVENT.

Distance estimation
───────────────────
Scale is estimated from the person bounding box height:
    pixels_per_metre ≈ person_bbox_height / PERSON_HEIGHT_M (1.7 m)

This is camera-agnostic — works for both close webcam and outdoor CCTV without
requiring explicit camera calibration.

Usage
─────
    detector = LitteringDetector(fps=25)
    for frame, trash_dets, person_boxes in video_stream:
        event = detector.update(frame, trash_dets, person_boxes)
        if event:
            # LitteringEvent — save clip, store in DB, send alert
            ...

    detector.reset()   # call between independent video sessions
"""

from __future__ import annotations

import math
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

PERSON_HEIGHT_M       = 1.70   # assumed average adult height (metres)
SEPARATION_DIST_M     = 1.50   # person distance that triggers SEPARATING state
ABANDON_DIST_M        = 3.00   # person distance that triggers ABANDONED event
JITTER_THRESHOLD_PX   = 15.0   # max trash centre movement (px) to count as static
STATIC_FRAMES_NEEDED  = 8      # consecutive frames trash must be static → DROPPED
LOST_TIMEOUT_S        = 2.0    # seconds person can be absent before triggering ABANDONED


# ─────────────────────────────────────────────────────────────────────────────
# Zone-based state (MODE A)
# ─────────────────────────────────────────────────────────────────────────────

class DetectorState(Enum):
    CLEAR          = auto()
    PERSON_PRESENT = auto()
    MONITORING     = auto()


# ─────────────────────────────────────────────────────────────────────────────
# Distance-based per-trash state (MODE B)
# ─────────────────────────────────────────────────────────────────────────────

class TrashRelState(Enum):
    """State of the relationship between ONE trash object and its owner person."""
    NEARBY     = auto()   # trash near person, possibly in hand
    DROPPED    = auto()   # trash static on surface, person still close
    SEPARATING = auto()   # person moving away from static trash
    ABANDONED  = auto()   # person clearly walked away → fire event


@dataclass
class TrashRelTracker:
    """Tracks the evolving relationship between one trash track_id and a person."""
    trash_id:       int
    person_id:      int
    owner_box:      tuple[int, int, int, int]
    state:          TrashRelState = TrashRelState.NEARBY
    static_frames:  int           = 0
    last_trash_cx:  float         = 0.0
    last_trash_cy:  float         = 0.0
    person_lost_at: Optional[float] = None
    max_distance_m: float         = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Shared data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PersonZone:
    """Records the last known position of a person (used after they leave)."""
    x1: int
    y1: int
    x2: int
    y2: int
    last_seen_frame: int

    def expanded(self, fraction: float = 0.20) -> tuple[int, int, int, int]:
        dw = int((self.x2 - self.x1) * fraction)
        dh = int((self.y2 - self.y1) * fraction)
        return (self.x1 - dw, self.y1 - dh, self.x2 + dw, self.y2 + dh)


@dataclass
class LitteringEvent:
    """Fired when illegal dumping is detected (MODE A or MODE B)."""
    detected_at_ts:          float
    frame_idx:               int
    material:                str
    det_score:               float
    trash_box:               tuple[int, int, int, int]
    person_box:              tuple[int, int, int, int]
    person_present:          bool  = True
    clip_frames:             list[np.ndarray] = field(default_factory=list)
    thumbnail:               Optional[np.ndarray] = None
    latitude:                Optional[float] = None
    longitude:               Optional[float] = None
    # Evidence fields (v2)
    incident_uid:            str   = field(default_factory=lambda: str(uuid.uuid4()))
    owner_person_id:         Optional[int]   = None
    distance_at_abandonment: Optional[float] = None
    detection_method:        str  = "zone"   # "zone" | "distance"


# ─────────────────────────────────────────────────────────────────────────────
# Main detector
# ─────────────────────────────────────────────────────────────────────────────

class LitteringDetector:
    """
    Hybrid littering detector combining zone-based (MODE A) and
    distance-based (MODE B) state machines.

    Args:
        fps:               Expected video fps.
        monitor_seconds:   MODE A — window to watch for new trash after person leaves.
        pre_event_seconds: How many seconds of frames to keep in circular buffer.
        person_conf:       Confidence threshold for person detector.
        zone_expand:       Person bbox expansion fraction for zone check (MODE A).
    """

    def __init__(
        self,
        fps: float = 25.0,
        monitor_seconds: float = 10.0,
        pre_event_seconds: float = 5.0,
        person_conf: float = 0.40,
        zone_expand: float = 0.20,
    ) -> None:
        self.fps               = max(fps, 1.0)
        self.monitor_frames    = int(monitor_seconds * self.fps)
        self.pre_buffer_frames = int(pre_event_seconds * self.fps)
        self.person_conf       = person_conf
        self.zone_expand       = zone_expand

        # MODE A state
        self.state: DetectorState          = DetectorState.CLEAR
        self.frame_idx: int                = 0
        self._frame_buffer: deque[np.ndarray] = deque(maxlen=self.pre_buffer_frames)
        self._known_trash_ids: set[int]    = set()
        self._person_zones: list[PersonZone] = []
        self._monitoring_start: int        = 0

        # Post-event clip capture
        self._capture_post: bool           = False
        self._post_frames_needed: int      = 0
        self._post_buffer: list[np.ndarray] = []
        self._pending_event: Optional[LitteringEvent] = None

        # MODE B state
        self._rel_trackers: dict[int, TrashRelTracker] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def update(
        self,
        frame: np.ndarray,
        trash_detections: list[dict],
        person_boxes: list[tuple[int, int, int, int]],
        person_ids: Optional[list[int]] = None,
    ) -> Optional[LitteringEvent]:
        """
        Process one frame.

        Args:
            frame:            Raw BGR frame.
            trash_detections: Each dict must contain 'track_id', 'box',
                              'material_name', 'det_score'.
            person_boxes:     (x1,y1,x2,y2) list from detect_persons().
            person_ids:       Optional ByteTrack IDs for persons (same order).

        Returns:
            LitteringEvent if detected this frame, else None.
        """
        self.frame_idx += 1
        self._frame_buffer.append(frame.copy())

        current_trash_ids = {d["track_id"] for d in trash_detections}

        # Post-event clip finalisation
        if self._capture_post and self._pending_event is not None:
            self._post_buffer.append(frame.copy())
            self._post_frames_needed -= 1
            if self._post_frames_needed <= 0:
                self._pending_event.clip_frames += self._post_buffer
                self._capture_post   = False
                self._pending_event  = None
                self._post_buffer    = []

        # MODE B — distance-based (runs only when persons visible)
        if person_boxes:
            event = self._update_distance_trackers(
                frame, trash_detections, person_boxes, person_ids or []
            )
            if event is not None:
                self._start_post_capture(event)
                return event

        # MODE A — zone-based
        if self.state == DetectorState.CLEAR:
            if person_boxes:
                self._enter_person_present(person_boxes, current_trash_ids)

        elif self.state == DetectorState.PERSON_PRESENT:
            if person_boxes:
                self._update_person_zones(person_boxes)
                self._known_trash_ids.update(current_trash_ids)
            else:
                self._enter_monitoring()

        elif self.state == DetectorState.MONITORING:
            if person_boxes:
                self._enter_person_present(person_boxes, current_trash_ids)
            elif self.frame_idx - self._monitoring_start > self.monitor_frames:
                self._enter_clear()
            else:
                new_ids = current_trash_ids - self._known_trash_ids
                if new_ids:
                    event = self._check_zone_overlap(new_ids, trash_detections, frame)
                    if event is not None:
                        self._known_trash_ids.update(current_trash_ids)
                        self._enter_clear()
                        self._start_post_capture(event)
                        return event
                self._known_trash_ids.update(current_trash_ids)

        return None

    def reset(self) -> None:
        """Reset all state — call between independent video sessions."""
        self.state               = DetectorState.CLEAR
        self.frame_idx           = 0
        self._frame_buffer.clear()
        self._known_trash_ids.clear()
        self._person_zones.clear()
        self._monitoring_start   = 0
        self._capture_post       = False
        self._post_frames_needed = 0
        self._post_buffer        = []
        self._pending_event      = None
        self._rel_trackers.clear()

    @property
    def current_state(self) -> str:
        return self.state.name

    @property
    def monitoring_progress(self) -> float:
        """0.0–1.0 fraction of monitor window elapsed (MODE A MONITORING only)."""
        if self.state != DetectorState.MONITORING:
            return 0.0
        elapsed = self.frame_idx - self._monitoring_start
        return min(elapsed / max(self.monitor_frames, 1), 1.0)

    # ─────────────────────────────────────────────────────────────────────────
    # MODE B helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _update_distance_trackers(
        self,
        frame: np.ndarray,
        trash_dets: list[dict],
        person_boxes: list[tuple[int, int, int, int]],
        person_ids: list[int],
    ) -> Optional[LitteringEvent]:
        now = time.time()
        scale = self._estimate_scale(person_boxes)

        pid_map: dict[int, tuple[int, int, int, int]] = {
            (person_ids[i] if i < len(person_ids) else -(i + 1)): pb
            for i, pb in enumerate(person_boxes)
        }

        current_trash_ids = {d["track_id"] for d in trash_dets}
        trash_by_id = {d["track_id"]: d for d in trash_dets}

        # Expire trackers whose trash disappeared
        for tid in list(self._rel_trackers.keys()):
            if tid not in current_trash_ids:
                del self._rel_trackers[tid]

        for det in trash_dets:
            tid  = det["track_id"]
            tb   = det["box"]
            tcx  = (tb[0] + tb[2]) / 2.0
            tcy  = (tb[1] + tb[3]) / 2.0

            nearest_pid, nearest_pb, dist_m = self._nearest_person(
                tcx, tcy, pid_map, scale
            )
            if nearest_pid is None or nearest_pb is None:
                continue

            tracker = self._rel_trackers.get(tid)
            if tracker is None:
                if dist_m < SEPARATION_DIST_M:
                    self._rel_trackers[tid] = TrashRelTracker(
                        trash_id=tid,
                        person_id=nearest_pid,
                        owner_box=nearest_pb,
                        last_trash_cx=tcx,
                        last_trash_cy=tcy,
                    )
                continue

            tracker.owner_box = nearest_pb
            is_static = math.hypot(
                tcx - tracker.last_trash_cx,
                tcy - tracker.last_trash_cy,
            ) < JITTER_THRESHOLD_PX
            tracker.last_trash_cx = tcx
            tracker.last_trash_cy = tcy
            tracker.max_distance_m = max(tracker.max_distance_m, dist_m)

            if tracker.state == TrashRelState.NEARBY:
                if is_static:
                    tracker.static_frames += 1
                    if (
                        tracker.static_frames >= STATIC_FRAMES_NEEDED
                        and dist_m < SEPARATION_DIST_M
                    ):
                        tracker.state         = TrashRelState.DROPPED
                        tracker.static_frames = 0
                else:
                    tracker.static_frames = 0

            elif tracker.state == TrashRelState.DROPPED:
                if dist_m >= SEPARATION_DIST_M:
                    tracker.state = TrashRelState.SEPARATING

            elif tracker.state == TrashRelState.SEPARATING:
                if dist_m >= ABANDON_DIST_M:
                    tracker.state = TrashRelState.ABANDONED
                    event = self._build_distance_event(frame, det, tracker, dist_m, nearest_pb)
                    del self._rel_trackers[tid]
                    return event

        # Person-lost timeout for SEPARATING trackers
        for tid, tracker in list(self._rel_trackers.items()):
            if tracker.state != TrashRelState.SEPARATING:
                continue
            if tracker.person_id not in pid_map:
                if tracker.person_lost_at is None:
                    tracker.person_lost_at = now
                elif now - tracker.person_lost_at >= LOST_TIMEOUT_S:
                    det = trash_by_id.get(tid)
                    if det:
                        event = self._build_distance_event(
                            frame, det, tracker,
                            tracker.max_distance_m, tracker.owner_box,
                        )
                        del self._rel_trackers[tid]
                        return event
            else:
                tracker.person_lost_at = None

        return None

    def _build_distance_event(
        self,
        frame: np.ndarray,
        det: dict,
        tracker: TrashRelTracker,
        distance_m: float,
        person_box: tuple[int, int, int, int],
    ) -> LitteringEvent:
        thumbnail = _make_thumbnail(
            frame, det["box"],
            PersonZone(*person_box, self.frame_idx),
            label=f"ABANDON {distance_m:.1f}m",
        )
        return LitteringEvent(
            detected_at_ts          = time.time(),
            frame_idx               = self.frame_idx,
            material                = det["material_name"],
            det_score               = det["det_score"],
            trash_box               = det["box"],
            person_box              = person_box,
            clip_frames             = list(self._frame_buffer),
            thumbnail               = thumbnail,
            owner_person_id         = tracker.person_id if tracker.person_id >= 0 else None,
            distance_at_abandonment = round(distance_m, 3),
            detection_method        = "distance",
        )

    @staticmethod
    def _estimate_scale(person_boxes: list[tuple[int, int, int, int]]) -> float:
        """pixels/metre estimate from the tallest person bbox height."""
        tallest = max((pb[3] - pb[1] for pb in person_boxes), default=0)
        return tallest / PERSON_HEIGHT_M if tallest > 0 else 50.0

    @staticmethod
    def _nearest_person(
        cx: float, cy: float,
        pid_map: dict[int, tuple[int, int, int, int]],
        scale: float,
    ) -> tuple[Optional[int], Optional[tuple], float]:
        best_pid, best_pb, best_d = None, None, float("inf")
        for pid, pb in pid_map.items():
            pcx = (pb[0] + pb[2]) / 2.0
            pcy = (pb[1] + pb[3]) / 2.0
            d_m = math.hypot(cx - pcx, cy - pcy) / max(scale, 1.0)
            if d_m < best_d:
                best_d, best_pid, best_pb = d_m, pid, pb
        return best_pid, best_pb, best_d

    # ─────────────────────────────────────────────────────────────────────────
    # MODE A helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _enter_person_present(
        self, person_boxes: list, current_trash_ids: set
    ) -> None:
        self.state = DetectorState.PERSON_PRESENT
        self._person_zones = [
            PersonZone(x1, y1, x2, y2, self.frame_idx)
            for (x1, y1, x2, y2) in person_boxes
        ]
        self._known_trash_ids.update(current_trash_ids)

    def _update_person_zones(self, person_boxes: list) -> None:
        self._person_zones = [
            PersonZone(x1, y1, x2, y2, self.frame_idx)
            for (x1, y1, x2, y2) in person_boxes
        ]

    def _enter_monitoring(self) -> None:
        self.state             = DetectorState.MONITORING
        self._monitoring_start = self.frame_idx

    def _enter_clear(self) -> None:
        self.state = DetectorState.CLEAR
        self._person_zones.clear()

    def _check_zone_overlap(
        self,
        new_trash_ids: set[int],
        trash_detections: list[dict],
        frame: np.ndarray,
    ) -> Optional[LitteringEvent]:
        for det in trash_detections:
            if det["track_id"] not in new_trash_ids:
                continue
            bx1, by1, bx2, by2 = det["box"]
            cx = (bx1 + bx2) // 2
            cy = (by1 + by2) // 2
            for zone in self._person_zones:
                zx1, zy1, zx2, zy2 = zone.expanded(self.zone_expand)
                if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                    thumbnail = _make_thumbnail(frame, det["box"], zone)
                    return LitteringEvent(
                        detected_at_ts   = time.time(),
                        frame_idx        = self.frame_idx,
                        material         = det["material_name"],
                        det_score        = det["det_score"],
                        trash_box        = det["box"],
                        person_box       = (zone.x1, zone.y1, zone.x2, zone.y2),
                        thumbnail        = thumbnail,
                        detection_method = "zone",
                    )
        return None

    def _start_post_capture(self, event: LitteringEvent) -> None:
        self._capture_post       = True
        self._post_frames_needed = int(3.0 * self.fps)
        self._post_buffer        = []
        event.clip_frames        = list(self._frame_buffer)
        self._pending_event      = event


# ─────────────────────────────────────────────────────────────────────────────
# Utility — thumbnail generation
# ─────────────────────────────────────────────────────────────────────────────

def _make_thumbnail(
    frame: np.ndarray,
    trash_box: tuple[int, int, int, int],
    zone: PersonZone,
    size: tuple[int, int] = (320, 240),
    label: str = "NEW LITTER!",
) -> np.ndarray:
    thumb = frame.copy()

    _draw_dashed_rect(thumb, (zone.x1, zone.y1), (zone.x2, zone.y2),
                      (0, 165, 255), 2, dash_len=12)
    cv2.putText(thumb, "Person zone", (zone.x1, max(zone.y1 - 6, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)

    tx1, ty1, tx2, ty2 = trash_box
    cv2.rectangle(thumb, (tx1, ty1), (tx2, ty2), (0, 0, 255), 2)
    cv2.putText(thumb, label, (tx1, max(ty1 - 6, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.rectangle(thumb, (0, 0), (thumb.shape[1], 28), (0, 0, 200), -1)
    cv2.putText(thumb, "LITTERING DETECTED", (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    return cv2.resize(thumb, size, interpolation=cv2.INTER_AREA)


def _draw_dashed_rect(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    dash_len: int = 10,
) -> None:
    x1, y1 = pt1
    x2, y2 = pt2
    for edge in [
        ((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1)),
    ]:
        (ex1, ey1), (ex2, ey2) = edge
        length = int(((ex2 - ex1) ** 2 + (ey2 - ey1) ** 2) ** 0.5)
        if length == 0:
            continue
        dx, dy = (ex2 - ex1) / length, (ey2 - ey1) / length
        seg, draw = 0, True
        while seg < length:
            end = min(seg + dash_len, length)
            if draw:
                cv2.line(img,
                         (int(ex1 + dx * seg),  int(ey1 + dy * seg)),
                         (int(ex1 + dx * end),  int(ey1 + dy * end)),
                         color, thickness)
            seg  = end
            draw = not draw
