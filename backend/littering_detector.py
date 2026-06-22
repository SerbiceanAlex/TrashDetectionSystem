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

PERSON_HEIGHT_M       = 1.70   # assumed average person height (metres)
SEPARATION_DIST_M     = 1.50   # person distance that triggers SEPARATING state
ABANDON_DIST_M        = 2.00   # person distance that triggers ABANDONED event (was 3.0 — nerealist: omul aruncă și face 1–2 pași)
JITTER_THRESHOLD_PX   = 15.0   # max trash centre movement (px) to count as static
STATIC_FRAMES_NEEDED  = 8      # consecutive frames trash must be static → DROPPED
LOST_TIMEOUT_S        = 5.0    # seconds person can be absent before ABANDONED (was 2.0 — too aggressive)
# Obiect care rămâne pe jos (static) atât timp DUPĂ ce a fost lăsat este o
# abandonare, chiar dacă persoana nu se îndepărtează 2 m (cazul wide-shot:
# aruncă și zăbovește în cadru). Dacă obiectul e ridicat înapoi, se mișcă și
# tracker-ul se resetează — anularea cerută de teză rămâne validă.
ABANDON_STATIC_S      = 1.2    # secunde de stat nemișcat în starea DROPPED → ABANDONED
TRASH_MISS_GRACE      = 12     # cadre de detecție lipsă tolerate înainte de a uita tracker-ul (clipire)
PICKUP_MOVE_PX        = 55.0   # deplasare mare a obiectului lângă persoană = ridicat înapoi (anulare)
# Raza în care un obiect NOU apărut lângă o persoană este asociat acelei
# persoane. Mai mare decât SEPARATION_DIST_M ca să prindă și aruncarea la
# distanță (obiectul aterizează la 2–3 m), nu doar lăsatul la picioare.
# Precizia rămâne protejată: obiectul trebuie să fie NOU (neexistent în
# baseline) ȘI să devină static pe jos înainte de a declanșa.
THROW_RANGE_M         = 3.0
# Dacă peste atât din box-ul obiectului e în interiorul siluetei persoanei,
# obiectul e ȚINUT în mână / în față, NU lăsat pe jos. Cât e ținut nu acumulăm
# progres de abandonare (altfel un obiect ținut nemișcat ar declanșa fals).
HELD_IN_PERSON_FRAC   = 0.50
CONFIRM_EVENT_S       = 3.0    # MODE A — wait this long after candidate event; cancel if person returns
                                # 3s = bun compromis: ignora reveniri rapide (<3s) dar prinde aruncari reale
EVENT_COOLDOWN_S      = 8.0    # suppress duplicate alerts immediately after one incident fires


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
    miss_frames:    int           = 0   # cadre consecutive fără detecție (clipire)
    dropped_static: int           = 0   # cadre nemișcat de când a intrat în DROPPED


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
    proximity_score:         float = 0.5    # 0-1: how close trash was to person zone


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
        self._baseline_trash_ids: set[int] = set()  # snapshot at monitoring entry
        self._person_zones: list[PersonZone] = []
        self._monitoring_start: int        = 0

        # Post-event clip capture
        self._capture_post: bool           = False
        self._post_frames_needed: int      = 0
        self._post_buffer: list[np.ndarray] = []
        self._pending_event: Optional[LitteringEvent] = None

        # MODE A — event confirmation window (debounce against brief person re-entries)
        self._confirm_frames: int           = int(CONFIRM_EVENT_S * self.fps)
        self._event_candidate: Optional[LitteringEvent] = None
        self._event_candidate_remaining: int = 0
        self._event_cooldown_remaining: int = 0

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
        if self._event_cooldown_remaining > 0:
            self._event_cooldown_remaining -= 1

        # Post-event clip finalisation
        if self._capture_post and self._pending_event is not None:
            self._post_buffer.append(frame.copy())
            self._post_frames_needed -= 1
            if self._post_frames_needed <= 0:
                self._pending_event.clip_frames += self._post_buffer
                self._capture_post   = False
                self._pending_event  = None
                self._post_buffer    = []

        # MODE B — distance-based (runs even when person is absent to handle timeouts)
        event = self._update_distance_trackers(
            frame, trash_detections, person_boxes, person_ids or []
        )
        if event is not None:
            if self._event_cooldown_remaining <= 0:
                self._start_post_capture(event)
                return event
            return None

        # MODE A — zone-based
        if self.state == DetectorState.CLEAR:
            # Baseline-ul scenei: obiectele vizibile cât timp NU există persoane
            # sunt preexistente și nu trebuie să declanșeze niciodată un incident.
            # NU se adaugă nimic la baseline cât timp persoana e prezentă —
            # un obiect apărut atunci (scos din mână, lăsat jos) rămâne candidat.
            self._known_trash_ids = set(current_trash_ids)
            if person_boxes:
                self._enter_person_present(person_boxes, current_trash_ids)

        elif self.state == DetectorState.PERSON_PRESENT:
            if person_boxes:
                self._update_person_zones(person_boxes)
            else:
                self._enter_monitoring()

        elif self.state == DetectorState.MONITORING:
            # If event candidate is pending, handle confirmation window first
            if self._event_candidate is not None:
                # Cancel only when a person returns to the evidence zone or to the
                # candidate object (likely picked it back up / re-entered the area).
                # A random passer-by elsewhere in the frame does not cancel.
                if self._person_returned_to_candidate(self._event_candidate, person_boxes):
                    self._event_candidate = None
                    self._event_candidate_remaining = 0
                    self._enter_person_present(person_boxes, current_trash_ids)
                    return None
                self._event_candidate_remaining -= 1
                if self._event_candidate_remaining <= 0:
                    # Confirmation window elapsed — fire the event for real.
                    confirmed = self._event_candidate
                    self._event_candidate = None
                    self._enter_clear()
                    self._start_post_capture(confirmed)
                    return confirmed
                return None

            if person_boxes:
                self._enter_person_present(person_boxes, current_trash_ids)
            elif self.frame_idx - self._monitoring_start > self.monitor_frames:
                self._enter_clear()
            else:
                new_ids = current_trash_ids - self._baseline_trash_ids
                if new_ids and self._event_cooldown_remaining <= 0:
                    candidate = self._check_zone_overlap(new_ids, trash_detections, frame)
                    if candidate is not None:
                        # Start confirmation window — do NOT fire yet; wait to see if person returns
                        self._event_candidate = candidate
                        self._event_candidate_remaining = self._confirm_frames

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
        self._event_candidate    = None
        self._event_candidate_remaining = 0
        self._event_cooldown_remaining = 0
        self._rel_trackers.clear()

    def finalize(self) -> Optional[LitteringEvent]:
        """Call when video stream ends to flush any pending events.

        Necesar mai ales pentru clipurile scurte (upload sau evaluare): dacă
        un candidat pe zonă este încă în fereastra de confirmare când se
        termină stream-ul, iar persoana nu s-a întors să ridice obiectul,
        evenimentul este real și trebuie emis — altfel un clip de câteva
        secunde cu o aruncare clară nu ar genera niciun incident.
        """
        if self._event_candidate is not None:
            confirmed = self._event_candidate
            self._event_candidate = None
            self._event_candidate_remaining = 0
            self._start_post_capture(confirmed)
            return confirmed

        if self._pending_event and self._capture_post:
            self._capture_post = False
            return self._pending_event

        for tid, tracker in list(self._rel_trackers.items()):
            if tracker.state in [TrashRelState.SEPARATING, TrashRelState.DROPPED]:
                # Force fire event since video ended
                det = {"box": (int(tracker.last_trash_cx), int(tracker.last_trash_cy), int(tracker.last_trash_cx+10), int(tracker.last_trash_cy+10)), "material_name": "unknown", "det_score": 0.5}
                # Find best matching det from last frame if possible, else mock
                event = self._build_distance_event(
                    self._frame_buffer[-1] if self._frame_buffer else np.zeros((10,10,3), dtype=np.uint8),
                    det, tracker, tracker.max_distance_m, tracker.owner_box
                )
                del self._rel_trackers[tid]
                return event
        return None

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
        now = self.frame_idx / self.fps
        scale = self._estimate_scale(person_boxes)

        pid_map: dict[int, tuple[int, int, int, int]] = {
            (person_ids[i] if i < len(person_ids) else -(i + 1)): pb
            for i, pb in enumerate(person_boxes)
        }

        current_trash_ids = {d["track_id"] for d in trash_dets}
        trash_by_id = {d["track_id"]: d for d in trash_dets}

        # Toleranță la clipirea detecției: nu uita tracker-ul la primul cadru
        # lipsă, ci abia după câteva (la fel ca pe calea de zonă). Altfel un
        # obiect lăsat pe jos care „pâlpâie" pierde tot progresul de abandonare.
        for tid, tracker in list(self._rel_trackers.items()):
            if tid not in current_trash_ids:
                tracker.miss_frames += 1
                if tracker.miss_frames > TRASH_MISS_GRACE:
                    del self._rel_trackers[tid]
            else:
                tracker.miss_frames = 0

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
                # Obiect preexistent (vizibil în scenă fără persoană) — nu se
                # atașează niciodată unei persoane; altfel ar produce fals
                # pozitiv „abandon" când persoana doar trece pe lângă el.
                if tid in self._known_trash_ids:
                    continue
                if dist_m < THROW_RANGE_M:
                    self._rel_trackers[tid] = TrashRelTracker(
                        trash_id=tid,
                        person_id=nearest_pid,
                        owner_box=nearest_pb,
                        last_trash_cx=tcx,
                        last_trash_cy=tcy,
                    )
                continue

            tracker.owner_box = nearest_pb
            move_px = math.hypot(
                tcx - tracker.last_trash_cx,
                tcy - tracker.last_trash_cy,
            )
            is_static = move_px < JITTER_THRESHOLD_PX
            # Obiectul e încă ținut în mână / în fața corpului? Atunci NU e lăsat
            # pe jos și nu trebuie să acumuleze progres de abandonare.
            is_held = self._trash_in_person_frac(tb, nearest_pb) > HELD_IN_PERSON_FRAC
            tracker.last_trash_cx = tcx
            tracker.last_trash_cy = tcy
            tracker.max_distance_m = max(tracker.max_distance_m, dist_m)

            if tracker.state == TrashRelState.NEARBY:
                if is_static and not is_held:
                    tracker.static_frames += 1
                    # Obiect NOU devenit static pe jos, în raza unei persoane =
                    # lăsat sau aruncat. Nu mai cerem să fie sub 1.5 m, ca să
                    # prindem și aruncarea (obiectul aterizează mai departe).
                    if (
                        tracker.static_frames >= STATIC_FRAMES_NEEDED
                        and dist_m < THROW_RANGE_M
                    ):
                        tracker.state         = TrashRelState.DROPPED
                        tracker.static_frames = 0
                else:
                    tracker.static_frames = 0

            elif tracker.state == TrashRelState.DROPPED:
                # Calea rapidă: persoana chiar se îndepărtează → separare.
                if dist_m >= SEPARATION_DIST_M:
                    tracker.state = TrashRelState.SEPARATING
                    tracker.dropped_static = 0
                elif move_px > PICKUP_MOVE_PX:
                    # Obiectul s-a deplasat MULT lângă persoană — probabil ridicat
                    # înapoi; resetează progresul (anulare conform tezei). Doar o
                    # mișcare mare contează, nu jitterul de re-detecție după o
                    # clipire, care altfel ar șterge tot progresul de abandonare.
                    tracker.dropped_static = 0
                    tracker.state = TrashRelState.NEARBY
                    tracker.static_frames = 0
                else:
                    # Obiectul rămâne (aproximativ) pe loc. Dacă persoana l-ar fi
                    # ridicat, s-ar fi mișcat mult (ramura de mai sus). Stat
                    # nemișcat destul = abandonare, chiar dacă persoana zăbovește
                    # în cadru (cazul wide-shot: aruncă și rămâne aproape).
                    if is_static and not is_held:
                        tracker.dropped_static += 1
                    if tracker.dropped_static >= int(ABANDON_STATIC_S * self.fps):
                        tracker.state = TrashRelState.ABANDONED
                        event = self._build_distance_event(frame, det, tracker, dist_m, nearest_pb)
                        del self._rel_trackers[tid]
                        return event

            elif tracker.state == TrashRelState.SEPARATING:
                if dist_m >= ABANDON_DIST_M:
                    tracker.state = TrashRelState.ABANDONED
                    event = self._build_distance_event(frame, det, tracker, dist_m, nearest_pb)
                    del self._rel_trackers[tid]
                    return event

        # Person-lost timeout for SEPARATING trackers
        for tid, tracker in list(self._rel_trackers.items()):
            if tracker.state in [TrashRelState.SEPARATING, TrashRelState.DROPPED]:
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
    def _trash_in_person_frac(
        tb: tuple[int, int, int, int], pb: tuple[int, int, int, int]
    ) -> float:
        """Fracțiunea din aria box-ului de gunoi aflată în interiorul persoanei."""
        ix1, iy1 = max(tb[0], pb[0]), max(tb[1], pb[1])
        ix2, iy2 = min(tb[2], pb[2]), min(tb[3], pb[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area = max((tb[2] - tb[0]) * (tb[3] - tb[1]), 1)
        return inter / area

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
        # NOTE: baseline-ul (_known_trash_ids) NU se actualizează aici —
        # obiectele vizibile în timpul prezenței persoanei (inclusiv cele din
        # mână) trebuie să rămână candidate pentru fereastra de monitorizare.
        self.state = DetectorState.PERSON_PRESENT
        self._person_zones = [
            PersonZone(x1, y1, x2, y2, self.frame_idx)
            for (x1, y1, x2, y2) in person_boxes
        ]

    def _update_person_zones(self, person_boxes: list) -> None:
        self._person_zones = [
            PersonZone(x1, y1, x2, y2, self.frame_idx)
            for (x1, y1, x2, y2) in person_boxes
        ]

    def _enter_monitoring(self) -> None:
        self.state             = DetectorState.MONITORING
        self._monitoring_start = self.frame_idx
        # Snapshot trash at moment person leaves — only NEW objects after this count
        self._baseline_trash_ids = set(self._known_trash_ids)

    def _enter_clear(self) -> None:
        self.state = DetectorState.CLEAR
        self._person_zones.clear()
        self._known_trash_ids.clear()
        self._baseline_trash_ids.clear()  # reset so next session starts fresh

    def _check_zone_overlap(
        self,
        new_trash_ids: set[int],
        trash_detections: list[dict],
        frame: np.ndarray,
    ) -> Optional[LitteringEvent]:
        # Fire only when a NEW trash object appears INSIDE the zone where the
        # person was last seen (expanded). A new track elsewhere in the frame
        # (vehicle, background object, tracker re-id) is NOT an incident.
        if not self._person_zones:
            return None

        zone_expand = max(self.zone_expand, 0.35)
        for det in trash_detections:
            if det["track_id"] not in new_trash_ids:
                continue

            tx = (det["box"][0] + det["box"][2]) / 2
            ty = (det["box"][1] + det["box"][3]) / 2

            zone = next(
                (z for z in self._person_zones
                 if _point_in_box(tx, ty, z.expanded(zone_expand))),
                None,
            )
            if zone is None:
                continue  # new object far from the person zone — ignore

            # Proximity score: 1.0 = trash centre at zone centre, 0.0 = far away
            zw = max(zone.x2 - zone.x1, 1)
            zh = max(zone.y2 - zone.y1, 1)
            px = max(0.0, 1.0 - abs(tx - (zone.x1 + zone.x2) / 2) / (zw * 1.5))
            py = max(0.0, 1.0 - abs(ty - (zone.y1 + zone.y2) / 2) / (zh * 1.5))
            proximity_score = round(px * py, 3)

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
                        proximity_score  = proximity_score,
                    )
        return None

    def _person_returned_to_candidate(
        self,
        candidate: LitteringEvent,
        person_boxes: list[tuple[int, int, int, int]],
    ) -> bool:
        """
        Cancel MODE A confirmation only when a person comes back to the relevant
        evidence zone. A random person elsewhere in the frame should not cancel
        a valid littering candidate.
        """
        if not person_boxes:
            return False

        zone = PersonZone(*candidate.person_box, self.frame_idx).expanded(0.35)
        tx1, ty1, tx2, ty2 = candidate.trash_box
        tcx = (tx1 + tx2) / 2.0
        tcy = (ty1 + ty2) / 2.0

        for px1, py1, px2, py2 in person_boxes:
            # Person overlaps the last known person zone.
            if _box_iou((px1, py1, px2, py2), zone) >= 0.05:
                return True

            # Person is physically near/covering the candidate trash object.
            pw = max(px2 - px1, 1)
            ph = max(py2 - py1, 1)
            expanded_person = (
                int(px1 - 0.20 * pw),
                int(py1 - 0.20 * ph),
                int(px2 + 0.20 * pw),
                int(py2 + 0.20 * ph),
            )
            if _point_in_box(tcx, tcy, expanded_person):
                return True

        return False

    def _start_post_capture(self, event: LitteringEvent) -> None:
        self._capture_post       = True
        self._post_frames_needed = int(3.0 * self.fps)
        self._post_buffer        = []
        event.clip_frames        = list(self._frame_buffer)
        self._pending_event      = event
        self._event_cooldown_remaining = int(EVENT_COOLDOWN_S * self.fps)


# ─────────────────────────────────────────────────────────────────────────────
# Utility — thumbnail generation
# ─────────────────────────────────────────────────────────────────────────────

def _make_thumbnail(
    frame: np.ndarray,
    trash_box: tuple[int, int, int, int],
    zone: PersonZone,
    size: tuple[int, int] = (320, 240),
) -> np.ndarray:
    thumb = frame.copy()

    # No face blur — visual evidence intentionally identifies the perpetrator
    # Zona persoanei nu se mai desenează pe dovadă: la distanțe mai mari de
    # 2-3 m devine imprecisă și induce în eroare la verificare.

    tx1, ty1, tx2, ty2 = trash_box
    cv2.rectangle(thumb, (tx1, ty1), (tx2, ty2), (0, 0, 255), 2)

    return cv2.resize(thumb, size, interpolation=cv2.INTER_AREA)


def _point_in_box(x: float, y: float, box: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def _box_iou(
    box: tuple[int, int, int, int],
    zone: tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = box
    bx1, by1, bx2, by2 = zone

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1)


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
