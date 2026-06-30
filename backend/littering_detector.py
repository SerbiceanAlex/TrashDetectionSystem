"""
LitteringDetector — mașina de stări care decide ABANDONAREA de gunoi în timp.

Rulează două moduri complementare pe fiecare cadru:
  • Mod ZONĂ: după ce o persoană pleacă, urmărește dacă apare un obiect NOU în
    zona unde a stat → incident.
  • Mod DISTANȚĂ: leagă un obiect de persoana de lângă el; dacă obiectul rămâne
    pe loc iar persoana se îndepărtează (sau dispare) → incident.

Distanțele se estimează din înălțimea casetei persoanei (~1,70 m), deci merge
fără calibrare de cameră. Un obiect ținut în mână sau ridicat înapoi NU
declanșează. Folosire: update() pe fiecare cadru, reset() între sesiuni.
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
# Constante (praguri reglate empiric)
# ─────────────────────────────────────────────────────────────────────────────

PERSON_HEIGHT_M       = 1.70   # înălțimea medie presupusă a persoanei (metri), pentru estimarea scării
SEPARATION_DIST_M     = 1.50   # distanța persoană-obiect de la care se intră în SEPARATING
ABANDON_DIST_M        = 2.00   # distanța de la care se declanșează ABANDONED (omul aruncă și face 1–2 pași)
JITTER_THRESHOLD_PX   = 15.0   # mișcare maximă a centrului obiectului (px) ca să-l considerăm static
STATIC_FRAMES_NEEDED  = 8      # cadre consecutive în care obiectul stă nemișcat → DROPPED
LOST_TIMEOUT_S        = 5.0    # secunde cât poate lipsi persoana înainte de ABANDONED
# Obiect care rămâne pe jos (static) atât timp DUPĂ ce a fost lăsat este o
# abandonare, chiar dacă persoana nu se îndepărtează 2 m (cazul wide-shot:
# aruncă și zăbovește în cadru). Dacă obiectul e ridicat înapoi, se mișcă și
# urmărirea se resetează — anularea cerută de teză rămâne validă.
ABANDON_STATIC_S      = 0.7    # secunde de stat nemișcat în starea DROPPED → ABANDONED.
                                # Scăzut de la 1.2s: la filmările de aproape tracker-ul
                                # reatribuie id-uri obiectului (instabilitate), iar progresul
                                # de abandonare se pierde înainte de prag. 0.7s prinde aruncarea
                                # în fereastra stabilă, fără a slăbi protecția (obiect ținut nu
                                # ajunge în DROPPED).
TRASH_MISS_GRACE      = 12     # cadre de detecție lipsă tolerate înainte de a uita obiectul urmărit (clipire)
PICKUP_MOVE_PX        = 55.0   # deplasare mare a obiectului lângă persoană = ridicat înapoi (anulare)
# Raza în care un obiect NOU apărut lângă o persoană este asociat acelei
# persoane. Mai mare decât SEPARATION_DIST_M ca să prindă și aruncarea la
# distanță (obiectul aterizează la 2–3 m), nu doar lăsatul la picioare.
# Precizia rămâne protejată: obiectul trebuie să fie NOU (neexistent în
# starea inițială) ȘI să devină static pe jos înainte de a declanșa.
THROW_RANGE_M         = 3.0
# Dacă peste atât din box-ul obiectului e în interiorul siluetei persoanei,
# obiectul e ȚINUT în mână / în față, NU lăsat pe jos. Cât e ținut nu acumulăm
# progres de abandonare (altfel un obiect ținut nemișcat ar declanșa fals).
HELD_IN_PERSON_FRAC   = 0.50
HELD_NOT_FLOOR_FRAC   = 0.66   # la filmări de aproape: obiect deasupra acestei fracțiuni
                               # din înălțimea cadrului = ridicat/ținut (NU pe podea) —
                               # robust la persoane filmate doar bust (fără picioare)
HELD_DISTANCE_M       = 1.50   # „ținut/lângă persoană" = mai aproape decât distanța de
                                # separare. Sub atât, obiectul e considerat în mână (braț
                                # întins) și NU acumulează abandonare — abia când persoana
                                # se îndepărtează clar (peste 1.5m) începe logica de abandon.
CONFIRM_EVENT_S       = 3.0    # mod ZONĂ — câte secunde așteptăm după un candidat; se anulează dacă
                                # persoana se întoarce. 3s = ignoră revenirile rapide, dar prinde aruncările reale
EVENT_COOLDOWN_S      = 8.0    # blochează alerte duplicate imediat după ce s-a declanșat un incident


# ─────────────────────────────────────────────────────────────────────────────
# Stare mod ZONĂ
# ─────────────────────────────────────────────────────────────────────────────

class DetectorState(Enum):
    """Starea globală a scenei în modul zonă."""
    CLEAR          = auto()   # fără persoane; obiectele vizibile definesc starea inițială
    PERSON_PRESENT = auto()   # persoană în cadru; îi reținem zona
    MONITORING     = auto()   # persoana a plecat; urmărim obiecte noi în zona ei


# ─────────────────────────────────────────────────────────────────────────────
# Stare per-obiect, mod DISTANȚĂ
# ─────────────────────────────────────────────────────────────────────────────

class TrashRelState(Enum):
    """Starea relației dintre UN obiect de gunoi și persoana de lângă el."""
    NEARBY     = auto()   # obiect lângă persoană, posibil în mână
    DROPPED    = auto()   # obiect nemișcat pe jos, persoana încă aproape
    SEPARATING = auto()   # persoana se îndepărtează de obiectul static
    ABANDONED  = auto()   # persoana a plecat clar → declanșează incident


@dataclass
class TrashRelTracker:
    """Urmărește evoluția relației dintre un obiect (track_id) și o persoană."""
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
# Tipuri de date comune
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PersonZone:
    """Ultima poziție cunoscută a unei persoane (folosită după ce pleacă)."""
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
    """Incidentul emis când se detectează o abandonare (mod zonă sau distanță)."""
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
    # Câmpuri de dovadă
    incident_uid:            str   = field(default_factory=lambda: str(uuid.uuid4()))
    owner_person_id:         Optional[int]   = None
    distance_at_abandonment: Optional[float] = None
    detection_method:        str  = "zone"   # valori interne: "zone" | "distance"


# ─────────────────────────────────────────────────────────────────────────────
# Detectorul principal
# ─────────────────────────────────────────────────────────────────────────────

class LitteringDetector:
    """
    Detector hibrid de abandonare, care combină mașina de stări pe zonă cu cea
    pe distanță.

    Argumente:
        fps:               fps-ul așteptat al fluxului video.
        monitor_seconds:   mod zonă — cât timp urmărim obiecte noi după plecarea persoanei.
        pre_event_seconds: câte secunde de cadre se păstrează în memoria pre-incident.
        person_conf:       pragul de încredere pentru detectorul de persoane.
        zone_expand:       cu cât se extinde caseta persoanei la verificarea de zonă.
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

        # Stare mod ZONĂ
        self.state: DetectorState          = DetectorState.CLEAR
        self.frame_idx: int                = 0
        self._frame_buffer: deque[np.ndarray] = deque(maxlen=self.pre_buffer_frames)
        self._known_trash_ids: set[int]    = set()
        self._baseline_trash_ids: set[int] = set()  # captură la intrarea în monitorizare
        self._person_zones: list[PersonZone] = []
        self._monitoring_start: int        = 0

        # Captura clipului post-incident
        self._capture_post: bool           = False
        self._post_frames_needed: int      = 0
        self._post_buffer: list[np.ndarray] = []
        self._pending_event: Optional[LitteringEvent] = None

        # Fereastra de confirmare (ignoră reintrările scurte ale persoanei)
        self._confirm_frames: int           = int(CONFIRM_EVENT_S * self.fps)
        self._event_candidate: Optional[LitteringEvent] = None
        self._event_candidate_remaining: int = 0
        self._event_cooldown_remaining: int = 0

        # Stare mod DISTANȚĂ
        self._rel_trackers: dict[int, TrashRelTracker] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # API public
    # ─────────────────────────────────────────────────────────────────────────

    def update(
        self,
        frame: np.ndarray,
        trash_detections: list[dict],
        person_boxes: list[tuple[int, int, int, int]],
        person_ids: Optional[list[int]] = None,
    ) -> Optional[LitteringEvent]:
        """
        Procesează un cadru și întoarce un incident dacă se declanșează acum,
        altfel None.

        Argumente:
            frame:            cadrul brut (BGR).
            trash_detections: dict-uri cu 'track_id', 'box', 'material_name', 'det_score'.
            person_boxes:     casetele persoanelor (x1,y1,x2,y2) de la detect_persons().
            person_ids:       opțional, ID-urile ByteTrack ale persoanelor (aceeași ordine).
        """
        self.frame_idx += 1
        self._frame_buffer.append(frame.copy())

        current_trash_ids = {d["track_id"] for d in trash_detections}
        if self._event_cooldown_remaining > 0:
            self._event_cooldown_remaining -= 1

        # Finalizarea clipului post-incident (umple buffer-ul de după eveniment)
        if self._capture_post and self._pending_event is not None:
            self._post_buffer.append(frame.copy())
            self._post_frames_needed -= 1
            if self._post_frames_needed <= 0:
                self._pending_event.clip_frames += self._post_buffer
                self._capture_post   = False
                self._pending_event  = None
                self._post_buffer    = []

        # Mod DISTANȚĂ (rulează și fără persoană, ca să trateze timeout-urile)
        event = self._update_distance_trackers(
            frame, trash_detections, person_boxes, person_ids or []
        )
        if event is not None:
            if self._event_cooldown_remaining <= 0:
                self._start_post_capture(event)
                return event
            return None

        # Mod ZONĂ
        if self.state == DetectorState.CLEAR:
            # Starea inițială a scenei: obiectele vizibile cât timp NU există persoane
            # sunt preexistente și nu trebuie să declanșeze niciodată un incident.
            # NU se adaugă nimic la starea inițială cât timp persoana e prezentă —
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
            # Dacă există un candidat, tratăm întâi fereastra de confirmare
            if self._event_candidate is not None:
                # Anulăm doar dacă o persoană se întoarce în zona dovezii sau la
                # obiectul candidat (probabil l-a ridicat / a reintrat în zonă).
                # Un trecător oarecare în altă parte a cadrului NU anulează.
                if self._person_returned_to_candidate(self._event_candidate, person_boxes):
                    self._event_candidate = None
                    self._event_candidate_remaining = 0
                    self._enter_person_present(person_boxes, current_trash_ids)
                    return None
                self._event_candidate_remaining -= 1
                if self._event_candidate_remaining <= 0:
                    # Fereastra de confirmare a expirat — declanșăm incidentul.
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
                        # Pornim fereastra de confirmare — NU declanșăm încă; așteptăm o eventuală revenire
                        self._event_candidate = candidate
                        self._event_candidate_remaining = self._confirm_frames

        return None

    def reset(self) -> None:
        """Resetează toată starea — se apelează între sesiuni video independente."""
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
        """Se apelează la finalul fluxului video, ca să emită evenimentele în așteptare.

        Necesar mai ales pentru clipurile scurte (upload sau evaluare): dacă
        un candidat pe zonă este încă în fereastra de confirmare când se
        termină stream-ul, iar persoana nu s-a întors să ridice obiectul,
        evenimentul este real și trebuie emis — altfel un clip de câteva
        secunde cu o aruncare clară nu ar genera niciun incident.
        """
        if self._event_candidate is not None:
            # Nu confirmam fortat un candidat de zona doar pentru ca videoul s-a
            # terminat. Daca fereastra de confirmare nu a expirat natural in
            # update(), riscul de fals pozitiv este prea mare (ex.: obiect tinut
            # in mana sau persoana care reintra imediat dupa finalul clipului).
            self._event_candidate = None
            self._event_candidate_remaining = 0
            return None

        # Fereastra post-incident e încă deschisă la finalul videoului: evenimentul
        # a fost DEJA emis de update() (și colectat). Aici doar închidem captura —
        # NU îl re-emitem, altfel s-ar crea un al doilea incident duplicat pentru
        # aceeași aruncare (cazul tipic: incidentul are loc în ultimele ~3s).
        if self._pending_event is not None and self._capture_post:
            self._capture_post = False
            self._pending_event = None

        for tid, tracker in list(self._rel_trackers.items()):
            if tracker.state in [TrashRelState.SEPARATING, TrashRelState.DROPPED]:
                # Videoul s-a terminat cu o abandonare în curs → o emitem forțat.
                det = {"box": (int(tracker.last_trash_cx), int(tracker.last_trash_cy), int(tracker.last_trash_cx+10), int(tracker.last_trash_cy+10)), "material_name": "unknown", "det_score": 0.5}
                # Folosim ultimul cadru disponibil pentru miniatură.
                event = self._build_distance_event(
                    self._frame_buffer[-1] if self._frame_buffer else np.zeros((10,10,3), dtype=np.uint8),
                    det, tracker, tracker.max_distance_m, tracker.owner_box
                )
                del self._rel_trackers[tid]
                return event
        return None

    @property
    def current_state(self) -> str:
        """Numele stării curente din modul zonă (pentru afișare în UI)."""
        return self.state.name

    @property
    def monitoring_progress(self) -> float:
        """Fracțiunea (0–1) din fereastra de monitorizare scursă (doar în MONITORING)."""
        if self.state != DetectorState.MONITORING:
            return 0.0
        elapsed = self.frame_idx - self._monitoring_start
        return min(elapsed / max(self.monitor_frames, 1), 1.0)

    # ─────────────────────────────────────────────────────────────────────────
    # Funcții ajutătoare pentru modul DISTANȚĂ
    # ─────────────────────────────────────────────────────────────────────────

    def _update_distance_trackers(
        self,
        frame: np.ndarray,
        trash_dets: list[dict],
        person_boxes: list[tuple[int, int, int, int]],
        person_ids: list[int],
    ) -> Optional[LitteringEvent]:
        """Avansează starea fiecărui obiect față de persoana lui; întoarce un incident dacă se abandonează."""
        now = self.frame_idx / self.fps
        scale = self._estimate_scale(person_boxes)

        pid_map: dict[int, tuple[int, int, int, int]] = {
            (person_ids[i] if i < len(person_ids) else -(i + 1)): pb
            for i, pb in enumerate(person_boxes)
        }

        current_trash_ids = {d["track_id"] for d in trash_dets}
        trash_by_id = {d["track_id"]: d for d in trash_dets}

        # Toleranță la clipirea detecției: nu uita obiectul urmărit la primul cadru
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
            # pe jos și nu trebuie să acumuleze progres de abandonare. „Ținut" =
            # se suprapune cu corpul SAU e aproape ȘI la înălțimea corpului/mâinii.
            # Un obiect aflat JOS (în zona tălpilor sau sub ele) NU e ținut, chiar
            # dacă persoana e aproape — altfel, la filmările de aproape (persoana
            # umple cadrul), distanța estimată e mereu sub prag și nicio aruncare
            # n-ar declanșa vreodată.
            obj_cy = (tb[1] + tb[3]) / 2.0
            at_body_height = obj_cy < nearest_pb[3] - 0.20 * (nearest_pb[3] - nearest_pb[1])
            # Robust și la filmări „bust" (persoana umple cadrul, fără picioare):
            # un obiect care NU e în partea de jos a CADRULUI (zona podelei) e
            # ridicat → ținut, chiar dacă caseta persoanei se termină la brâu și
            # at_body_height eșuează. Se aplică doar când persoana e aproape.
            not_near_floor = obj_cy < HELD_NOT_FLOOR_FRAC * frame.shape[0]
            is_held = (
                self._trash_in_person_frac(tb, nearest_pb) > HELD_IN_PERSON_FRAC
                or (dist_m <= HELD_DISTANCE_M and at_body_height)
                or (dist_m <= HELD_DISTANCE_M and not_near_floor)
            )
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
                if is_held:
                    # Obiectul este inca langa persoana sau revine langa ea.
                    # Anulam progresul ca sa evitam incidente false cand
                    # obiectul este tinut in mana cu bratul intins.
                    tracker.dropped_static = 0
                    tracker.static_frames = 0
                    tracker.person_lost_at = None
                    tracker.state = TrashRelState.NEARBY
                # Calea rapidă: persoana chiar se îndepărtează → separare.
                elif dist_m >= SEPARATION_DIST_M:
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
                if is_held:
                    # Persoana s-a apropiat din nou de obiect; anulam
                    # separarea in loc sa declansam abandonare.
                    tracker.dropped_static = 0
                    tracker.static_frames = 0
                    tracker.person_lost_at = None
                    tracker.state = TrashRelState.NEARBY
                elif dist_m >= ABANDON_DIST_M:
                    tracker.state = TrashRelState.ABANDONED
                    event = self._build_distance_event(frame, det, tracker, dist_m, nearest_pb)
                    del self._rel_trackers[tid]
                    return event

        # Timeout pentru persoana dispărută (obiecte urmărite în SEPARATING/DROPPED)
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
        """Construiește incidentul (cu miniatură) pentru o abandonare detectată pe distanță."""
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
        """Estimează pixeli/metru din înălțimea celei mai înalte casete de persoană."""
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
        """Găsește cea mai apropiată persoană de un punct; întoarce (id, casetă, distanță_m)."""
        best_pid, best_pb, best_d = None, None, float("inf")
        for pid, pb in pid_map.items():
            # Distanta se masoara pana la boxul persoanei, nu pana la centrul
            # corpului. Un obiect tinut cu bratul intins poate fi departe de
            # centrul persoanei, dar tot este aproape de silueta persoanei.
            dx = max(pb[0] - cx, 0.0, cx - pb[2])
            dy = max(pb[1] - cy, 0.0, cy - pb[3])
            d_m = math.hypot(dx, dy) / max(scale, 1.0)
            if d_m < best_d:
                best_d, best_pid, best_pb = d_m, pid, pb
        return best_pid, best_pb, best_d

    # ─────────────────────────────────────────────────────────────────────────
    # Funcții ajutătoare pentru modul ZONĂ
    # ─────────────────────────────────────────────────────────────────────────

    def _enter_person_present(
        self, person_boxes: list, current_trash_ids: set
    ) -> None:
        """Intră în starea PERSON_PRESENT și reține zonele persoanelor curente."""
        # NOTĂ: starea inițială (_known_trash_ids) NU se actualizează aici —
        # obiectele vizibile în timpul prezenței persoanei (inclusiv cele din
        # mână) trebuie să rămână candidate pentru fereastra de monitorizare.
        self.state = DetectorState.PERSON_PRESENT
        self._person_zones = [
            PersonZone(x1, y1, x2, y2, self.frame_idx)
            for (x1, y1, x2, y2) in person_boxes
        ]

    def _update_person_zones(self, person_boxes: list) -> None:
        """Actualizează zonele cu pozițiile curente ale persoanelor."""
        self._person_zones = [
            PersonZone(x1, y1, x2, y2, self.frame_idx)
            for (x1, y1, x2, y2) in person_boxes
        ]

    def _enter_monitoring(self) -> None:
        """Persoana a plecat → intră în MONITORING și fixează starea inițială de obiecte."""
        self.state             = DetectorState.MONITORING
        self._monitoring_start = self.frame_idx
        # Captură a obiectelor la plecarea persoanei — doar cele NOI de acum contează.
        self._baseline_trash_ids = set(self._known_trash_ids)

    def _enter_clear(self) -> None:
        """Revine la starea CLEAR și golește zonele și starea inițială."""
        self.state = DetectorState.CLEAR
        self._person_zones.clear()
        self._known_trash_ids.clear()
        self._baseline_trash_ids.clear()  # resetare, ca sesiunea următoare să pornească curat

    def _check_zone_overlap(
        self,
        new_trash_ids: set[int],
        trash_detections: list[dict],
        frame: np.ndarray,
    ) -> Optional[LitteringEvent]:
        """Declanșează doar dacă un obiect NOU apare ÎN zona unde a stat persoana."""
        # Un obiect urmărit nou în altă parte a cadrului (vehicul, fundal, re-id)
        # NU e incident.
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
                continue  # obiect nou, departe de zona persoanei — ignorat

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

    def _person_returned_to_candidate(
        self,
        candidate: LitteringEvent,
        person_boxes: list[tuple[int, int, int, int]],
    ) -> bool:
        """
        Anulează confirmarea (mod zonă) doar când o persoană revine în zona
        relevantă a dovezii. O persoană oarecare în altă parte a cadrului nu
        trebuie să anuleze un candidat valid de abandonare.
        """
        if not person_boxes:
            return False

        zone = PersonZone(*candidate.person_box, self.frame_idx).expanded(0.35)
        tx1, ty1, tx2, ty2 = candidate.trash_box
        tcx = (tx1 + tx2) / 2.0
        tcy = (ty1 + ty2) / 2.0

        for px1, py1, px2, py2 in person_boxes:
            # Persoana se suprapune cu ultima zonă cunoscută a persoanei.
            if _box_iou((px1, py1, px2, py2), zone) >= 0.05:
                return True

            # Persoana e fizic lângă / acoperă obiectul candidat.
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
        """Pregătește capturarea celor ~3s de după incident și pornește cooldown-ul."""
        self._capture_post       = True
        self._post_frames_needed = int(3.0 * self.fps)
        self._post_buffer        = []
        event.clip_frames        = list(self._frame_buffer)
        self._pending_event      = event
        self._event_cooldown_remaining = int(EVENT_COOLDOWN_S * self.fps)


# ─────────────────────────────────────────────────────────────────────────────
# Utilitar — generarea miniaturii
# ─────────────────────────────────────────────────────────────────────────────

def _make_thumbnail(
    frame: np.ndarray,
    trash_box: tuple[int, int, int, int],
    zone: PersonZone,
    size: tuple[int, int] = (320, 240),
) -> np.ndarray:
    """Desenează caseta obiectului pe cadru și întoarce o miniatură (dovada)."""
    thumb = frame.copy()

    # Fără blur pe față — dovada vizuală identifică intenționat făptașul.
    # Zona persoanei nu se mai desenează: la peste 2-3 m devine imprecisă și
    # induce în eroare la verificare.

    tx1, ty1, tx2, ty2 = trash_box
    cv2.rectangle(thumb, (tx1, ty1), (tx2, ty2), (0, 0, 255), 2)

    return cv2.resize(thumb, size, interpolation=cv2.INTER_AREA)


def _point_in_box(x: float, y: float, box: tuple[int, int, int, int]) -> bool:
    """Întoarce True dacă punctul (x, y) este în interiorul casetei."""
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def _box_iou(
    box: tuple[int, int, int, int],
    zone: tuple[int, int, int, int],
) -> float:
    """Intersection-over-Union între două casete (0 = disjuncte, 1 = identice)."""
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
