"""Привязка игроков к командам: ручные клики + автоподсказка по форме.

Ручная разметка — основной режим: на видео нет стартовых линий, поэтому
пользователь кликами ставит A/B. Автоподсказка нужна только для игроков,
которые появились позже или потеряли ID: цвет торса сравнивается с
эталонами уже размеченных треков.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from ultitrack.cvat.tracks import (
    LABEL_PLAYER,
    TEAM_A,
    TEAM_B,
    TEAM_UNKNOWN,
    Track,
    TrackSet,
)
from ultitrack.team_classifier import _kmeans_2, extract_torso_feature


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_team(value: object) -> str:
    """Приводит клики/старые JSON к одному из A, B, unknown."""
    if value is None:
        return TEAM_UNKNOWN
    text = str(value).strip()
    if text in ("", "?", "u", "U", "unknown", "None", "null", "0"):
        return TEAM_UNKNOWN
    upper = text.upper()
    if upper in (TEAM_A, TEAM_B):
        return upper
    raise ValueError(f"Команда должна быть A, B или unknown, получено: {value!r}")


@dataclass
class TeamBindings:
    """Текущие привязки трек -> команда.

    ``manual`` — треки, которые пользователь поставил руками. Только они
    считаются эталонами для автоподсказки; остальные можно пересчитать.
    """

    video: str = ""
    assignments: dict[int, str] = field(default_factory=dict)
    confidence: dict[int, float] = field(default_factory=dict)
    manual: set[int] = field(default_factory=set)
    # Треки, которые не надо тащить в CVAT: судьи, запасные, ложные боксы.
    excluded: set[int] = field(default_factory=set)
    source: str = "none"
    updated: str = field(default_factory=_now_iso)

    def set(self, track_id: int, team: str, *, manual: bool = True, confidence: float = 1.0) -> None:
        track_id = int(track_id)
        team = normalize_team(team)
        self.assignments[track_id] = team
        self.confidence[track_id] = float(confidence) if team != TEAM_UNKNOWN else 0.0
        if manual:
            self.manual.add(track_id)
        self.updated = _now_iso()

    def set_excluded(self, track_id: int, excluded: bool = True) -> None:
        track_id = int(track_id)
        if excluded:
            self.excluded.add(track_id)
        else:
            self.excluded.discard(track_id)
        self.updated = _now_iso()

    def swap(self) -> None:
        for track_id, team in list(self.assignments.items()):
            if team == TEAM_A:
                self.assignments[track_id] = TEAM_B
            elif team == TEAM_B:
                self.assignments[track_id] = TEAM_A
        self.updated = _now_iso()

    def counts(self) -> dict[str, int]:
        result = {TEAM_A: 0, TEAM_B: 0, TEAM_UNKNOWN: 0}
        for team in self.assignments.values():
            result[team if team in result else TEAM_UNKNOWN] += 1
        return result

    def to_dict(self) -> dict:
        return {
            "video": self.video,
            "assignments": {str(track_id): team for track_id, team in sorted(self.assignments.items())},
            "confidence": {
                str(track_id): round(float(value), 3)
                for track_id, value in sorted(self.confidence.items())
            },
            "manual": sorted(int(track_id) for track_id in self.manual),
            "excluded": sorted(int(track_id) for track_id in self.excluded),
            "source": self.source,
            "updated": self.updated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TeamBindings":
        if not isinstance(data, dict):
            raise ValueError("teams.json должен быть объектом")
        raw_assignments = data.get("assignments")
        if raw_assignments is None:
            # Совместимость с простым форматом {"1": "A", "2": "B"}.
            raw_assignments = {
                key: value
                for key, value in data.items()
                if str(key).isdigit()
            }
        assignments: dict[int, str] = {}
        for key, value in (raw_assignments or {}).items():
            assignments[int(key)] = normalize_team(value)
        confidence = {
            int(key): float(value)
            for key, value in (data.get("confidence") or {}).items()
        }
        manual = {int(value) for value in (data.get("manual") or [])}
        excluded = {int(value) for value in (data.get("excluded") or [])}
        return cls(
            video=str(data.get("video") or ""),
            assignments=assignments,
            confidence=confidence,
            manual=manual,
            excluded=excluded,
            source=str(data.get("source") or "none"),
            updated=str(data.get("updated") or _now_iso()),
        )

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
        return path

    @classmethod
    def load(cls, path: Path, *, video: str = "") -> "TeamBindings":
        path = Path(path)
        if not path.exists():
            return cls(video=video)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            bindings = cls.from_dict(raw)
        except (OSError, ValueError, TypeError) as exc:
            print(f"[CVAT] Не удалось прочитать {path}: {exc}. Начинаю с пустых команд.")
            return cls(video=video)
        if not bindings.video:
            bindings.video = video
        return bindings


def apply_bindings(track_set: TrackSet, bindings: TeamBindings) -> None:
    """Переносит привязки в поля Track, чтобы их видели XML/CSV."""
    for track in track_set.player_tracks:
        team = normalize_team(bindings.assignments.get(track.id, TEAM_UNKNOWN))
        track.team = team
        track.team_conf = (
            float(bindings.confidence.get(track.id, 0.0))
            if team != TEAM_UNKNOWN
            else 0.0
        )
        if track.id in bindings.manual:
            track.team_source = "manual"
        elif team != TEAM_UNKNOWN:
            track.team_source = bindings.source
        else:
            track.team_source = "none"
        track.excluded = track.id in bindings.excluded
    for track in track_set.disc_tracks:
        track.team = TEAM_UNKNOWN
        track.team_conf = 0.0
        track.team_source = "none"
        track.excluded = track.id in bindings.excluded


def _evenly_spaced(items: list[int], count: int) -> list[int]:
    if count <= 0:
        return []
    if len(items) <= count:
        return items
    indexes = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[int(index)] for index in sorted(set(indexes))]


def _read_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
    ok, frame = cap.read()
    return frame if ok else None


def track_colour_features(
    video_path: str | Path,
    tracks: list[Track],
    *,
    sample_count: int = 8,
) -> dict[int, np.ndarray]:
    """Медианный Lab-признак торса для каждого трека.

    Читает до ``sample_count`` кадров на трек, поэтому не зависит от того,
    попал ли игрок в первый кадр.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV не открыл видео: {video_path}")

    cache: dict[int, np.ndarray] = {}
    result: dict[int, np.ndarray] = {}
    try:
        for track in tracks:
            if track.label != LABEL_PLAYER or not track.frames or track.excluded:
                continue
            frame_indexes = _evenly_spaced(sorted(track.frames), sample_count)
            features: list[np.ndarray] = []
            for frame_idx in frame_indexes:
                frame = cache.get(frame_idx)
                if frame is None:
                    frame = _read_frame(cap, frame_idx)
                    if frame is None:
                        continue
                    if len(cache) >= 6:
                        cache.clear()
                    cache[frame_idx] = frame
                box = track.frames.get(frame_idx)
                if box is None:
                    continue
                feature = extract_torso_feature(frame, np.asarray(box[:4], dtype=np.float32))
                if feature is not None:
                    features.append(feature)
            if features:
                result[track.id] = np.median(np.stack(features), axis=0).astype(np.float32)
    finally:
        cap.release()

    return result


def _team_centers(
    features: dict[int, np.ndarray], bindings: TeamBindings
) -> tuple[np.ndarray, np.ndarray] | None:
    seeds: dict[str, list[np.ndarray]] = {TEAM_A: [], TEAM_B: []}
    for track_id in bindings.manual:
        if int(track_id) in bindings.excluded:
            continue
        team = bindings.assignments.get(track_id)
        feature = features.get(int(track_id))
        if team in seeds and feature is not None:
            seeds[team].append(feature)
    if not seeds[TEAM_A] or not seeds[TEAM_B]:
        return None
    center_a = np.median(np.stack(seeds[TEAM_A]), axis=0).astype(np.float32)
    center_b = np.median(np.stack(seeds[TEAM_B]), axis=0).astype(np.float32)
    return center_a, center_b


def _sigmoid_team_probability(d_own: float, d_other: float, separation: float) -> float:
    scale = max(0.02, 0.35 * float(separation))
    z = float(np.clip((d_other - d_own) / scale, -60.0, 60.0))
    return float(1.0 / (1.0 + math.exp(-z)))


def assign_from_seeds(
    track_set: TrackSet,
    features: dict[int, np.ndarray],
    bindings: TeamBindings,
    *,
    min_probability: float = 0.70,
    max_own_distance: float = 1.20,
) -> TeamBindings:
    """Достраивает команды для неразмеченных треков по цвету формы.

    Ручные привязки не трогаются. Если вручную размечена только одна
    команда, автоподсказка не запускается: иначе «не А» может оказаться
    судьёй или запасным.
    """
    centers = _team_centers(features, bindings)
    if centers is None:
        print(
            "[CVAT] Автоподсказка по цвету: нужны эталоны обеих команд "
            "(хотя бы по одному ручному треку A и B)."
        )
        bindings.updated = _now_iso()
        return bindings

    center_a, center_b = centers
    separation = float(np.linalg.norm(center_a - center_b))
    if separation < 1e-6:
        return bindings

    for track in track_set.player_tracks:
        if track.excluded or track.id in bindings.manual:
            continue
        feature = features.get(track.id)
        if feature is None:
            bindings.assignments[track.id] = TEAM_UNKNOWN
            bindings.confidence[track.id] = 0.0
            continue
        d_a = float(np.linalg.norm(feature - center_a))
        d_b = float(np.linalg.norm(feature - center_b))
        if d_a <= d_b:
            team, d_own, d_other = TEAM_A, d_a, d_b
        else:
            team, d_own, d_other = TEAM_B, d_b, d_a
        probability = _sigmoid_team_probability(d_own, d_other, separation)
        if probability >= min_probability and d_own <= max_own_distance:
            bindings.assignments[track.id] = team
            bindings.confidence[track.id] = probability
        else:
            bindings.assignments[track.id] = TEAM_UNKNOWN
            bindings.confidence[track.id] = 0.0

    bindings.source = "color"
    bindings.updated = _now_iso()
    return bindings


def init_from_kmeans(
    track_set: TrackSet,
    features: dict[int, np.ndarray],
    bindings: TeamBindings | None = None,
) -> TeamBindings:
    """Стартовая раскладка A/B без ручных кликов (k-means по цвету).

    Команды A/B здесь условные: если кластеры поменялись местами, в UI
    есть кнопка «Swap A/B».
    """
    bindings = bindings or TeamBindings(video=track_set.meta.video_name)
    items = [
        (track.id, features[track.id])
        for track in track_set.player_tracks
        if track.id in features and not track.excluded
    ]
    if len(items) < 2:
        print("[CVAT] k-means: слишком мало треков с цветом формы.")
        return bindings

    ids = [track_id for track_id, _ in items]
    matrix = np.stack([feature for _, feature in items]).astype(np.float32)
    centers, labels = _kmeans_2(matrix)
    order = np.lexsort((centers[:, 2], centers[:, 1], centers[:, 0]))
    remap = {int(old): new for new, old in enumerate(order)}
    centers = centers[order]

    assigned: dict[int, str] = {}
    for row, track_id in enumerate(ids):
        cluster = remap[int(labels[row])]
        team = TEAM_A if cluster == 0 else TEAM_B
        other = centers[1 - cluster]
        d_own = float(np.linalg.norm(matrix[row] - centers[cluster]))
        d_other = float(np.linalg.norm(matrix[row] - other))
        separation = float(np.linalg.norm(centers[0] - centers[1]))
        assigned[track_id] = team
        bindings.confidence[track_id] = _sigmoid_team_probability(
            d_own, d_other, separation
        )

    bindings.assignments = assigned
    bindings.manual = set()
    bindings.source = "kmeans"
    bindings.updated = _now_iso()
    return bindings


def bindings_from_pairs(pairs: dict[int, str], video: str = "") -> TeamBindings:
    bindings = TeamBindings(video=video, source="manual")
    for track_id, team in pairs.items():
        bindings.set(int(track_id), team, manual=True)
    return bindings
