"""Треки для предразметки CVAT: детекция YOLO + ByteTrack + диски.

Здесь нет тактической логики 10 слотов: для CVAT нужны все игроки и все
диски, которые видит модель, с устойчивыми ID. Команды сюда не входят —
они живут в ``teams.py`` и назначаются вручную/по цвету формы.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

import cv2
import supervision as sv

from ultitrack.config import (
    CONFIDENCE_THRESHOLD,
    DISC_CONFIDENCE_THRESHOLD,
    IMAGE_SIZE,
    MODEL_PATH,
)
from ultitrack.court_mapper import FIELD_L, FIELD_W, CourtMapper

LABEL_PLAYER = "player"
LABEL_DISC = "disc"

TEAM_A = "A"
TEAM_B = "B"
TEAM_UNKNOWN = "unknown"
TEAM_VALUES = (TEAM_A, TEAM_B, TEAM_UNKNOWN)

SCHEMA_VERSION = 1

# Диск, улетевший за линию, принимаем только рядом с предыдущей позицией:
# иначе на бровке ловятся запасные диски и блики.
DISC_FIELD_MARGIN_M = 2.0
DISC_MAX_JUMP_PX = 500.0
DISC_MAX_OFF_FIELD_PX = 220.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _round(value: float, digits: int = 2) -> float:
    return round(float(value), digits)


@dataclass
class VideoMeta:
    """Метаданные ролика, без которых CVAT не соберёт таск."""

    video: str
    video_name: str
    width: int
    height: int
    fps: float
    total_frames: int
    model: str
    conf: float
    disc_conf: float
    imgsz: int
    field_filter: bool
    field_margin: float
    min_track_frames: int = 1
    truncated: bool = False
    created: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["fps"] = _round(self.fps, 3)
        data["conf"] = _round(self.conf, 3)
        data["disc_conf"] = _round(self.disc_conf, 3)
        data["field_margin"] = _round(self.field_margin, 2)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "VideoMeta":
        known = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})


@dataclass
class Track:
    """Один объект: рамки по кадрам (0-based) и привязка к команде."""

    id: int
    label: str
    frames: dict[int, tuple[float, float, float, float, float]] = field(
        default_factory=dict
    )
    source_id: int | None = None
    team: str = TEAM_UNKNOWN
    team_conf: float = 0.0
    team_source: str = "none"
    # Судья, запасной или явный ложный трек: остаётся в tracks.json,
    # но не попадает в CVAT XML.
    excluded: bool = False

    @property
    def first_frame(self) -> int:
        return min(self.frames) if self.frames else 0

    @property
    def last_frame(self) -> int:
        return max(self.frames) if self.frames else 0

    def box_at(self, frame: int) -> tuple[float, float, float, float, float] | None:
        return self.frames.get(int(frame))

    def center_at(self, frame: int) -> tuple[float, float] | None:
        box = self.box_at(frame)
        if box is None:
            return None
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    def to_dict(self) -> dict:
        return {
            "id": int(self.id),
            "label": self.label,
            "source_id": self.source_id,
            "team": self.team,
            "team_conf": _round(self.team_conf, 3),
            "team_source": self.team_source,
            "excluded": bool(self.excluded),
            "frames": {
                str(frame): [_round(v, 2) for v in box]
                for frame, box in sorted(self.frames.items())
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Track":
        frames: dict[int, tuple[float, float, float, float, float]] = {}
        for raw_frame, raw_box in (data.get("frames") or {}).items():
            values = [float(v) for v in raw_box]
            if len(values) == 4:
                values.append(1.0)
            if len(values) != 5:
                raise ValueError(f"Рамка трека {data.get('id')} должна иметь 5 чисел")
            frames[int(raw_frame)] = tuple(values)  # type: ignore[assignment]
        return cls(
            id=int(data["id"]),
            label=str(data.get("label") or LABEL_PLAYER),
            frames=frames,
            source_id=(
                None if data.get("source_id") is None else int(data["source_id"])
            ),
            team=str(data.get("team") or TEAM_UNKNOWN),
            team_conf=float(data.get("team_conf") or 0.0),
            team_source=str(data.get("team_source") or "none"),
            excluded=bool(data.get("excluded", False)),
        )


@dataclass
class TrackSet:
    meta: VideoMeta
    tracks: list[Track] = field(default_factory=list)

    @property
    def player_tracks(self) -> list[Track]:
        return [track for track in self.tracks if track.label == LABEL_PLAYER]

    @property
    def disc_tracks(self) -> list[Track]:
        return [track for track in self.tracks if track.label == LABEL_DISC]

    @property
    def track_by_id(self) -> dict[int, Track]:
        return {track.id: track for track in self.tracks}

    def at_frame(self, frame: int) -> list[tuple[Track, tuple[float, float, float, float, float]]]:
        frame = int(frame)
        return [
            (track, box)
            for track in self.tracks
            if (box := track.box_at(frame)) is not None
        ]

    def save(self, path: Path) -> Path:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "meta": self.meta.to_dict(),
            "tracks": [track.to_dict() for track in self.tracks],
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(path)
        return path

    @classmethod
    def load(cls, path: Path) -> "TrackSet":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or "meta" not in raw:
            raise ValueError(f"{path}: ожидался JSON с полями meta/tracks")
        version = int(raw.get("schema_version", 0) or 0)
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"{path}: schema_version={version} новее поддерживаемой "
                f"{SCHEMA_VERSION}; обновите UltiTrack"
            )
        meta = VideoMeta.from_dict(raw["meta"])
        tracks = [Track.from_dict(item) for item in raw.get("tracks", [])]
        return cls(meta=meta, tracks=tracks)

    def counts(self) -> dict[str, int]:
        players = self.player_tracks
        teams = {team: 0 for team in TEAM_VALUES}
        excluded = 0
        for track in players:
            if track.excluded:
                excluded += 1
                continue
            teams[track.team if track.team in teams else TEAM_UNKNOWN] += 1
        excluded += sum(1 for track in self.disc_tracks if track.excluded)
        return {
            "players": len(players),
            "discs": len(self.disc_tracks),
            "A": teams[TEAM_A],
            "B": teams[TEAM_B],
            "unknown": teams[TEAM_UNKNOWN],
            "excluded": excluded,
        }


def resolve_video_path(value: str | Path) -> Path:
    """Принимает путь или имя файла в data/raw (можно без .mp4)."""
    from ultitrack.config import RAW_VIDEOS_DIR

    path = Path(value).expanduser()
    if path.exists():
        return path.resolve()
    candidate = RAW_VIDEOS_DIR / path
    if candidate.exists():
        return candidate.resolve()
    if not path.suffix:
        candidate = RAW_VIDEOS_DIR / f"{path.name}.mp4"
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Видео не найдено: {value}. Положите файл в {RAW_VIDEOS_DIR} "
        "или укажите полный путь."
    )


def default_output_dir(video_path: Path) -> Path:
    from ultitrack.config import DATA_DIR

    return DATA_DIR / "cvat" / Path(video_path).stem


def _is_inside_field(
    mapper: CourtMapper | None, cx: float, cy: float, margin_m: float = DISC_FIELD_MARGIN_M
) -> bool:
    if mapper is None or mapper.homography_matrix is None:
        return True
    point = mapper.transform_point_to_2d(cx, cy, ground_offset_px=0.0)
    if point is None:
        return True
    x, y = float(point[0]), float(point[1])
    return -margin_m <= x <= FIELD_W + margin_m and -margin_m <= y <= FIELD_L + margin_m


def _pick_disc(
    detections: sv.Detections,
    mapper: CourtMapper | None,
    previous_center: tuple[float, float] | None,
) -> tuple[float, float, float, float, float] | None:
    """Выбирает один диск на кадр: приоритет — внутри поля.

    Если диск уже вёлcя и улетел за линию, разрешаем кандидата рядом с
    предыдущей позицией. Запасные диски на бровке без предыдущего трека
    игнорируются.
    """
    if len(detections) == 0 or detections.confidence is None:
        return None

    candidates: list[
        tuple[float, float, float, float, float, bool, float, float]
    ] = []
    for xyxy, confidence in zip(detections.xyxy, detections.confidence):
        x1, y1, x2, y2 = (float(v) for v in xyxy)
        width, height = x2 - x1, y2 - y1
        if not (4.0 <= width <= 120.0 and 4.0 <= height <= 120.0):
            continue
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        distance = (
            float("inf")
            if previous_center is None
            else math.hypot(cx - previous_center[0], cy - previous_center[1])
        )
        if distance > DISC_MAX_JUMP_PX:
            continue
        in_field = _is_inside_field(mapper, cx, cy)
        score = float(confidence) + (0.30 if in_field else 0.0)
        if previous_center is not None:
            score += max(0.0, (DISC_MAX_JUMP_PX - distance) / (DISC_MAX_JUMP_PX * 2.0))
        candidates.append(
            (x1, y1, x2, y2, float(confidence), in_field, distance, score)
        )

    if not candidates:
        return None

    inside = [item for item in candidates if item[5]]
    if inside:
        best = max(inside, key=lambda item: item[7])
    elif previous_center is not None:
        near = [item for item in candidates if item[6] <= DISC_MAX_OFF_FIELD_PX]
        if not near:
            return None
        best = max(near, key=lambda item: item[7])
    else:
        return None

    return (
        round(best[0], 2),
        round(best[1], 2),
        round(best[2], 2),
        round(best[3], 2),
        round(best[4], 3),
    )


def _load_mapper(width: int, height: int) -> CourtMapper | None:
    mapper = CourtMapper()
    if mapper.load_saved_homography((width, height)):
        print(f"[CVAT] Гомография загружена: {mapper.homography_path}")
        return mapper
    print(
        "[CVAT] Гомография не загружена — фильтр поля отключён, "
        "диски за линией могут попасть в разметку."
    )
    return None


def extract_tracks(
    video_path: str | Path,
    *,
    conf: float = CONFIDENCE_THRESHOLD,
    disc_conf: float = DISC_CONFIDENCE_THRESHOLD,
    imgsz: int = IMAGE_SIZE,
    device: str | None = None,
    field_filter: bool = True,
    field_margin: float = 1.0,
    min_track_frames: int = 3,
    max_frames: int = 0,
    progress_every: int = 120,
) -> TrackSet:
    """Прогоняет YOLO + ByteTrack по видео и возвращает все треки.

    ``min_track_frames`` отсекает короткие ложные срабатывания. Диск
    экспортируется одним треком: один кадр — одна рамка.
    """
    from trackers import ByteTrackTracker
    from ultralytics import YOLO

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Видео не найдено: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV не открыл видео: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0
    total_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    mapper = _load_mapper(width, height) if field_filter else None
    if not field_filter:
        print("[CVAT] Фильтр поля отключён (--no-field-filter).")

    model = YOLO(str(MODEL_PATH))
    tracker = ByteTrackTracker(
        lost_track_buffer=max(30, int(round(fps * 5))),
        frame_rate=max(1, int(round(fps))),
    )

    # source tracker_id -> {frame: (x1, y1, x2, y2, conf)}
    player_frames: dict[
        int, dict[int, tuple[float, float, float, float, float]]
    ] = defaultdict(dict)
    disc_frames: dict[int, tuple[float, float, float, float, float]] = {}
    previous_disc: tuple[float, float] | None = None
    missed_disc_frames = 0

    frame_idx = 0
    started = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames and frame_idx >= max_frames:
            break

        results = model(
            frame,
            imgsz=imgsz,
            conf=min(float(conf), float(disc_conf)),
            classes=[0, 1],
            verbose=False,
            device=device,
        )[0]
        detections = sv.Detections.from_ultralytics(results)

        if detections.class_id is not None and len(detections) > 0:
            player_mask = detections.class_id == 1
            players = detections[player_mask]
            if players.confidence is not None:
                players = players[players.confidence >= float(conf)]
            if len(players) > 0:
                players = players.with_nms(threshold=0.75)
                if mapper is not None:
                    players = mapper.filter_detections_in_field(
                        players,
                        margin_m=float(field_margin),
                        # Для CVAT обрезанные у нижнего края ноги — не повод
                        # выбрасывать игрока: координаты на поле нам не нужны.
                        drop_cut_feet=False,
                    )
            if len(players) > 0:
                if frame_idx == 0:
                    # ByteTrack отдаёт -1 на самом первом кадре; два прогона
                    # создают настоящие ID, как это делает PlayerTracker.
                    tracked = players
                    for _ in range(2):
                        tracked = tracker.update(tracked)
                else:
                    tracked = tracker.update(players)
                if tracked.tracker_id is not None:
                    for xyxy, tracker_id, confidence in zip(
                        tracked.xyxy, tracked.tracker_id, tracked.confidence
                    ):
                        tracker_id = int(tracker_id)
                        if tracker_id < 0:
                            continue
                        player_frames[tracker_id][frame_idx] = (
                            round(float(xyxy[0]), 2),
                            round(float(xyxy[1]), 2),
                            round(float(xyxy[2]), 2),
                            round(float(xyxy[3]), 2),
                            round(float(confidence), 3),
                        )

            disc_mask = detections.class_id == 0
            discs = detections[disc_mask]
            if discs.confidence is not None:
                discs = discs[discs.confidence >= float(disc_conf)]
            if len(discs) > 0:
                discs = discs.with_nms(threshold=0.5)
                picked = _pick_disc(discs, mapper, previous_disc)
                if picked is not None:
                    disc_frames[frame_idx] = picked
                    previous_disc = (
                        (picked[0] + picked[2]) / 2.0,
                        (picked[1] + picked[3]) / 2.0,
                    )
                    missed_disc_frames = 0
                else:
                    missed_disc_frames += 1
            else:
                missed_disc_frames += 1
            if missed_disc_frames > 20:
                # Диск потерян надолго: следующие детекции ищем заново, иначе
                # после броска не вернём его из-за ограничения прыжка.
                previous_disc = None

        frame_idx += 1
        if progress_every and frame_idx % progress_every == 0:
            elapsed = time.time() - started
            rate = frame_idx / max(elapsed, 1e-6)
            remaining = (
                (min(total_hint, max_frames) - frame_idx) / rate
                if total_hint and not max_frames
                else 0.0
            )
            eta = f", осталось ~{remaining / 60:.1f} мин" if remaining > 0 else ""
            print(
                f"[CVAT] {frame_idx}/{total_hint or '?'} кадров "
                f"({rate:.1f} кадр/с{eta})"
            )

    cap.release()

    tracks: list[Track] = []
    ordered_sources = sorted(
        (
            (min(by_frame), source_id, by_frame)
            for source_id, by_frame in player_frames.items()
            if len(by_frame) >= max(1, int(min_track_frames))
        ),
        key=lambda item: (item[0], item[1]),
    )
    for new_id, (_, source_id, by_frame) in enumerate(ordered_sources, start=1):
        tracks.append(
            Track(
                id=new_id,
                label=LABEL_PLAYER,
                source_id=int(source_id),
                frames=dict(sorted(by_frame.items())),
            )
        )

    if disc_frames:
        tracks.append(
            Track(
                id=0,
                label=LABEL_DISC,
                source_id=None,
                frames=dict(sorted(disc_frames.items())),
            )
        )

    meta = VideoMeta(
        video=str(video_path),
        video_name=video_path.name,
        width=width,
        height=height,
        fps=fps,
        total_frames=frame_idx,
        model=MODEL_PATH.name,
        conf=float(conf),
        disc_conf=float(disc_conf),
        imgsz=int(imgsz),
        field_filter=bool(field_filter),
        field_margin=float(field_margin),
        min_track_frames=int(min_track_frames),
        truncated=bool(max_frames and frame_idx >= max_frames and total_hint > frame_idx),
    )
    track_set = TrackSet(meta=meta, tracks=tracks)
    counts = track_set.counts()
    print(
        f"[CVAT] Готово: {frame_idx} кадров, игроков {counts['players']}, "
        f"диск {counts['discs']}, команд пока неизвестно "
        f"({counts['unknown']} unknown)."
    )
    return track_set

