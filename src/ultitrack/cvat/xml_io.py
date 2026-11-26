"""Чтение и запись CVAT for video 1.1 (interpolation mode).

Формат: ``<track>`` с keyframe-рамками, ``outside="1"`` закрывает трек на
время пропажи, атрибут ``team`` immutable (одинаковый для всего трека).
CVAT интерполирует рамки между keyframe, поэтому шаг keyframe можно
увеличить (``stride``), не теряя плавности.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

from ultitrack.cvat.teams import TeamBindings, normalize_team
from ultitrack.cvat.tracks import (
    LABEL_DISC,
    LABEL_PLAYER,
    TEAM_UNKNOWN,
    TEAM_VALUES,
    Track,
    TrackSet,
)

DEFAULT_PLAYER_LABEL = LABEL_PLAYER
DEFAULT_DISC_LABEL = LABEL_DISC
DEFAULT_TEAM_ATTR = "team"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _text(parent: ET.Element, tag: str, value: object = "") -> ET.Element:
    element = ET.SubElement(parent, tag)
    element.text = str(value)
    return element


def _split_segments(frames: list[int], bridge_gap: int) -> list[list[int]]:
    """Режет присутствие трека на сегменты; короткие пропуски склеивает."""
    if not frames:
        return []
    bridge_gap = max(0, int(bridge_gap))
    segments: list[list[int]] = [[frames[0]]]
    for frame in frames[1:]:
        if frame - segments[-1][-1] <= bridge_gap + 1:
            segments[-1].append(frame)
        else:
            segments.append([frame])
    return segments


def _keyframes(frames: list[int], stride: int) -> list[int]:
    """Первая/последняя рамка сегмента плюс каждая stride-я."""
    stride = max(1, int(stride))
    keys = set(frames[::stride])
    keys.add(frames[0])
    keys.add(frames[-1])
    return sorted(keys)


def _clamp_box(
    box: tuple[float, float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = (float(value) for value in box[:4])
    if width > 0:
        x1 = min(max(x1, 0.0), float(width - 1))
        x2 = min(max(x2, 0.0), float(width - 1))
    if height > 0:
        y1 = min(max(y1, 0.0), float(height - 1))
        y2 = min(max(y2, 0.0), float(height - 1))
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    return round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)


def _box_specs(
    track: Track,
    total_frames: int,
    stride: int,
    bridge_gap: int,
) -> list[tuple[int, tuple[float, float, float, float], bool]]:
    """Список (frame, xyxy, outside) для keyframe-рамок CVAT."""
    frames = sorted(track.frames)
    if not frames:
        return []
    segments = _split_segments(frames, bridge_gap)
    specs: list[tuple[int, tuple[float, float, float, float], bool]] = []
    for index, segment in enumerate(segments):
        for frame in _keyframes(segment, stride):
            box = track.frames.get(frame)
            if box is None:
                continue
            specs.append((frame, tuple(float(v) for v in box[:4]), False))
        close_frame = segment[-1] + 1
        if close_frame >= total_frames:
            continue
        # Закрываем и в конце ролика, и перед длинной пропажей, чтобы CVAT
        # не растягивал последнюю рамку через всю пропажу.
        is_last_segment = index == len(segments) - 1
        if is_last_segment or close_frame < segments[index + 1][0]:
            last_box = track.frames[segment[-1]]
            specs.append(
                (close_frame, tuple(float(v) for v in last_box[:4]), True)
            )
    return specs


def _write_labels(
    task: ET.Element,
    player_label: str,
    disc_label: str,
    team_attr: str,
) -> None:
    labels = ET.SubElement(task, "labels")

    player = ET.SubElement(labels, "label")
    _text(player, "name", player_label)
    attributes = ET.SubElement(player, "attributes")
    attribute = ET.SubElement(attributes, "attribute")
    _text(attribute, "name", team_attr)
    _text(attribute, "mutable", "False")
    _text(attribute, "input_type", "select")
    _text(attribute, "default_value", TEAM_UNKNOWN)
    _text(attribute, "values", "\n".join(TEAM_VALUES))

    disc = ET.SubElement(labels, "label")
    _text(disc, "name", disc_label)
    ET.SubElement(disc, "attributes")


def write_cvat_video_xml(
    track_set: TrackSet,
    output_path: Path,
    *,
    player_label: str = DEFAULT_PLAYER_LABEL,
    disc_label: str = DEFAULT_DISC_LABEL,
    team_attr: str = DEFAULT_TEAM_ATTR,
    stride: int = 3,
    disc_stride: int = 1,
    bridge_gap: int = 10,
    source: str = "auto",
    dumped: str | None = None,
) -> Path:
    """Пишет annotations.xml, который импортируется в CVAT for video 1.1."""
    if player_label == disc_label:
        raise ValueError("Метки игрока и диска должны различаться")
    if not team_attr:
        raise ValueError("Имя атрибута команды не может быть пустым")

    meta = track_set.meta
    total_frames = max(1, int(meta.total_frames))
    width = max(0, int(meta.width))
    height = max(0, int(meta.height))

    root = ET.Element("annotations")
    _text(root, "version", "1.1")

    meta_el = ET.SubElement(root, "meta")
    task = ET.SubElement(meta_el, "task")
    _text(task, "id", 0)
    _text(task, "name", Path(meta.video_name).stem or "ultitrack")
    _text(task, "size", total_frames)
    _text(task, "mode", "interpolation")
    _text(task, "overlap", 0)
    _text(task, "bugtracker", "")
    _text(task, "flipped", "False")
    created = meta.created or _now_iso()
    _text(task, "created", created)
    _text(task, "updated", created)
    _write_labels(task, player_label, disc_label, team_attr)

    segments = ET.SubElement(task, "segments")
    segment = ET.SubElement(segments, "segment")
    _text(segment, "id", 0)
    _text(segment, "start", 0)
    _text(segment, "stop", total_frames - 1)
    _text(segment, "url", "")

    owner = ET.SubElement(task, "owner")
    _text(owner, "username", "ultitrack")
    _text(owner, "email", "")

    original_size = ET.SubElement(task, "original_size")
    _text(original_size, "width", width)
    _text(original_size, "height", height)

    _text(meta_el, "dumped", dumped or _now_iso())

    for track in sorted(track_set.tracks, key=lambda item: item.id):
        if track.excluded:
            continue
        is_player = track.label == LABEL_PLAYER
        if not is_player and track.label != LABEL_DISC:
            continue
        label = player_label if is_player else disc_label
        step = stride if is_player else disc_stride
        rendered: list[tuple[int, tuple[float, float, float, float], bool]] = []
        for frame, xyxy, outside in _box_specs(
            track, total_frames, step, bridge_gap
        ):
            clamped = _clamp_box((*xyxy, 1.0), width, height)
            if clamped is not None:
                rendered.append((frame, clamped, outside))
        if not rendered:
            continue

        track_el = ET.SubElement(
            root,
            "track",
            {
                "id": str(track.id),
                "label": label,
                "source": source,
            },
        )
        team = normalize_team(track.team) if is_player else TEAM_UNKNOWN
        for frame, (x1, y1, x2, y2), outside in rendered:
            box_el = ET.SubElement(
                track_el,
                "box",
                {
                    "frame": str(frame),
                    "outside": "1" if outside else "0",
                    "occluded": "0",
                    "keyframe": "1",
                    "xtl": f"{x1:.2f}",
                    "ytl": f"{y1:.2f}",
                    "xbr": f"{x2:.2f}",
                    "ybr": f"{y2:.2f}",
                },
            )
            if is_player:
                attribute = ET.SubElement(box_el, "attribute", {"name": team_attr})
                attribute.text = team

    ET.indent(root, space="  ")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    tree = ET.ElementTree(root)
    tree.write(temporary, encoding="utf-8", xml_declaration=True)
    temporary.replace(output_path)
    return output_path


def write_labels_json(
    output_path: Path,
    *,
    player_label: str = DEFAULT_PLAYER_LABEL,
    disc_label: str = DEFAULT_DISC_LABEL,
    team_attr: str = DEFAULT_TEAM_ATTR,
) -> Path:
    """Справка для настройки меток в CVAT (не импортируется автоматически)."""
    payload = {
        "labels": [
            {
                "name": player_label,
                "type": "rectangle",
                "attributes": [
                    {
                        "name": team_attr,
                        "input_type": "select",
                        "mutable": False,
                        "default_value": TEAM_UNKNOWN,
                        "values": list(TEAM_VALUES),
                    }
                ],
            },
            {
                "name": disc_label,
                "type": "rectangle",
                "attributes": [],
            },
        ]
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_path


def write_summary_csv(track_set: TrackSet, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "track_id",
                "label",
                "team",
                "team_conf",
                "team_source",
                "excluded",
                "first_frame",
                "last_frame",
                "frames",
                "mean_conf",
            ]
        )
        for track in sorted(track_set.tracks, key=lambda item: item.id):
            confidences = [box[4] for box in track.frames.values()]
            writer.writerow(
                [
                    track.id,
                    track.label,
                    track.team,
                    f"{track.team_conf:.3f}",
                    track.team_source,
                    1 if track.excluded else 0,
                    track.first_frame,
                    track.last_frame,
                    len(track.frames),
                    f"{sum(confidences) / len(confidences):.3f}" if confidences else "0",
                ]
            )
    return output_path


# --------------------------------------------------------------------------
# Импорт обратно из CVAT


@dataclass
class ParsedBox:
    frame: int
    xyxy: tuple[float, float, float, float]
    outside: bool = False
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass
class ParsedTrack:
    id: int
    label: str
    team: str
    boxes: list[ParsedBox]
    source: str = ""

    @property
    def first_frame(self) -> int:
        present = [box.frame for box in self.boxes if not box.outside]
        return min(present) if present else 0

    @property
    def last_frame(self) -> int:
        present = [box.frame for box in self.boxes if not box.outside]
        return max(present) if present else 0

    @property
    def keyframe_count(self) -> int:
        return sum(1 for box in self.boxes if not box.outside)


@dataclass
class ParsedCvat:
    path: str
    name: str
    width: int
    height: int
    total_frames: int
    labels: list[str]
    tracks: list[ParsedTrack]

    def team_assignments(self) -> dict[int, str]:
        return {
            track.id: track.team
            for track in self.tracks
            if track.label.lower() in (LABEL_PLAYER.lower(), "person")
        }


def _parse_box_xyxy(box_el: ET.Element) -> tuple[float, float, float, float] | None:
    if box_el.get("xtl") is not None:
        return (
            float(box_el.get("xtl", "0")),
            float(box_el.get("ytl", "0")),
            float(box_el.get("xbr", "0")),
            float(box_el.get("ybr", "0")),
        )
    if box_el.get("x") is not None:
        x = float(box_el.get("x", "0"))
        y = float(box_el.get("y", "0"))
        w = float(box_el.get("w", "0"))
        h = float(box_el.get("h", "0"))
        return x, y, x + w, y + h
    return None


def parse_cvat_video_xml(
    path: Path, *, team_attr: str = DEFAULT_TEAM_ATTR
) -> ParsedCvat:
    path = Path(path)
    root = ET.parse(path).getroot()
    task = root.find("meta/task")
    name = (task.findtext("name") if task is not None else "") or path.stem
    total_frames = int(task.findtext("size", "0") if task is not None else "0") or 0
    original_size = task.find("original_size") if task is not None else None
    width = int((original_size.findtext("width") or "0")) if original_size is not None else 0
    height = int((original_size.findtext("height") or "0")) if original_size is not None else 0
    labels = [
        element.text or ""
        for element in root.findall("meta/task/labels/label/name")
        if element.text
    ]

    tracks: list[ParsedTrack] = []
    skipped_shapes = 0
    for track_el in root.findall("track"):
        try:
            track_id = int(track_el.get("id", "-1"))
        except ValueError:
            continue
        label = track_el.get("label", "")
        boxes: list[ParsedBox] = []
        team_votes: Counter[str] = Counter()
        for child in track_el:
            if child.tag != "box":
                if child.tag in ("polygon", "polyline", "points", "ellipse"):
                    skipped_shapes += 1
                continue
            xyxy = _parse_box_xyxy(child)
            if xyxy is None:
                continue
            frame = int(child.get("frame", "0"))
            outside = bool(int(child.get("outside", "0") or 0))
            attributes = {
                attribute.get("name", ""): (attribute.text or "")
                for attribute in child.findall("attribute")
                if attribute.get("name")
            }
            team = attributes.get(team_attr)
            if team:
                try:
                    team_votes[normalize_team(team)] += 1
                except ValueError:
                    team_votes[TEAM_UNKNOWN] += 1
            boxes.append(
                ParsedBox(
                    frame=frame,
                    xyxy=xyxy,
                    outside=outside,
                    attributes=attributes,
                )
            )
        boxes.sort(key=lambda item: item.frame)
        team = team_votes.most_common(1)[0][0] if team_votes else TEAM_UNKNOWN
        tracks.append(
            ParsedTrack(
                id=track_id,
                label=label,
                team=team,
                boxes=boxes,
                source=track_el.get("source", ""),
            )
        )

    if total_frames <= 0:
        max_frame = max(
            (box.frame for track in tracks for box in track.boxes), default=0
        )
        total_frames = max_frame + 1

    if skipped_shapes:
        print(
            f"[CVAT] В {path} пропущено не-box фигур: {skipped_shapes}. "
            "Пайплайн умеет читать только рамки."
        )

    return ParsedCvat(
        path=str(path),
        name=name,
        width=width,
        height=height,
        total_frames=total_frames,
        labels=labels,
        tracks=tracks,
    )


def _interpolate(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
    ratio: float,
) -> tuple[float, float, float, float]:
    return tuple(
        left[index] + (right[index] - left[index]) * ratio for index in range(4)
    )  # type: ignore[return-value]


def expand_present_boxes(
    track: ParsedTrack, total_frames: int = 0
) -> list[tuple[int, tuple[float, float, float, float]]]:
    """Разворачивает keyframe в покадровые рамки с линейной интерполяцией.

    Учитывает семантику CVAT: ``outside="1"`` прячет объект до следующего
    ``outside="0"``, а последняя рамка без закрытия держится до конца ролика.
    """
    result: list[tuple[int, tuple[float, float, float, float]]] = []
    current: ParsedBox | None = None
    for box in sorted(track.boxes, key=lambda item: item.frame):
        if box.outside:
            if current is not None:
                for frame in range(current.frame + 1, box.frame):
                    result.append((frame, current.xyxy))
            current = None
            continue
        if current is not None and box.frame > current.frame + 1:
            span = box.frame - current.frame
            for offset in range(1, span):
                ratio = offset / span
                result.append(
                    (current.frame + offset, _interpolate(current.xyxy, box.xyxy, ratio))
                )
        result.append((box.frame, box.xyxy))
        current = box

    if current is not None and total_frames > current.frame + 1:
        for frame in range(current.frame + 1, total_frames):
            result.append((frame, current.xyxy))
    return result


def write_import_json(
    parsed: ParsedCvat,
    output_path: Path,
    *,
    per_frame: bool = False,
) -> Path:
    tracks_payload = []
    for track in parsed.tracks:
        if per_frame:
            boxes = [
                [frame, *[round(value, 2) for value in xyxy]]
                for frame, xyxy in expand_present_boxes(track, parsed.total_frames)
            ]
        else:
            boxes = [
                [box.frame, *[round(value, 2) for value in box.xyxy]]
                for box in track.boxes
                if not box.outside
            ]
        tracks_payload.append(
            {
                "id": track.id,
                "label": track.label,
                "team": track.team,
                "source": track.source,
                "boxes": boxes,
            }
        )
    payload = {
        "meta": {
            "source_xml": parsed.path,
            "name": parsed.name,
            "width": parsed.width,
            "height": parsed.height,
            "total_frames": parsed.total_frames,
            "labels": parsed.labels,
            "per_frame": per_frame,
            "parsed": _now_iso(),
        },
        "tracks": tracks_payload,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return output_path


def write_import_csv(parsed: ParsedCvat, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "track_id",
                "label",
                "team",
                "first_frame",
                "last_frame",
                "keyframes",
                "present_frames",
            ]
        )
        for track in sorted(parsed.tracks, key=lambda item: item.id):
            present_frames = len(expand_present_boxes(track, parsed.total_frames))
            writer.writerow(
                [
                    track.id,
                    track.label,
                    track.team,
                    track.first_frame,
                    track.last_frame,
                    track.keyframe_count,
                    present_frames,
                ]
            )
    return output_path


def write_import_teams(
    parsed: ParsedCvat,
    output_path: Path,
    *,
    player_label: str = DEFAULT_PLAYER_LABEL,
) -> Path:
    """Сохраняет поправленные в CVAT команды в формате teams.json."""
    bindings = TeamBindings(video=parsed.name, source="cvat")
    for track in parsed.tracks:
        if track.label.lower() != player_label.lower():
            continue
        bindings.set(track.id, track.team, manual=True, confidence=1.0)
    return bindings.save(output_path)
