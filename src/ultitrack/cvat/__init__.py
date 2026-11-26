"""Предразметка для CVAT: треки модели, ручные команды и обмен XML.

Пакет не импортирует YOLO и ultralytics на уровне ``__init__``: XML/JSON
и привязки команд должны работать без загрузки весов.
"""

from ultitrack.cvat.tracks import (
    LABEL_DISC,
    LABEL_PLAYER,
    TEAM_A,
    TEAM_B,
    TEAM_UNKNOWN,
    TEAM_VALUES,
    Track,
    TrackSet,
    VideoMeta,
)

__all__ = [
    "LABEL_DISC",
    "LABEL_PLAYER",
    "TEAM_A",
    "TEAM_B",
    "TEAM_UNKNOWN",
    "TEAM_VALUES",
    "Track",
    "TrackSet",
    "VideoMeta",
]
