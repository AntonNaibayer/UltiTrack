"""Классификация игроков по командам (A/B) для зального алтимата 5x5.

Модуль работает поверх уже сматченных слотов RosterTracker: он не участвует
в матчинге и не может сломать привязку слотов к игрокам. Задача — по кропу
торса определить цвет формы и разложить 10 слотов ровно на 5 + 5.

Алгоритм:
  1. На первом кадре проверяется стартовая расстановка: если 10 слотов
     делятся на две линии по 5 у противоположных концов поля, команды
     берутся из геометрии линий. Это надёжнее цвета формы: белые/тёмные
     майки, блики и судьи в кадре дают выбросы.
  2. Если стартовых линий нет (поинт уже в игре), для каждого слота на
     первых кадрах копится Lab-признак торса.
  3. По медианам признаков запускается k-means на 2 кластера.
  4. Баланс 5x5 обеспечивается венгерским алгоритмом: каждый кластер
     дублируется 5 раз, поэтому 6x4 математически невозможно.
  5. После калибровки метка команды фиксируется, а team_conf сглаживается
     EMA по кадрам, чтобы цвет фишки не мигал.

Калибровка занимает TEAM_CALIBRATION_SECONDS секунд. До неё team_conf = 0,
поэтому радар и аннотации используют нейтральный цвет.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from ultitrack.config import (
    FIELD_LENGTH,
    TEAM_BACKGROUND_DIST,
    TEAM_CONFIDENCE_EMA_BETA,
    TEAM_FEATURE_EMA_ALPHA,
    TEAM_LINEUP_MAX_SPREAD_M,
    TEAM_LINEUP_MIN_GAP_M,
    TEAM_MAX_OBSERVATIONS,
    TEAM_MIN_OBSERVATIONS,
    TEAM_MIN_PIXELS,
    TEAM_TORSO_X0,
    TEAM_TORSO_X1,
    TEAM_TORSO_Y0,
    TEAM_TORSO_Y1,
)

TEAM_A = "A"
TEAM_B = "B"
EXPECTED_SLOTS = 10
TEAM_SIZE = 5

# При малом разделении центров команды плохо различимы: не даём уверенности
# улететь к 1.0 только из-за маленькой разницы расстояний.
_CONFIDENCE_SEPARATION_REF = 0.30

# Веса компонент Lab-признака: цвет (a, b) важнее яркости (L), а разброс —
# вспомогательный сигнал. Такой вектор устойчивее к теням и бликам паркета.
_FEATURE_WEIGHTS = np.array([0.6, 1.0, 1.0, 0.3, 0.4, 0.4], dtype=np.float32)


@dataclass(frozen=True)
class TeamAssignment:
    """Результат классификации одного слота."""

    team: str
    confidence: float


def _clip_box(
    xyxy: np.ndarray, frame_w: int, frame_h: int
) -> tuple[int, int, int, int] | None:
    values = np.asarray(xyxy, dtype=np.float64).reshape(-1)
    if values.size < 4 or not np.all(np.isfinite(values[:4])):
        return None
    x1, y1, x2, y2 = (int(round(float(v))) for v in values[:4])
    x1 = max(0, min(x1, frame_w - 2))
    y1 = max(0, min(y1, frame_h - 2))
    x2 = max(x1 + 2, min(x2, frame_w))
    y2 = max(y1 + 2, min(y2, frame_h))
    if x2 - x1 < 8 or y2 - y1 < 16:
        return None
    return x1, y1, x2, y2


def _lab_pixels(image: np.ndarray) -> np.ndarray:
    """BGR -> Lab (float32), пиксели построчно."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    return lab.reshape(-1, 3).astype(np.float32)


def _background_median(
    frame: np.ndarray,
    torso: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
) -> np.ndarray | None:
    """Медианный Lab фона в кольце вокруг торса.

    Кольцо берётся с небольшим отступом вокруг кропа торса и исключает сам
    торс. На зальном видео там обычно паркет/стена, поэтому медиана даёт
    адаптивный цвет фона без хардкода цвета зала.
    """
    tx1, ty1, tx2, ty2 = torso
    bw, bh = tx2 - tx1, ty2 - ty1
    ex = max(2, int(round(0.15 * bw)))
    ey = max(2, int(round(0.12 * bh)))
    rx1, ry1 = max(0, tx1 - ex), max(0, ty1 - ey)
    rx2, ry2 = min(frame_w, tx2 + ex), min(frame_h, ty2 + ey)
    if rx2 - rx1 < 4 or ry2 - ry1 < 4:
        return None

    ring = frame[ry1:ry2, rx1:rx2]
    mask = np.ones(ring.shape[:2], dtype=bool)
    mask[ty1 - ry1 : ty2 - ry1, tx1 - rx1 : tx2 - rx1] = False
    pixels = _lab_pixels(ring)[mask.reshape(-1)]
    if len(pixels) < 20:
        return None
    return np.median(pixels, axis=0)


def extract_torso_feature(
    frame: np.ndarray, xyxy: np.ndarray
) -> np.ndarray | None:
    """Считает Lab-признак торса игрока или None, если кроп слишком мал.

    Кроп: 20..80% ширины и 20..55% высоты бокса. Так голова, шорты, ноги и
    фон паркета попадают в кроп минимально. Дополнительно пиксели, близкие
    к медианному цвету фона вокруг торса, отбрасываются.
    """
    if frame is None or frame.size == 0:
        return None
    frame_h, frame_w = frame.shape[:2]
    box = _clip_box(xyxy, frame_w, frame_h)
    if box is None:
        return None

    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    tx1 = x1 + int(round(TEAM_TORSO_X0 * bw))
    tx2 = x1 + int(round(TEAM_TORSO_X1 * bw))
    ty1 = y1 + int(round(TEAM_TORSO_Y0 * bh))
    ty2 = y1 + int(round(TEAM_TORSO_Y1 * bh))
    tx1, tx2 = max(0, tx1), min(frame_w, tx2)
    ty1, ty2 = max(0, ty1), min(frame_h, ty2)
    if tx2 - tx1 < 4 or ty2 - ty1 < 4:
        return None

    crop = frame[ty1:ty2, tx1:tx2]
    if crop.size == 0:
        return None
    pixels = _lab_pixels(crop)
    if len(pixels) < 12:
        return None

    bg = _background_median(frame, (tx1, ty1, tx2, ty2), frame_w, frame_h)
    if bg is not None:
        dist_to_bg = np.linalg.norm(pixels - bg, axis=1)
        keep = dist_to_bg > TEAM_BACKGROUND_DIST
        # Если фон «съел» почти весь торс, лучше не отбрасывать пиксели:
        # значит, кольцо попало в форму/соседнего игрока.
        if int(keep.sum()) >= max(TEAM_MIN_PIXELS, int(0.25 * len(pixels))):
            pixels = pixels[keep]

    if len(pixels) < TEAM_MIN_PIXELS:
        return None

    # Отсекаем блики и выбросы по каждому каналу через MAD.
    median = np.median(pixels, axis=0)
    mad = np.median(np.abs(pixels - median), axis=0) + 1.0
    inliers = np.all(np.abs(pixels - median) <= 2.5 * mad, axis=1)
    if int(inliers.sum()) >= TEAM_MIN_PIXELS:
        pixels = pixels[inliers]

    mean = pixels.mean(axis=0)
    std = pixels.std(axis=0)

    # Нормируем L в 0..1, a/b из 0..255 со смещением 128 в -1..1.
    feature = np.array(
        [
            mean[0] / 255.0,
            (mean[1] - 128.0) / 128.0,
            (mean[2] - 128.0) / 128.0,
            std[0] / 255.0,
            std[1] / 128.0,
            std[2] / 128.0,
        ],
        dtype=np.float32,
    )
    return feature * _FEATURE_WEIGHTS


def _kmeans_2(features: np.ndarray, iterations: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Детерминированный k-means на 2 кластера (без random seed)."""
    n = len(features)
    if n == 0:
        raise ValueError("kmeans: пустой набор признаков")
    if n == 1:
        return features.copy(), np.zeros(1, dtype=int)

    diff = features[:, None, :] - features[None, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    i, j = np.unravel_index(int(np.argmax(d2)), d2.shape)
    centers = features[[i, j]].copy()

    labels = np.zeros(n, dtype=int)
    for _ in range(iterations):
        dist = np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
        labels = np.argmin(dist, axis=1)
        if len(np.unique(labels)) < 2:
            # Один кластер пуст — разносим центры по самой дальней точке.
            far = int(np.argmax(dist[:, 0]))
            centers[1] = features[far]
            continue

        new_centers = centers.copy()
        for k in range(2):
            members = features[labels == k]
            if len(members) > 0:
                new_centers[k] = members.mean(axis=0)
        if np.allclose(new_centers, centers, atol=1e-6):
            break
        centers = new_centers

    return centers, labels


class TeamClassifier:
    """Копит цвет торса по слотам и выдаёт стабильное разбиение 5x5."""

    def __init__(
        self,
        calibration_frames: int = 45,
        min_observations: int = TEAM_MIN_OBSERVATIONS,
        max_observations: int = TEAM_MAX_OBSERVATIONS,
        feature_ema_alpha: float = TEAM_FEATURE_EMA_ALPHA,
        confidence_ema_beta: float = TEAM_CONFIDENCE_EMA_BETA,
    ) -> None:
        self.calibration_frames = max(1, int(calibration_frames))
        self.min_observations = max(1, int(min_observations))
        self.max_observations = max(self.min_observations, int(max_observations))
        self.feature_ema_alpha = float(np.clip(feature_ema_alpha, 0.0, 1.0))
        self.confidence_ema_beta = float(np.clip(confidence_ema_beta, 0.0, 1.0))

        self._observations: dict[int, list[np.ndarray]] = defaultdict(list)
        self._features: dict[int, np.ndarray] = {}
        self._teams: dict[int, str] = {}
        self._confidence: dict[int, float] = {}
        self._centers: np.ndarray | None = None
        self._separation = 0.0
        self._frames_seen = 0
        self._calibrated = False
        self._timeout_warned = False
        self._initial_positions: dict[int, np.ndarray] = {}
        self._calibration_source = ""

    # ------------------------------------------------------------------ API

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    @property
    def team_map(self) -> dict[int, str]:
        return dict(self._teams)

    @property
    def calibration_source(self) -> str:
        """``lineup`` — команды взяты из стартовых линий, ``color`` — из формы."""
        return self._calibration_source

    def set_initial_positions(self, positions: dict[int, np.ndarray]) -> None:
        """Запоминает 2D-позиции слотов на первом кадре поинта.

        По ним проверяется стартовая расстановка: 5 игроков на одной линии
        эндзоны, 5 — на другой. Если расстановка подтверждена, она надёжнее
        цвета формы (блики, белые/тёмные майки, судьи в кадре).
        """
        self._initial_positions = {
            int(slot_id): np.asarray(pos, dtype=np.float32).reshape(2).copy()
            for slot_id, pos in positions.items()
        }

    def reference_feature(self, slot_id: int) -> np.ndarray | None:
        """Возвращает текущий цветовой эталон слота для anti-swap матчинга."""
        feature = self._features.get(int(slot_id))
        return None if feature is None else feature.copy()

    @staticmethod
    def feature_for_box(frame: np.ndarray, box: np.ndarray) -> np.ndarray | None:
        """Публичная обёртка над extract_torso_feature для RosterTracker."""
        return extract_torso_feature(frame, box)

    def reset(self) -> None:
        self._observations.clear()
        self._features.clear()
        self._teams.clear()
        self._confidence.clear()
        self._centers = None
        self._separation = 0.0
        self._frames_seen = 0
        self._calibrated = False
        self._timeout_warned = False
        self._initial_positions.clear()
        self._calibration_source = ""

    def update(
        self, frame: np.ndarray, slot_boxes: dict[int, np.ndarray]
    ) -> dict[int, TeamAssignment]:
        """Обновляет признаки по сматченным слотам и возвращает назначения.

        `slot_boxes` — только слоты, у которых на этом кадре есть детекция.
        Потерянные слоты сохраняют последнюю команду и confidence.
        """
        self._frames_seen += 1

        for slot_id, box in slot_boxes.items():
            feature = extract_torso_feature(frame, box)
            if feature is None:
                continue

            if not self._calibrated:
                history = self._observations[int(slot_id)]
                history.append(feature)
                if len(history) > self.max_observations:
                    del history[0 : len(history) - self.max_observations]
            else:
                previous = self._features.get(int(slot_id))
                if previous is None:
                    self._features[int(slot_id)] = feature
                else:
                    alpha = self.feature_ema_alpha
                    self._features[int(slot_id)] = (
                        (1.0 - alpha) * previous + alpha * feature
                    )

        if not self._calibrated:
            if self._ready():
                self._calibrate()
            elif (
                not self._timeout_warned
                and self._frames_seen > self.calibration_frames * 6
            ):
                self._timeout_warned = True
                print(
                    "[TeamClassifier] Недостаточно наблюдений для калибровки "
                    f"({len(self._observations)}/10 слотов); цвета остаются нейтральными."
                )

        if self._calibrated:
            self._update_confidences()
            return {
                slot_id: TeamAssignment(team, self._confidence.get(slot_id, 0.5))
                for slot_id, team in self._teams.items()
            }
        return {}

    # ------------------------------------------------------------- калибровка

    def _ready(self) -> bool:
        if len(self._observations) != EXPECTED_SLOTS:
            return False
        if self._frames_seen < self.calibration_frames:
            return False
        return all(
            len(history) >= self.min_observations
            for history in self._observations.values()
        )

    def _detect_lineup(
        self, slot_ids: list[int]
    ) -> tuple[list[int], list[int]] | None:
        """Ищет стартовые линии: 5 игроков у одного конца поля, 5 — у другого.

        Возвращает (нижняя_линия, верхняя_линия) — списки slot_id. Нижняя
        линия (меньший y) считается командой A. Если расстановка не похожа
        на линии (поинт уже в игре), возвращает None и работает цвет.
        """
        if len(slot_ids) != EXPECTED_SLOTS:
            return None
        positions = {sid: self._initial_positions.get(sid) for sid in slot_ids}
        if any(pos is None for pos in positions.values()):
            return None

        ys = np.array([positions[sid][1] for sid in slot_ids], dtype=np.float64)
        order = np.argsort(ys)
        sorted_y = ys[order]
        gaps = np.diff(sorted_y)
        if len(gaps) == 0:
            return None
        gap_index = int(np.argmax(gaps))
        if float(gaps[gap_index]) < TEAM_LINEUP_MIN_GAP_M:
            return None

        low_idx = order[: gap_index + 1]
        high_idx = order[gap_index + 1 :]
        if len(low_idx) != TEAM_SIZE or len(high_idx) != TEAM_SIZE:
            return None

        low_y = ys[low_idx]
        high_y = ys[high_idx]
        if float(np.ptp(low_y)) > TEAM_LINEUP_MAX_SPREAD_M:
            return None
        if float(np.ptp(high_y)) > TEAM_LINEUP_MAX_SPREAD_M:
            return None
        # Линии должны быть на противоположных половинах поля, иначе это
        # просто две группы игроков в одной зоне.
        if float(low_y.mean()) >= FIELD_LENGTH / 2.0:
            return None
        if float(high_y.mean()) <= FIELD_LENGTH / 2.0:
            return None

        low_slots = [slot_ids[int(i)] for i in low_idx]
        high_slots = [slot_ids[int(i)] for i in high_idx]
        return low_slots, high_slots

    def _calibrate(self) -> None:
        slot_ids = sorted(self._observations)
        features = {
            slot_id: np.median(
                np.stack(self._observations[slot_id]), axis=0
            ).astype(np.float32)
            for slot_id in slot_ids
        }
        lineup = self._detect_lineup(slot_ids)

        if lineup is not None:
            team_a_slots, team_b_slots = lineup
            centers = np.stack(
                [
                    np.stack([features[sid] for sid in team_a_slots]).mean(axis=0),
                    np.stack([features[sid] for sid in team_b_slots]).mean(axis=0),
                ]
            ).astype(np.float32)
            self._centers = centers
            self._separation = float(np.linalg.norm(centers[0] - centers[1]))
            team_by_slot = {sid: TEAM_A for sid in team_a_slots}
            team_by_slot.update({sid: TEAM_B for sid in team_b_slots})
            for slot_id in slot_ids:
                team = team_by_slot[slot_id]
                self._teams[slot_id] = team
                self._features[slot_id] = features[slot_id]
                self._confidence[slot_id] = self._confidence_for(
                    features[slot_id], team
                )
            self._calibration_source = "lineup"
        else:
            feature_array = np.stack([features[sid] for sid in slot_ids])
            centers, _ = _kmeans_2(feature_array)
            # Детерминированный порядок команд: сортируем центры по L, a, b,
            # чтобы при одинаковых данных A/B не менялись местами между запусками.
            order = np.lexsort((centers[:, 2], centers[:, 1], centers[:, 0]))
            centers = centers[order]

            cost = np.linalg.norm(
                feature_array[:, None, :] - centers[None, :, :], axis=2
            )
            # 10 слотов -> 10 колонок: по 5 копий каждого центра. Венгерский
            # алгоритм обязан выбрать ровно 5 колонок на команду.
            balanced_cost = np.repeat(cost, TEAM_SIZE, axis=1)
            row_idx, col_idx = linear_sum_assignment(balanced_cost)

            self._centers = centers
            self._separation = float(np.linalg.norm(centers[0] - centers[1]))
            for row, col in zip(row_idx, col_idx):
                slot_id = slot_ids[int(row)]
                team = TEAM_A if int(col) < TEAM_SIZE else TEAM_B
                self._teams[slot_id] = team
                self._features[slot_id] = features[slot_id]
                self._confidence[slot_id] = self._confidence_for(
                    features[slot_id], team
                )
            self._calibration_source = "color"

        self._calibrated = True
        counts = {team: list(self._teams.values()).count(team) for team in ("A", "B")}
        source_text = (
            "по стартовым линиям"
            if self._calibration_source == "lineup"
            else "по цвету формы"
        )
        print(
            f"[TeamClassifier] Калибровка завершена ({source_text}): "
            f"A={counts['A']}, B={counts['B']}, "
            f"разделение центров={self._separation:.2f}"
        )

    def _confidence_for(self, feature: np.ndarray, team: str) -> float:
        if self._centers is None:
            return 0.5
        own = 0 if team == TEAM_A else 1
        distances = np.linalg.norm(self._centers - feature[None, :], axis=1)
        d_own = float(distances[own])
        d_other = float(distances[1 - own])
        if self._separation < 1e-3:
            return 0.5
        scale = max(0.02, 0.35 * self._separation)
        z = np.clip((d_other - d_own) / scale, -60.0, 60.0)
        raw = float(1.0 / (1.0 + np.exp(-z)))
        quality = float(
            np.clip(self._separation / _CONFIDENCE_SEPARATION_REF, 0.0, 1.0)
        )
        return 0.5 + quality * (raw - 0.5)

    def _update_confidences(self) -> None:
        for slot_id, team in self._teams.items():
            feature = self._features.get(slot_id)
            if feature is None:
                continue
            raw = self._confidence_for(feature, team)
            previous = self._confidence.get(slot_id, raw)
            beta = self.confidence_ema_beta
            self._confidence[slot_id] = (1.0 - beta) * previous + beta * raw
