from __future__ import annotations

import json
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment

from ultitrack.config import (
    APPEARANCE_COST_WEIGHT,
    APPEARANCE_DIRECT_MAX_DIST,
    APPEARANCE_RADIUS_M,
    ENDZONE_LENGTH,
    FIELD_LENGTH,
    FIELD_WIDTH,
    PILE_RADIUS_M,
    RESPAWN_MAX_APPEARANCE_DIST,
    TEAM_COLORS_BGR,
)
from ultitrack.court_mapper import FIELD_L, FIELD_W, CourtMapper

# ---------- константы ----------
MAX_SEARCH_M = 3.0           # свежие потери ищем в радиусе 3м
MAX_SEARCH_LOST_M = 9.0      # если потерян >= 15 кадров, ищем шире (рывок/уход из кадра)
CROWD_RADIUS_M = 1.5         # перекрытие слотов — пропускаем прямой tracker_id
CROWD_IOU = 0.05             # в плотной паре слотов прямой матч режем уже при таком IoU
DIRECT_MATCH_MAX_IOU = 0.45  # обычный порог перекрытия для доверия tracker_id

MAX_PLAYER_SPEED_MPS = 9.5   # физический предел скорости игрока (м/с)
KALMAN_ACCEL_STD = 4.0       # ожидаемое ускорение/манёвр игрока (м/с^2)
RESPAWN_MAX_DIST_M = 8.0     # не подбираем «мёртвый» слот на игрока с другого конца поля

# Коды статуса слота (см. update) -> строки в telemetry v2.
TELEMETRY_STATUS = {1: "active", 0: "lost", 2: "coasting"}


@dataclass
class PlayerSlot:
    slot_id: int
    pos_2d: np.ndarray
    anchor_pos: np.ndarray
    fps: float = 30.0
    anchor_vel: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    pos_history: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=10))
    vel_2d: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    tracker_id: int | None = None
    frames_lost: int = 0
    near_edge: bool = False
    team: str = "A"
    team_conf: float = 0.0

    # --- Калман (состояние: [x, y, vx, vy] в метрах и м/с) ---
    kf: cv2.KalmanFilter = field(init=False)
    pred_pos: np.ndarray = field(init=False)
    pred_vel: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    prev_pos: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        dt = 1.0 / max(float(self.fps), 1.0)
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=np.float32,
        )
        self.kf.transitionMatrix = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        # Дискретная модель белого шума ускорения (DWNA)
        dt2, dt3, dt4 = dt**2, dt**3, dt**4
        var_a = KALMAN_ACCEL_STD**2
        self.kf.processNoiseCov = (
            np.array(
                [
                    [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
                    [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
                    [dt3 / 2.0, 0.0, dt2, 0.0],
                    [0.0, dt3 / 2.0, 0.0, dt2],
                ],
                dtype=np.float32,
            )
            * var_a
        )

        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.25
        self.kf.errorCovPost = np.diag([0.5, 0.5, 4.0, 4.0]).astype(np.float32)

        state = np.array([self.pos_2d[0], self.pos_2d[1], 0.0, 0.0], dtype=np.float32).reshape(4, 1)
        self.kf.statePost = state.copy()
        self.kf.statePre = state.copy()

        self.pred_pos = self.pos_2d.copy()
        self.prev_pos = self.pos_2d.copy()


class RosterTracker:
    def __init__(self, fps: float = 30.0, team_classifier: object | None = None) -> None:
        self.fps = fps if fps > 0 else 30.0
        self.slots: list[PlayerSlot] = []
        # Классификатор команд не участвует в матчинге: он получает уже
        # сматченные боксы и только заполняет slot.team / slot.team_conf.
        self.team_classifier = team_classifier

        self.telemetry_frames: list[dict] = []
        self.review_frames: list[dict] = []
        self.dropped: list[tuple[np.ndarray, str]] = []
        self.last_stats: dict[str, int] = {"in_field": 0, "matched": 0, "slots": 0, "spare": 0}
        self.frames_total: int = 0
        self.slot_matched: dict[int, int] = {}
        self.reject_counts: Counter = Counter()
        self.reject_by_slot: dict[int, Counter] = {}
        self.pile_coasts: int = 0
        # (slot_id, pos_2d, status_code) для ВСЕХ слотов, включая потерянные.
        self._last_telemetry: list[tuple[int, np.ndarray, int]] = []

    def initialize_roster(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        coords_2d: np.ndarray,
    ) -> None:
        self.slots.clear()
        self.telemetry_frames.clear()
        self.review_frames.clear()
        self._last_telemetry.clear()
        self.pile_coasts = 0
        if self.team_classifier is not None:
            try:
                self.team_classifier.reset()
            except Exception as exc:  # noqa: BLE001
                print(f"[Roster] Классификатор команд отключён: {exc}")
                self.team_classifier = None

        if detections.tracker_id is None or len(detections) == 0:
            return

        slot_num = 1
        for xyxy, tid_raw, pos in zip(detections.xyxy, detections.tracker_id, coords_2d):
            hist: deque[np.ndarray] = deque(maxlen=10)
            hist.append(pos.copy())

            self.slots.append(
                PlayerSlot(
                    slot_id=slot_num,
                    pos_2d=pos.copy(),
                    anchor_pos=pos.copy(),
                    fps=self.fps,
                    pos_history=hist,
                    tracker_id=int(tid_raw),
                )
            )
            slot_num += 1

        # Позиции первого кадра нужны классификатору для проверки стартовой
        # расстановки: 5 игроков на одной линии эндзоны, 5 — на другой.
        if self.team_classifier is not None:
            try:
                self.team_classifier.set_initial_positions(
                    {slot.slot_id: slot.pos_2d.copy() for slot in self.slots}
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[Roster] Классификатор команд отключён: {exc}")
                self.team_classifier = None

    @staticmethod
    def _compute_box_ious(boxes: np.ndarray) -> np.ndarray:
        n = len(boxes)
        if n == 0:
            return np.zeros((0, 0), dtype=np.float32)
        x1, y1 = boxes[:, 0][:, None], boxes[:, 1][:, None]
        x2, y2 = boxes[:, 2][:, None], boxes[:, 3][:, None]
        iw = np.maximum(0.0, np.minimum(x2, x2.T) - np.maximum(x1, x1.T))
        ih = np.maximum(0.0, np.minimum(y2, y2.T) - np.maximum(y1, y1.T))
        inter = iw * ih
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area[:, None] + area[None, :] - inter + 1e-6
        ious = inter / union
        np.fill_diagonal(ious, 0.0)
        return ious

    @staticmethod
    def _is_near_boundary(pos: np.ndarray, margin: float = 1.0) -> bool:
        return bool(
            pos[0] < margin
            or pos[0] > FIELD_W - margin
            or pos[1] < margin
            or pos[1] > FIELD_L - margin
        )

    @staticmethod
    def _compute_smooth_velocity(hist: deque[np.ndarray]) -> np.ndarray:
        if len(hist) < 2:
            return np.zeros(2, dtype=np.float32)
        pts = np.array(hist, dtype=np.float32)
        n = len(pts)
        if n >= 4:
            vel = (pts[-2:].mean(axis=0) - pts[:2].mean(axis=0)) / max(1, n - 2)
        else:
            vel = (pts[-1] - pts[0]) / max(1, n - 1)
        speed = float(np.linalg.norm(vel))
        if speed > 0.45:
            vel = vel * (0.45 / speed)
        return vel

    @staticmethod
    def _predict_slot(slot: PlayerSlot) -> np.ndarray:
        return slot.pred_pos.copy()

    @staticmethod
    def _max_search_radius(slot: PlayerSlot) -> float:
        if slot.frames_lost >= 15:
            # Постепенно расширяем поиск: после рывка игрок может оказаться
            # в 7-9 м, но мгновенно искать по всему полю нельзя.
            return min(MAX_SEARCH_LOST_M, 5.0 + (slot.frames_lost - 15) * 0.08)
        if slot.near_edge:
            return 6.0
        return min(3.0 + slot.frames_lost * 0.4, MAX_SEARCH_M)

    def _appearance_distance(
        self,
        slot_index: int,
        det_index: int,
        det_features: list[np.ndarray | None] | None,
    ) -> float | None:
        """Расстояние между цветом детекции и эталоном слота."""
        if det_features is None or self.team_classifier is None:
            return None
        if not (0 <= det_index < len(det_features)):
            return None
        feature = det_features[det_index]
        if feature is None:
            return None
        reference = self.team_classifier.reference_feature(
            self.slots[slot_index].slot_id
        )
        if reference is None:
            return None
        return float(np.linalg.norm(feature - reference))

    def _match_all(
        self,
        detections: sv.Detections,
        det_coords: np.ndarray,
        crowded_slots: set[int],
        max_iou_per_det: np.ndarray,
        in_strict: np.ndarray,
        appearance_slots: set[int] | None = None,
        det_features: list[np.ndarray | None] | None = None,
        lost_slots: set[int] | None = None,
    ) -> dict[int, int]:
        matched: dict[int, int] = {}
        used_dets: set[int] = set()
        appearance_slots = appearance_slots or set()
        lost_slots = lost_slots or set()

        # Шаг 1: прямой tracker_id
        tid_to_det = {int(tid): idx for idx, tid in enumerate(detections.tracker_id)}
        for s_idx, slot in enumerate(self.slots):
            if slot.tracker_id is None or slot.tracker_id not in tid_to_det:
                continue
            d_idx = tid_to_det[slot.tracker_id]
            if d_idx in used_dets:
                continue

            # В тесной паре слотов не доверяем ByteTrack ID, если цвет
            # детекции сильно не совпадает с эталоном слота: при пересечении
            # игроков ID могли поменяться.
            if s_idx in appearance_slots:
                app_dist = self._appearance_distance(s_idx, d_idx, det_features)
                if (
                    app_dist is not None
                    and app_dist > APPEARANCE_DIRECT_MAX_DIST
                ):
                    continue

            # В плотной паре слотов доверяем tracker_id только при почти нулевом
            # перекрытии; иначе отдаём решение венгерскому алгоритму.
            iou_limit = CROWD_IOU if s_idx in crowded_slots else DIRECT_MATCH_MAX_IOU
            if max_iou_per_det[d_idx] > iou_limit:
                continue

            pred = slot.pred_pos
            if float(np.linalg.norm(det_coords[d_idx] - pred)) <= 3.5:
                slot_in = (0 <= slot.anchor_pos[0] <= FIELD_W) and (0 <= slot.anchor_pos[1] <= FIELD_L)
                if slot_in and not in_strict[d_idx] and float(np.linalg.norm(det_coords[d_idx] - slot.anchor_pos)) > 2.0:
                    continue

                matched[s_idx] = d_idx
                used_dets.add(d_idx)

        # Шаг 2: венгерский алгоритм
        free_s = [s for s in range(len(self.slots)) if s not in matched]
        free_d = [d for d in range(len(detections)) if d not in used_dets]

        if free_s and free_d:
            INF = 1e6
            C = np.full((len(free_s), len(free_d)), INF, dtype=np.float32)

            for r, s_idx in enumerate(free_s):
                slot = self.slots[s_idx]
                max_r = self._max_search_radius(slot)
                anchor_speed = float(np.linalg.norm(slot.anchor_vel))
                dt_frames = float(slot.frames_lost + 1)
                slot_in = (0 <= slot.anchor_pos[0] <= FIELD_W) and (0 <= slot.anchor_pos[1] <= FIELD_L)

                for c, d_idx in enumerate(free_d):
                    dist_pred = float(np.linalg.norm(det_coords[d_idx] - slot.pred_pos))
                    dist_anchor = float(np.linalg.norm(det_coords[d_idx] - slot.anchor_pos))
                    dist = min(dist_pred, dist_anchor)

                    if dist > max_r:
                        continue

                    traj_pen = 0.0
                    disp = det_coords[d_idx] - slot.anchor_pos
                    disp_norm = float(np.linalg.norm(disp))
                    if anchor_speed > 0.04 and dt_frames <= 18.0 and disp_norm > 0.35:
                        actual_vel = disp / dt_frames
                        traj_pen = float(np.linalg.norm(actual_vel - slot.anchor_vel)) * 5.0

                    cost = dist * 14.0 + traj_pen
                    app_dist = (
                        self._appearance_distance(s_idx, d_idx, det_features)
                        if (s_idx in appearance_slots or s_idx in lost_slots)
                        else None
                    )
                    # После долгой потери не цепляем игрока с чужим цветом
                    # формы — иначе он получает не свою команду.
                    if (
                        s_idx in lost_slots
                        and app_dist is not None
                        and app_dist > RESPAWN_MAX_APPEARANCE_DIST
                    ):
                        continue
                    if s_idx in appearance_slots and app_dist is not None:
                        cost += app_dist * APPEARANCE_COST_WEIGHT

                    if not in_strict[d_idx]:
                        if slot_in:
                            cost += 10000.0
                        elif dist_anchor > 2.0:
                            cost += 10000.0

                    C[r, c] = cost

            row_ind, col_ind = linear_sum_assignment(C)
            for r, c in zip(row_ind, col_ind):
                if C[r, c] < INF * 0.5:
                    matched[free_s[r]] = free_d[c]

        return matched

    def update(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        court_mapper: CourtMapper,
    ) -> tuple[sv.Detections, np.ndarray, np.ndarray, np.ndarray]:
        if not self.slots:
            self._last_telemetry = []
            return (
                sv.Detections.empty(),
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=int),
                np.empty((0,), dtype=int),
            )

        self.dropped = []
        self.frames_total += 1
        dt = 1.0 / self.fps
        max_step = MAX_PLAYER_SPEED_MPS * dt

        # 0) ПРЕДСКАЗАНИЕ КАЛМАНА
        for slot in self.slots:
            if slot.frames_lost > 0:
                slot.kf.statePost[2:4] *= 0.85

            pred = slot.kf.predict()
            slot.pred_pos = pred[:2].flatten()
            slot.pred_vel = pred[2:4].flatten()
            slot.prev_pos = slot.pos_2d.copy()

        # 1) Фильтр поля
        in_strict = np.zeros(len(detections), dtype=bool)
        if len(detections) > 0:
            in_strict_raw, in_buffer = court_mapper.get_field_masks(
                detections,
                strict_margin_m=0.0,
                buffer_margin_m=0.9, # Вернули нормальный буфер
                drop_cut_feet=False,
            )
            in_buffer = np.asarray(in_buffer, dtype=bool)
            in_strict_raw = np.asarray(in_strict_raw, dtype=bool)

            for xyxy in detections.xyxy[~in_buffer]:
                self.dropped.append((xyxy.copy(), "OUT_OF_FIELD"))
            detections = detections[in_buffer]
            in_strict = in_strict_raw[in_buffer]

        self.last_stats = {"in_field": len(detections), "matched": 0, "slots": len(self.slots), "spare": 0}
        det_coords = court_mapper.transform_to_2d(detections) if len(detections) > 0 else np.empty((0, 2))

        # 2) Перекрытия и матчинг
        crowded_slots: set[int] = set()
        close_slots: set[int] = set()
        pile_slots: set[int] = set()
        lost_slots: set[int] = set()
        blocked_dets: set[int] = set()
        if len(detections) > 0 and detections.tracker_id is not None:
            ious = self._compute_box_ious(detections.xyxy)
            max_iou_per_det = ious.max(axis=1) if len(detections) > 1 else np.zeros(len(detections))
            for i in range(len(self.slots)):
                for j in range(i + 1, len(self.slots)):
                    dist = float(
                        np.linalg.norm(self.slots[i].pos_2d - self.slots[j].pos_2d)
                    )
                    if dist < CROWD_RADIUS_M:
                        crowded_slots.add(i)
                        crowded_slots.add(j)
                    if dist < APPEARANCE_RADIUS_M:
                        close_slots.add(i)
                        close_slots.add(j)
                    if dist < PILE_RADIUS_M:
                        pile_slots.add(i)
                        pile_slots.add(j)
            lost_slots = {
                s_idx
                for s_idx, slot in enumerate(self.slots)
                if slot.frames_lost >= 15
            }

            # В тесных парах и после долгой потери сравниваем цвет детекции
            # с эталоном слота. Это включается только после калибровки команд.
            det_features: list[np.ndarray | None] | None = None
            appearance_slots: set[int] = set()
            if (
                (close_slots or lost_slots)
                and self.team_classifier is not None
                and bool(getattr(self.team_classifier, "calibrated", False))
            ):
                try:
                    det_features = [
                        self.team_classifier.feature_for_box(frame, box)
                        for box in detections.xyxy
                    ]
                    if any(feature is not None for feature in det_features):
                        appearance_slots = {
                            s_idx
                            for s_idx in (close_slots | lost_slots)
                            if self.team_classifier.reference_feature(
                                self.slots[s_idx].slot_id
                            )
                            is not None
                        }
                    else:
                        det_features = None
                except Exception:
                    det_features = None

            matched_det_indices = self._match_all(
                detections,
                det_coords,
                crowded_slots,
                max_iou_per_det,
                in_strict,
                appearance_slots,
                det_features,
                lost_slots,
            )

            # «Куча»: если двум слотам пришли детекции с другими ByteTrack ID,
            # не даём им поменяться игроками. Лучше несколько кадров coast,
            # чем мгновенная смена игрока/команды.
            if pile_slots and matched_det_indices:
                filtered: dict[int, int] = {}
                for s_idx, d_idx in matched_det_indices.items():
                    slot = self.slots[s_idx]
                    det_tid = (
                        int(detections.tracker_id[d_idx])
                        if detections.tracker_id is not None
                        else None
                    )
                    if (
                        s_idx in pile_slots
                        and slot.tracker_id is not None
                        and det_tid != slot.tracker_id
                    ):
                        blocked_dets.add(d_idx)
                        self.pile_coasts += 1
                        continue
                    filtered[s_idx] = d_idx
                matched_det_indices = filtered
        else:
            matched_det_indices = {}

        # 3) Диагностика отказов: почему несопоставленные слоты не взяли
        # свободные детекции (иначе reject_counts в summarize_stages пуст).
        used_dets = set(matched_det_indices.values())
        spare_idx = [
            i
            for i in range(len(detections))
            if i not in used_dets and i not in blocked_dets
        ]
        for s_idx, slot in enumerate(self.slots):
            if s_idx not in matched_det_indices:
                self._record_reject_simple(slot, spare_idx, det_coords)

        # 4) RESPAWN
        dead_slot_idx = None
        for i, slot in enumerate(self.slots):
            if slot.frames_lost > 150:
                dead_slot_idx = i
                break

        if dead_slot_idx is not None and spare_idx:
            dead_slot = self.slots[dead_slot_idx]
            best_spare = None
            best_conf = 0.0
            for d_idx in spare_idx:
                if not in_strict[d_idx]:
                    continue
                # Respawn — это локальное восстановление, а не поиск любого
                # свободного игрока по всему полю. Иначе слот может забрать
                # игрока другой команды и «сменить команду» на ровном месте.
                dist = float(np.linalg.norm(det_coords[d_idx] - dead_slot.pred_pos))
                if dist > RESPAWN_MAX_DIST_M:
                    continue
                # Не отдаём слот игроку с явно чужим цветом формы: иначе
                # после возврата с поля игрок получит команду соседа.
                if self.team_classifier is not None and bool(
                    getattr(self.team_classifier, "calibrated", False)
                ):
                    reference = self.team_classifier.reference_feature(
                        dead_slot.slot_id
                    )
                    if reference is not None:
                        feature = self.team_classifier.feature_for_box(
                            frame, detections.xyxy[d_idx]
                        )
                        if (
                            feature is not None
                            and float(np.linalg.norm(feature - reference))
                            > RESPAWN_MAX_APPEARANCE_DIST
                        ):
                            continue
                conf = float(detections.confidence[d_idx]) if detections.confidence is not None else 1.0
                if conf > 0.4 and conf > best_conf:
                    best_conf = conf
                    best_spare = d_idx

            if best_spare is not None:
                d_idx = best_spare
                new_pos = det_coords[d_idx].copy()
                slot = dead_slot

                slot.tracker_id = int(detections.tracker_id[d_idx])
                slot.pos_2d = new_pos.copy()
                slot.prev_pos = new_pos.copy()
                slot.pred_pos = new_pos.copy()
                slot.anchor_pos = new_pos.copy()
                slot.frames_lost = 0
                slot.pos_history.clear()
                slot.pos_history.append(new_pos.copy())
                slot.vel_2d = np.zeros(2, dtype=np.float32)
                slot.anchor_vel = np.zeros(2, dtype=np.float32)

                state = np.array([new_pos[0], new_pos[1], 0.0, 0.0], dtype=np.float32).reshape(4, 1)
                slot.kf.statePost = state.copy()
                slot.kf.statePre = state.copy()
                slot.kf.errorCovPost = np.diag([0.5, 0.5, 4.0, 4.0]).astype(np.float32)
                slot.kf.errorCovPre = slot.kf.errorCovPost.copy()

                matched_det_indices[dead_slot_idx] = d_idx
                spare_idx.remove(d_idx)
                print(f"[Roster] RESPAWN: Слот #{slot.slot_id} переназначен на игрока ID={slot.tracker_id}")

        # 5) Обновление позиций (без фикса ног, как договорились)
        kept_det_indices: list[int] = []
        matched_slot_ids: list[int] = []
        assigned_slot_ids: list[int] = []
        smoothed_coords: list[np.ndarray] = []
        status_array: list[int] = []
        self._last_telemetry = []

        for s_idx, slot in enumerate(self.slots):
            if s_idx in matched_det_indices:
                d_idx = matched_det_indices[s_idx]
                new_pos = det_coords[d_idx].copy()

                if slot.frames_lost > 0:
                    slot.pos_history.clear()
                    slot.pos_history.append(slot.anchor_pos.copy())

                slot.pos_history.append(new_pos)
                smooth_vel = self._compute_smooth_velocity(slot.pos_history)

                box_h = max(20.0, float(detections.xyxy[d_idx][3] - detections.xyxy[d_idx][1]))
                dist_scale = float(np.clip((120.0 / box_h) ** 2, 0.5, 4.0))
                slot.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * (0.20 * dist_scale)

                measurement = np.array([[new_pos[0]], [new_pos[1]]], dtype=np.float32)
                was_lost_long = slot.frames_lost >= 15
                slot.kf.correct(measurement)

                slot.pos_2d = slot.kf.statePost[:2].flatten()

                if was_lost_long:
                    # После долгой потери/рывка не растягиваем догон на
                    # десятки кадров: сразу ставим слот на найденного игрока.
                    slot.pos_2d = new_pos.copy()
                    slot.kf.statePost[:2] = new_pos.reshape(2, 1)
                    slot.kf.statePost[2:4] = 0.0
                else:
                    # Лимит шага за кадр
                    step_vec = slot.pos_2d - slot.prev_pos
                    step_dist = float(np.linalg.norm(step_vec))
                    if step_dist > max_step:
                        slot.pos_2d = slot.prev_pos + (step_vec / step_dist) * max_step
                        slot.kf.statePost[:2] = slot.pos_2d.reshape(2, 1)

                # Ограничение скорости
                vel = slot.kf.statePost[2:4].flatten()
                vel_norm = float(np.linalg.norm(vel))
                if vel_norm > MAX_PLAYER_SPEED_MPS:
                    vel = vel * (MAX_PLAYER_SPEED_MPS / vel_norm)
                    slot.kf.statePost[2:4] = vel.reshape(2, 1)
                slot.vel_2d = vel.copy()

                slot.anchor_pos = new_pos.copy()
                slot.anchor_vel = smooth_vel.copy()
                slot.tracker_id = int(detections.tracker_id[d_idx])
                slot.frames_lost = 0
                slot.near_edge = self._is_near_boundary(new_pos, margin=1.0)

                self.slot_matched[slot.slot_id] = self.slot_matched.get(slot.slot_id, 0) + 1
                kept_det_indices.append(d_idx)
                matched_slot_ids.append(slot.slot_id)
                assigned_slot_ids.append(slot.slot_id)
                smoothed_coords.append(slot.pos_2d.copy())
                status_array.append(1)
                self._last_telemetry.append((slot.slot_id, slot.pos_2d.copy(), 1))
            else:
                slot.frames_lost += 1
                slot.kf.statePost = slot.kf.statePre.copy()
                slot.kf.errorCovPost = slot.kf.errorCovPre.copy()

                slot.pos_2d = slot.pred_pos.copy()
                slot.vel_2d = slot.pred_vel.copy()

                smoothed_coords.append(slot.pos_2d.copy())
                assigned_slot_ids.append(slot.slot_id)
                status_code = 2 if slot.frames_lost > 30 else 0
                status_array.append(status_code)
                self._last_telemetry.append((slot.slot_id, slot.pos_2d.copy(), status_code))

                if slot.frames_lost > 60:
                    slot.tracker_id = None
                    slot.pos_history.clear()

        kept_set = set(kept_det_indices)
        for d_idx in range(len(detections)):
            if d_idx not in kept_set:
                self.dropped.append((detections.xyxy[d_idx].copy(), "NO_SLOT"))

        self.last_stats["matched"] = len(kept_det_indices)
        self.last_stats["spare"] = len(detections) - len(kept_det_indices)

        order = np.array(kept_det_indices, dtype=int) if kept_det_indices else np.array([], dtype=int)
        out_detections = detections[order] if len(order) > 0 else sv.Detections.empty()
        if len(out_detections) > 0:
            out_detections.tracker_id = np.array(matched_slot_ids, dtype=int)

        # Классификация команд — строго после матчинга и обновления позиций.
        # Ошибка классификатора не должна ломать трекинг: отключаем его и
        # продолжаем с нейтральными цветами.
        if self.team_classifier is not None:
            try:
                slot_boxes = {
                    self.slots[s_idx].slot_id: detections.xyxy[d_idx].copy()
                    for s_idx, d_idx in matched_det_indices.items()
                }
                assignments = self.team_classifier.update(frame, slot_boxes)
                for slot in self.slots:
                    assignment = assignments.get(slot.slot_id)
                    if assignment is not None:
                        slot.team = assignment.team
                        slot.team_conf = float(assignment.confidence)
            except Exception as exc:  # noqa: BLE001
                print(f"[Roster] Классификатор команд отключён: {exc}")
                self.team_classifier = None

        return (
            out_detections,
            np.array(smoothed_coords, dtype=np.float32),
            np.array(assigned_slot_ids, dtype=int),
            np.array(status_array, dtype=int),
        )

    def _record_reject_simple(self, slot, spare_idx, det_coords):
        pred = self._predict_slot(slot)
        max_r = self._max_search_radius(slot)

        reason = "NO_NEARBY_DET"
        best: tuple[float, int] | None = None
        for d in spare_idx:
            dist = float(np.linalg.norm(det_coords[d] - slot.anchor_pos))
            if best is None or dist < best[0]:
                best = (dist, d)

        if best is not None and best[0] <= 8.0:
            d = best[1]
            dist_pred = float(np.linalg.norm(det_coords[d] - pred))
            if dist_pred > max_r:
                reason = "OUT_OF_SEARCH"
            else:
                reason = "OTHER"

        self.reject_counts[reason] += 1
        self.reject_by_slot.setdefault(slot.slot_id, Counter())[reason] += 1

    def get_team_map(self) -> dict[int, str]:
        """slot_id -> "A"/"B".

        Пока классификатор команд не подтвердил цвет (team_conf > 0), слот не
        попадает в карту: радар и аннотации используют нейтральный цвет.
        """
        return {
            slot.slot_id: slot.team
            for slot in self.slots
            if slot.team_conf > 0.0
        }

    def record_telemetry(
        self,
        frame_idx,
        players,
        coords_2d,
        disc_2d,
        disc_box,
        holder_slot_id: int | None = None,
    ):
        """Пишет ВСЕ слоты, включая потерянные (status lost/coasting).

        Раньше сохранялись только слоты, сматченные на текущем кадре, поэтому
        при перекрытии игрок исчезал из JSON и фишка мигала в 2D-повторе.
        """
        team_map = self.get_team_map()
        if self._last_telemetry:
            snapshot = self._last_telemetry
        else:
            # Fallback, если record_telemetry вызван до update().
            snapshot = [
                (
                    slot.slot_id,
                    slot.pos_2d,
                    1 if slot.frames_lost == 0 else (2 if slot.frames_lost > 30 else 0),
                )
                for slot in self.slots
            ]
        slots_by_id = {slot.slot_id: slot for slot in self.slots}

        players_list = []
        for slot_id, pos, status_code in snapshot:
            slot = slots_by_id.get(slot_id)
            if slot is None:
                continue
            # team_map содержит только слоты с подтверждённой командой
            # (team_conf > 0). До калибровки пишем null, а не ложную "A".
            players_list.append(
                {
                    "id": slot_id,
                    "team": team_map.get(slot_id),
                    "team_conf": round(float(slot.team_conf), 2),
                    "x": round(float(pos[0]), 2),
                    "y": round(float(pos[1]), 2),
                    "vx": round(float(slot.vel_2d[0]), 2),
                    "vy": round(float(slot.vel_2d[1]), 2),
                    "status": TELEMETRY_STATUS.get(status_code, "unknown"),
                    "has_disc": holder_slot_id == slot_id,
                }
            )
        disc_data = None
        if disc_2d is not None and disc_box is not None:
            disc_data = {
                "x": round(float(disc_2d[0]), 2),
                "y": round(float(disc_2d[1]), 2),
                "state": disc_box[4],
                "holder_slot_id": holder_slot_id,
            }
        self.telemetry_frames.append(
            {
                "frame": frame_idx,
                "t": round(frame_idx / self.fps, 3),
                "players": players_list,
                "disc": disc_data,
            }
        )

    def save_telemetry(self, output_path: Path, point_id: int = 1) -> None:
        payload = {
            "meta": {
                "schema_version": 2,
                "point_id": point_id,
                "fps": round(self.fps, 2),
                "field": {
                    "width": FIELD_WIDTH,
                    "length": FIELD_LENGTH,
                    "endzone": ENDZONE_LENGTH,
                },
                "teams": {
                    "A": {"color_bgr": list(TEAM_COLORS_BGR["A"]), "attack_dir": None},
                    "B": {"color_bgr": list(TEAM_COLORS_BGR["B"]), "attack_dir": None},
                },
            },
            "frames": self.telemetry_frames,
            # События появятся в analytics/events.py (Фаза 3); пустой список
            # сохраняет форму telemetry v2 уже сейчас.
            "events": [],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        print(f"[Telemetry] Данные поинта сохранены: {output_path} ({len(self.telemetry_frames)} кадров)")

    def record_review_frame(self, frame_idx, raw_players, active_players_count, disc_box, expected_players_count):
        reasons: list[str] = []
        boxes_list: list[dict] = []
        if len(raw_players) > 0:
            confs = raw_players.confidence if raw_players.confidence is not None else [1.0] * len(raw_players)
            for xyxy, conf in zip(raw_players.xyxy, confs):
                boxes_list.append({"cls": 1, "xyxy": [round(float(v), 1) for v in xyxy], "conf": round(float(conf), 2)})
        if active_players_count < expected_players_count:
            reasons.append(f"МАЛО ИГРОКОВ НА ПОЛЕ ({active_players_count}/{expected_players_count})")
        if disc_box is not None:
            dx1, dy1, dx2, dy2, mode = disc_box
            boxes_list.append({"cls": 0, "xyxy": [float(dx1), float(dy1), float(dx2), float(dy2)], "mode": mode})
            if mode != "YOLO" and not mode.startswith("HELD"):
                reasons.append(f"ДИСК ПОТЕРЯН ({mode})")
        else:
            reasons.append("НЕТ ДИСКА")
        self.review_frames.append({"frame": frame_idx, "suspect": len(reasons) > 0, "reasons": reasons, "boxes": boxes_list})

    def save_review_data(self, output_path: Path, video_filename: str) -> None:
        payload = {"video": video_filename, "total_frames": len(self.review_frames), "frames": self.review_frames}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        suspect_count = sum(1 for f in self.review_frames if f["suspect"])
        print(f"[Review] Лог для проверки сохранён: {output_path} (подозрительных кадров: {suspect_count})")


def summarize_stages(stage_log: list[dict], n_slots: int, roster: "RosterTracker | None" = None) -> None:
    total = len(stage_log)
    hist = Counter(int(st.get("matched", 0)) for st in stage_log)
    print(f"\n[Stages] Кадров: {total}, слотов: {n_slots}")
    print("  Сколько слотов сопоставлено на кадре -> число кадров:")
    for k in sorted(hist):
        print(f"    {k:>2}: {hist[k]}")
    incomplete = [st for st in stage_log if st.get("matched", 0) < n_slots]
    with_candidate = sum(1 for st in incomplete if st.get("spare", 0) > 0)
    print(f"  Неполный состав: {len(incomplete)} кадров. Из них:")
    print(f"    есть НЕИСПОЛЬЗОВАННЫЕ детекции: {with_candidate}")
    print(f"    неиспользованных нет: {len(incomplete) - with_candidate}")
    if roster is not None and roster.frames_total:
        print("  Доля кадров, где слот сопоставлен:")
        for s in roster.slots:
            rate = roster.slot_matched.get(s.slot_id, 0) / roster.frames_total
            print(f"    #{s.slot_id} (команда {s.team}): {rate:.0%}")
        total_rej = sum(roster.reject_counts.values())
        if total_rej:
            print("  Почему слот не взял детекцию:")
            for reason, n in roster.reject_counts.most_common():
                print(f"    {reason:<20} {n:>6}  ({n / total_rej:.0%})")
        if roster.pile_coasts:
            print(
                "  Pile ID-change coast: "
                f"{roster.pile_coasts} кадров (смена tracker_id в куче отложена)"
            )