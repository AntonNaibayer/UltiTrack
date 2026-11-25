from collections import deque

import cv2
import numpy as np

from ultitrack.config import (
    ENDZONE_LENGTH,
    FIELD_LENGTH,
    FIELD_WIDTH,
    RADAR_SMOOTHING_BETA,
    RADAR_SMOOTHING_D_CUTOFF,
    RADAR_SMOOTHING_MIN_CUTOFF,
    RADAR_SSAA,
    RADAR_TRAIL_LEN,
    RADAR_USE_SMOOTHING,
    TEAM_COLORS_BGR,
    UNKNOWN_TEAM_COLOR_BGR,
)


class OneEuroFilter:
    """Изотропный адаптивный фильтр для сглаживания 2D-координат (без разъезда осей X/Y)."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: np.ndarray | None = None
        self.dx_prev: np.ndarray | None = None
        self.t_prev: float | None = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float32)
        if self.t_prev is None or self.x_prev is None or self.dx_prev is None:
            self.t_prev = t
            self.x_prev = x_arr.copy()
            self.dx_prev = np.zeros_like(x_arr, dtype=np.float32)
            return self.x_prev

        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev

        dx = (x_arr - self.x_prev) / dt
        a_d = self._alpha(dt, self.d_cutoff)
        dx_hat = self._exp_smooth(dx, self.dx_prev, a_d)

        # Скалярная скорость вместо покомпонентной — диагонали остаются прямыми
        speed = float(np.linalg.norm(dx_hat))
        cutoff = self.min_cutoff + self.beta * speed
        a = self._alpha(dt, cutoff)

        x_hat = self._exp_smooth(x_arr, self.x_prev, a)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

    @staticmethod
    def _alpha(dt: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    @staticmethod
    def _exp_smooth(x: np.ndarray, prev: np.ndarray, alpha: float) -> np.ndarray:
        return alpha * x + (1.0 - alpha) * prev


class TacticalRadar:
    def __init__(
        self,
        width: int = 180,
        height: int = 360,
        use_smoothing: bool = RADAR_USE_SMOOTHING,
    ) -> None:
        self.width = width
        self.height = height
        self.use_smoothing = use_smoothing

        self.ssaa = RADAR_SSAA
        self.w_px = width * self.ssaa
        self.h_px = height * self.ssaa

        # Используем размеры из конфига
        self.scale_x = self.w_px / FIELD_WIDTH
        self.scale_y = self.h_px / FIELD_LENGTH

        self.filters: dict[int, OneEuroFilter] = {}
        self.trails: dict[int, deque[tuple[int, int]]] = {}
        self.disc_filter: OneEuroFilter | None = None

        self.frame_time = 0.0

    def _draw_pitch(self) -> np.ndarray:
        pitch = np.full((self.h_px, self.w_px, 3), (45, 110, 65), dtype=np.uint8)
        cv2.rectangle(pitch, (0, 0), (self.w_px - 1, self.h_px - 1), (255, 255, 255), 2 * self.ssaa, cv2.LINE_AA)

        # Рисуем эндзоны и центр по конфигу
        far_ez_y = int(ENDZONE_LENGTH * self.scale_y)
        near_ez_y = int((FIELD_LENGTH - ENDZONE_LENGTH) * self.scale_y)
        cv2.line(pitch, (0, far_ez_y), (self.w_px, far_ez_y), (255, 255, 255), 2 * self.ssaa, cv2.LINE_AA)
        cv2.line(pitch, (0, near_ez_y), (self.w_px, near_ez_y), (255, 255, 255), 2 * self.ssaa, cv2.LINE_AA)

        mid_y = int((FIELD_LENGTH / 2.0) * self.scale_y)
        cv2.line(pitch, (0, mid_y), (self.w_px, mid_y), (200, 230, 200), 1 * self.ssaa, cv2.LINE_AA)
        return pitch

    def render(
        self,
        coords_2d: np.ndarray,
        tracker_ids: np.ndarray | None,
        radar_status: np.ndarray | None,
        player_teams: dict[int, str],
        disc_2d: np.ndarray | None = None,
        fps: float = 30.0,
    ) -> np.ndarray:
        radar = self._draw_pitch()
        active_ids: set[int] = set()

        self.frame_time += 1.0 / (fps if fps > 0 else 30.0)

        if tracker_ids is not None and len(coords_2d) > 0 and radar_status is not None:
            for raw_pos, tid, status in zip(coords_2d, tracker_ids, radar_status):
                tid_int = int(tid)
                active_ids.add(tid_int)

                if tid_int not in self.trails:
                    self.trails[tid_int] = deque(maxlen=RADAR_TRAIL_LEN)

                if self.use_smoothing:
                    if tid_int not in self.filters:
                        self.filters[tid_int] = OneEuroFilter(
                            min_cutoff=RADAR_SMOOTHING_MIN_CUTOFF,
                            beta=RADAR_SMOOTHING_BETA,
                            d_cutoff=RADAR_SMOOTHING_D_CUTOFF,
                        )
                    pos = self.filters[tid_int](raw_pos, self.frame_time)
                else:
                    pos = raw_pos

                x_m, y_m = float(pos[0]), float(pos[1])
                # int(round(...)) для суб-пиксельной плавности
                px = int(round(np.clip(x_m * self.scale_x, 6 * self.ssaa, self.w_px - 7 * self.ssaa)))
                py = int(round(np.clip(y_m * self.scale_y, 6 * self.ssaa, self.h_px - 7 * self.ssaa)))

                if status == 1:
                    self.trails[tid_int].append((px, py))
                    # Цвет фишки и хвоста берём из карты команд; пока классификатор
                    # не заполнил её, используется нейтральный оранжевый.
                    team = player_teams.get(tid_int)
                    color = TEAM_COLORS_BGR.get(team, UNKNOWN_TEAM_COLOR_BGR)
                    for i in range(1, len(self.trails[tid_int])):
                        cv2.line(
                            radar,
                            self.trails[tid_int][i - 1],
                            self.trails[tid_int][i],
                            color,
                            1 * self.ssaa,
                            cv2.LINE_AA,
                        )

                    cv2.circle(radar, (px, py), 6 * self.ssaa, color, -1, cv2.LINE_AA)
                    cv2.circle(radar, (px, py), 6 * self.ssaa, (255, 255, 255), 1 * self.ssaa, cv2.LINE_AA)
                else:
                    self.trails[tid_int].clear()
                    color = (150, 150, 150)
                    cv2.circle(radar, (px, py), 6 * self.ssaa, color, 1 * self.ssaa, cv2.LINE_AA)

                cv2.putText(
                    radar,
                    str(tid_int),
                    (px + 6 * self.ssaa, py - 4 * self.ssaa),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35 * self.ssaa,
                    (255, 255, 255),
                    1 * self.ssaa,
                    cv2.LINE_AA,
                )

        for old_tid in list(self.trails.keys()):
            if old_tid not in active_ids:
                del self.trails[old_tid]
                self.filters.pop(old_tid, None)

        if disc_2d is not None:
            if self.disc_filter is None:
                self.disc_filter = OneEuroFilter(min_cutoff=1.0, beta=0.1)
            smoothed_disc = self.disc_filter(disc_2d, self.frame_time)
            dx_m, dy_m = float(smoothed_disc[0]), float(smoothed_disc[1])
            dpx = int(round(np.clip(dx_m * self.scale_x, 4 * self.ssaa, self.w_px - 5 * self.ssaa)))
            dpy = int(round(np.clip(dy_m * self.scale_y, 4 * self.ssaa, self.h_px - 5 * self.ssaa)))
            cv2.circle(radar, (dpx, dpy), 5 * self.ssaa, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(radar, (dpx, dpy), 4 * self.ssaa, (0, 255, 255), -1, cv2.LINE_AA)

        final_radar = cv2.resize(radar, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return final_radar