"""CourtMapper: пиксели кадра -> метры на поле.

Что изменилось по сравнению с первой версией:
  * точки игроков и диска сначала "выпрямляются" моделью дисторсии (calib/lens.json),
    YOLO по-прежнему работает на исходном кадре;
  * калибровочные точки берутся из calib/court_points.json (любые точки на полу с известными
    координатами в метрах, не только углы поля), невидимые можно пропускать клавишей S;
  * после клика печатается ошибка в метрах (подгонка и leave-one-out), а метрическая сетка
    рисуется поверх кадра для визуальной проверки;
  * гомография сохраняется в calib/homography.json и при следующем запуске загружается сама
    (с возможностью проверить сетку и перекалибровать клавишей R);
  * добавлены строгая и буферная маски поля для гистерезис-фильтрации игроков на бровке;
  * добавлена поддержка сторон со стеной у линии (не считаются аутом, координата прижимается к линии).

Система координат: поле 20 x 40 м, x - поперёк (0..20), y - вдоль (0..40),
энд-зоны по 6 м (y = 0..6 и y = 34..40). Калибровочные точки могут лежать и вне поля.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from ultitrack.config import ENDZONE_LENGTH, FIELD_LENGTH, FIELD_WIDTH, ROOT_DIR
from ultitrack.lens import DivisionLens

CALIB_DIR = ROOT_DIR / "calib"
LENS_PATH = CALIB_DIR / "lens.json"
POINTS_PATH = CALIB_DIR / "court_points.json"
HOMOGRAPHY_PATH = CALIB_DIR / "homography.json"
FIELD_POLYGON_PATH = CALIB_DIR / "field_polygon.json"

FIELD_W, FIELD_L = FIELD_WIDTH, FIELD_LENGTH
ENDZONE = ENDZONE_LENGTH
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Стороны, где стена стоит прямо у линии (выход за линию — ошибка проекции, а не аут):
# left -> x = 0 (стена с окнами), right -> x = 20, far -> y = 0, near -> y = 40.
WALL_NO_OUT_MARGINS_M: dict[str, float] = {
    "left": 3.0,
    "right": 0.0,
    "far": 0.0,
    "near": 0.0,
}


class CourtMapper:
    def __init__(
        self,
        lens_path: Path = LENS_PATH,
        points_path: Path = POINTS_PATH,
        homography_path: Path = HOMOGRAPHY_PATH,
        wall_margins: dict[str, float] | None = None,
    ) -> None:
        self.points_path = Path(points_path)
        self.homography_path = Path(homography_path)
        self.wall_margins: dict[str, float] = {
            **WALL_NO_OUT_MARGINS_M,
            **(wall_margins or {}),
        }
        self.lens: DivisionLens | None = (
            DivisionLens.load(lens_path) if Path(lens_path).exists() else None
        )
        if self.lens is None:
            print(f"[CourtMapper] ВНИМАНИЕ: {lens_path} не найден, дисторсия не корректируется.")

        self.homography_matrix: np.ndarray | None = None  # исправленные пиксели -> метры
        self.h_inv: np.ndarray | None = None
        self.frame_size: tuple[int, int] | None = None  # (w, h)
        self.calib_points: list[dict] = []  # name, pixel, world
        self.field_polygon: np.ndarray | None = None  # (N, 2) пиксели исходного кадра
        self._load_field_polygon()

    # ------------------------------------------------------------------ геометрия

    def _undistort(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        return self.lens.undistort(pts) if self.lens is not None else pts

    def _clamp_wall_coords(self, coords: np.ndarray) -> np.ndarray:
        """Если проекция ушла за стену в пределах wall_margins, прижимает её к линии поля."""
        res = coords.copy()
        m_left = self.wall_margins["left"]
        m_right = self.wall_margins["right"]
        m_far = self.wall_margins["far"]
        m_near = self.wall_margins["near"]

        if m_left > 0.0:
            mask = (res[:, 0] < 0.0) & (res[:, 0] >= -m_left)
            res[mask, 0] = 0.0
        if m_right > 0.0:
            mask = (res[:, 0] > FIELD_W) & (res[:, 0] <= FIELD_W + m_right)
            res[mask, 0] = FIELD_W
        if m_far > 0.0:
            mask = (res[:, 1] < 0.0) & (res[:, 1] >= -m_far)
            res[mask, 1] = 0.0
        if m_near > 0.0:
            mask = (res[:, 1] > FIELD_L) & (res[:, 1] <= FIELD_L + m_near)
            res[mask, 1] = FIELD_L
        return res

    def world_to_raw_pixels(self, pts_world: np.ndarray) -> np.ndarray:
        """Метры -> пиксели ИСХОДНОГО (искажённого) кадра. Нужно для рисования сетки на кадре."""
        assert self.h_inv is not None
        pts = np.asarray(pts_world, dtype=np.float32).reshape(-1, 1, 2)
        und = cv2.perspectiveTransform(pts, self.h_inv).reshape(-1, 2).astype(np.float64)
        with np.errstate(all="ignore"):
            return self.lens.distort(und) if self.lens is not None else und

    def transform_to_2d(
        self,
        detections: sv.Detections,
        foot_offset_ratio: float = 0.01,
        clamp_walls: bool = True,
    ) -> np.ndarray:
        """Переводит нижнюю центральную точку боксов в метры на поле.
        foot_offset_ratio слегка приподнимает точку от нижнего края бокса (на 1% высоты),
        компенсируя толщину подошвы и перспективное наложение ступней без искажения дальней зоны."""
        if self.homography_matrix is None or len(detections) == 0:
            return np.empty((0, 2), dtype=np.float32)

        feet = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER).astype(np.float64)
        if foot_offset_ratio > 0.0:
            heights = detections.xyxy[:, 3] - detections.xyxy[:, 1]
            feet[:, 1] -= heights * foot_offset_ratio

        feet_u = self._undistort(feet).astype(np.float32)
        out = cv2.perspectiveTransform(feet_u.reshape(-1, 1, 2), self.homography_matrix).reshape(-1, 2)
        if clamp_walls:
            out = self._clamp_wall_coords(out)
        return out

    def transform_point_to_2d(
        self, x_px: float, y_px: float, ground_offset_px: float = 0.0
    ) -> np.ndarray | None:
        """Одна точка (например, диск) -> метры. Для диска в полёте проекция по полу неточна
        (диск над полом), поэтому для диска в руках лучше брать позицию держателя."""
        if self.homography_matrix is None:
            return None
        pt = self._undistort(np.array([[x_px, y_px + ground_offset_px]])).astype(np.float32)
        out = cv2.perspectiveTransform(pt.reshape(1, 1, 2), self.homography_matrix)
        return out[0, 0]

    def get_field_masks(
        self,
        detections: sv.Detections,
        strict_margin_m: float = 0.0,
        buffer_margin_m: float = 0.9,
        drop_cut_feet: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Возвращает две булевы маски:
        1. in_strict — игрок строго внутри разметки (плюс стороны со стеной из wall_margins).
        2. in_buffer — игрок внутри поля с допуском на выбегание в аут (для удержания трека)."""
        n = len(detections)
        if self.homography_matrix is None or n == 0:
            empty = np.zeros(n, dtype=bool)
            return empty, empty

        # Проверяем до прижатия к стене, чтобы далёкие выбросы за пределами wall_margins отсеивались
        coords = self.transform_to_2d(detections, clamp_walls=False)
        x, y = coords[:, 0], coords[:, 1]

        s_left = max(strict_margin_m, self.wall_margins["left"])
        s_right = max(strict_margin_m, self.wall_margins["right"])
        s_far = max(strict_margin_m, self.wall_margins["far"])
        s_near = max(strict_margin_m, self.wall_margins["near"])

        b_left = max(buffer_margin_m, self.wall_margins["left"])
        b_right = max(buffer_margin_m, self.wall_margins["right"])
        b_far = max(buffer_margin_m, self.wall_margins["far"])
        b_near = max(buffer_margin_m, self.wall_margins["near"])

        in_strict = (
            (x >= -s_left)
            & (x <= FIELD_W + s_right)
            & (y >= -s_far)
            & (y <= FIELD_L + s_near)
        )
        in_buffer = (
            (x >= -b_left)
            & (x <= FIELD_W + b_right)
            & (y >= -b_far)
            & (y <= FIELD_L + b_near)
        )

        if self.field_polygon is not None:
            feet = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            inside_px = np.array(
                [
                    cv2.pointPolygonTest(self.field_polygon, (float(fx), float(fy)), False) >= 0
                    for fx, fy in feet
                ],
                dtype=bool,
            )
            # Не даём пиксельному полигону отбрасывать игроков у стен из wall_margins
            in_wall_zone = np.zeros(n, dtype=bool)
            if self.wall_margins["left"] > 0.0:
                in_wall_zone |= (x >= -self.wall_margins["left"]) & (x <= 0.5)
            if self.wall_margins["right"] > 0.0:
                in_wall_zone |= (x >= FIELD_W - 0.5) & (x <= FIELD_W + self.wall_margins["right"])
            if self.wall_margins["far"] > 0.0:
                in_wall_zone |= (y >= -self.wall_margins["far"]) & (y <= 0.5)
            if self.wall_margins["near"] > 0.0:
                in_wall_zone |= (y >= FIELD_L - 0.5) & (y <= FIELD_L + self.wall_margins["near"])

            in_strict &= inside_px | in_wall_zone
            in_buffer &= inside_px | in_wall_zone

        if drop_cut_feet and self.frame_size is not None:
            not_cut = detections.xyxy[:, 3] < self.frame_size[1] - 3
            in_strict &= not_cut
            in_buffer &= not_cut

        return in_strict, in_buffer

    def filter_detections_in_field(
        self,
        detections: sv.Detections,
        margin_m: float = 0.8,
        drop_cut_feet: bool = True,
    ) -> sv.Detections:
        """Оставляет игроков внутри поля (+ отступ margin_m). Если у бокса нижний край упирается
        в низ кадра, ноги обрезаны, и координата на полу будет врать: такой бокс отбрасывается."""
        if self.homography_matrix is None or len(detections) == 0:
            return detections

        _, in_buffer = self.get_field_masks(
            detections,
            strict_margin_m=0.0,
            buffer_margin_m=margin_m,
            drop_cut_feet=drop_cut_feet,
        )
        return detections[in_buffer]

    # ------------------------------------------------------------------ калибровка

    def _load_field_polygon(self) -> None:
        if not FIELD_POLYGON_PATH.exists():
            return
        data = json.loads(FIELD_POLYGON_PATH.read_text())
        pts = np.asarray(data["points"], dtype=np.float32)
        if pts.shape[0] >= 4 and pts.ndim == 2 and pts.shape[1] == 2:
            self.field_polygon = pts
            print(f"[CourtMapper] Полигон поля загружен: {len(pts)} точек")

    def calibrate_field_polygon(self, frame: np.ndarray) -> None:
        """Клик по 4 углам красной разметки (по часовой стрелке). S — пропустить."""
        win = "UltiTrack: click 4 field corners (S - skip)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)
        clicked: list[tuple[int, int]] = []

        def cb(event: int, x: int, y: int, flags: int, param: None) -> None:
            if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < 4:
                clicked.append((x, y))

        cv2.setMouseCallback(win, cb)
        while True:
            canvas = frame.copy()
            for i, (px, py) in enumerate(clicked):
                cv2.circle(canvas, (px, py), 6, (0, 255, 0), -1)
                cv2.putText(canvas, str(i + 1), (px + 8, py - 8), FONT, 0.7, (0, 255, 0), 2)
            if len(clicked) >= 2:
                cv2.polylines(
                    canvas,
                    [np.array(clicked, np.int32).reshape(-1, 1, 2)],
                    len(clicked) >= 4,
                    (0, 255, 0),
                    2,
                )
            msg = f"Click corner {len(clicked) + 1}/4 (S - skip)"
            cv2.rectangle(canvas, (10, 10), (600, 50), (0, 0, 0), -1)
            cv2.putText(canvas, msg, (20, 37), FONT, 0.65, (255, 255, 255), 2)
            cv2.imshow(win, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("s"), ord("S")):
                cv2.destroyWindow(win)
                return
            if len(clicked) >= 4 and key in (13, 32):
                break

        cv2.destroyWindow(win)
        self.field_polygon = np.array(clicked, dtype=np.float32)
        FIELD_POLYGON_PATH.parent.mkdir(parents=True, exist_ok=True)
        FIELD_POLYGON_PATH.write_text(json.dumps({"points": clicked}, indent=2, ensure_ascii=False))
        print(f"[CourtMapper] Полигон сохранён: {FIELD_POLYGON_PATH}")

    def select_court_keypoints(self, frame: np.ndarray, force: bool = False) -> None:
        """Загружает сохранённую гомографию (и даёт проверить сетку) либо калибрует заново."""
        h, w = frame.shape[:2]
        self.frame_size = (w, h)
        if self.lens is None:
            self.lens = DivisionLens(w, h)  # нулевая дисторсия
        elif (self.lens.width, self.lens.height) != (w, h):
            print(f"[CourtMapper] ВНИМАНИЕ: lens.json для {self.lens.width}x{self.lens.height}, кадр {w}x{h}.")

        if not force and self._load_homography():
            if self._confirm_overlay(frame):
                return

        refs = self._load_reference_points()
        while True:
            clicked = self._click_points(frame, refs)
            idxs = sorted(clicked)
            names = [refs[i][0] for i in idxs]
            pixels = [clicked[i] for i in idxs]
            worlds = [refs[i][1] for i in idxs]

            H, fit_err, loo_err = self._fit_homography(pixels, worlds)
            self._report(names, fit_err, loo_err)

            self.homography_matrix = H
            self.h_inv = np.linalg.inv(H)
            self.calib_points = [
                {"name": n, "pixel": list(map(float, p)), "world": list(map(float, wd))}
                for n, p, wd in zip(names, pixels, worlds)
            ]
            if self._confirm_overlay(frame):
                break

        self._save_homography()

    def _load_reference_points(self) -> list[tuple[str, tuple[float, float]]]:
        data = json.loads(self.points_path.read_text())
        return [(p["name"], (float(p["xy"][0]), float(p["xy"][1]))) for p in data["points"]]

    def _fit_homography(
        self, pixels: list, worlds: list
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        src = self._undistort(np.array(pixels, dtype=np.float64)).astype(np.float32)
        dst = np.array(worlds, dtype=np.float32)

        H, _ = cv2.findHomography(src, dst, 0)
        if H is None:
            raise RuntimeError("Не удалось посчитать гомографию: проверьте точки (не на одной прямой?).")

        proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
        fit_err = np.linalg.norm(proj - dst, axis=1)

        # leave-one-out: ошибка точки, когда она НЕ участвовала в подгонке.
        # На малом числе точек именно она выдаёт неточный клик.
        loo_err = np.full(len(src), np.nan)
        if len(src) >= 6:
            for i in range(len(src)):
                keep = np.arange(len(src)) != i
                Hi, _ = cv2.findHomography(src[keep], dst[keep], 0)
                if Hi is None:
                    continue
                p = cv2.perspectiveTransform(src[i].reshape(1, 1, 2), Hi).reshape(2)
                loo_err[i] = np.linalg.norm(p - dst[i])
        return H, fit_err, loo_err

    @staticmethod
    def _report(names: list[str], fit_err: np.ndarray, loo_err: np.ndarray) -> None:
        print("\nТочка                                   | подгонка, м | leave-one-out, м")
        for n, f, l in zip(names, fit_err, loo_err):
            flag = "  <-- проверьте клик" if (np.isfinite(l) and l > 0.5) else ""
            print(f"{n[:38]:<38}  | {f:11.2f} | {l:16.2f}{flag}")
        print(f"Средняя ошибка подгонки: {fit_err.mean():.2f} м, максимум: {fit_err.max():.2f} м")
        if np.isfinite(loo_err).any():
            print(f"Leave-one-out: средняя {np.nanmean(loo_err):.2f} м, максимум {np.nanmax(loo_err):.2f} м")

    # ------------------------------------------------------------------ UI

    @staticmethod
    def _draw_magnifier(canvas: np.ndarray, frame: np.ndarray, mouse: tuple[int, int]) -> None:
        h, w = frame.shape[:2]
        half, zoom = 20, 5
        mx, my = mouse
        x1, y1 = max(0, mx - half), max(0, my - half)
        x2, y2 = min(w, mx + half + 1), min(h, my + half + 1)
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            return
        big = cv2.resize(patch, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
        bh, bw = big.shape[:2]
        cx, cy = (mx - x1) * zoom + zoom // 2, (my - y1) * zoom + zoom // 2
        cv2.line(big, (cx, 0), (cx, bh), (0, 0, 255), 1)
        cv2.line(big, (0, cy), (bw, cy), (0, 0, 255), 1)
        ox = w - bw - 10
        canvas[10 : 10 + bh, ox : ox + bw] = big

    def _click_points(
        self, frame: np.ndarray, refs: list[tuple[str, tuple[float, float]]]
    ) -> dict[int, tuple[int, int]]:
        h, w = frame.shape[:2]
        clicked: dict[int, tuple[int, int]] = {}
        st = {"i": 0, "mouse": (w // 2, h // 2)}
        win = "UltiTrack: click court points (S - skip, RMB - undo, ENTER - done)"

        def cb(event: int, x: int, y: int, flags: int, param: None) -> None:
            st["mouse"] = (x, y)
            if event == cv2.EVENT_LBUTTONDOWN and st["i"] < len(refs):
                clicked[st["i"]] = (x, y)
                st["i"] += 1
            elif event == cv2.EVENT_RBUTTONDOWN and st["i"] > 0:
                st["i"] -= 1
                clicked.pop(st["i"], None)

        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)
        cv2.setMouseCallback(win, cb)

        while True:
            canvas = frame.copy()
            for idx, (px, py) in clicked.items():
                cv2.circle(canvas, (px, py), 5, (0, 255, 0), -1)
                cv2.putText(canvas, str(idx + 1), (px + 8, py - 8), FONT, 0.7, (0, 255, 0), 2)
            self._draw_magnifier(canvas, frame, st["mouse"])

            if st["i"] < len(refs):
                name, (wx, wy) = refs[st["i"]]
                msg = f"[{st['i'] + 1}/{len(refs)}] {name} ({wx:.1f}, {wy:.1f}) m | S - skip"
            else:
                msg = f"All points passed. Clicked: {len(clicked)}. Press ENTER"
            cv2.rectangle(canvas, (10, 10), (1100, 50), (0, 0, 0), -1)
            cv2.putText(canvas, msg, (20, 37), FONT, 0.65, (255, 255, 255), 2)

            cv2.imshow(win, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("s"), ord("S")) and st["i"] < len(refs):
                st["i"] += 1
            elif key in (13, 32) and len(clicked) >= 6:
                break

        cv2.destroyWindow(win)
        return clicked

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Метрическая сетка 5x5 м поверх исходного кадра + кликнутые (зелёные) и
        предсказанные гомографией (красные) калибровочные точки."""
        canvas = frame.copy()

        def line(p0: tuple, p1: tuple, color: tuple, thickness: int) -> None:
            t = np.linspace(0.0, 1.0, 80)[:, None]
            pts = np.array(p0, dtype=np.float64) * (1 - t) + np.array(p1, dtype=np.float64) * t
            px = self.world_to_raw_pixels(pts)
            ok = np.isfinite(px).all(axis=1) & (np.abs(px) < 1e5).all(axis=1)
            px = px[ok]
            if len(px) >= 2:
                cv2.polylines(
                    canvas, [px.astype(np.int32).reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA
                )

        for x in range(0, int(FIELD_W) + 1, 5):
            line((x, 0), (x, FIELD_L), (0, 200, 0), 1)
        for y in range(0, int(FIELD_L) + 1, 5):
            line((0, y), (FIELD_W, y), (0, 200, 0), 1)
        for y in (ENDZONE, FIELD_L - ENDZONE):
            line((0, y), (FIELD_W, y), (255, 255, 0), 2)
        for a, b in [
            ((0, 0), (FIELD_W, 0)),
            ((FIELD_W, 0), (FIELD_W, FIELD_L)),
            ((FIELD_W, FIELD_L), (0, FIELD_L)),
            ((0, FIELD_L), (0, 0)),
        ]:
            line(a, b, (0, 255, 255), 2)

        for p in self.calib_points:
            cv2.circle(canvas, tuple(int(v) for v in p["pixel"]), 6, (0, 255, 0), 2)
            pred = self.world_to_raw_pixels(np.array([p["world"]]))[0]
            if np.isfinite(pred).all():
                cx, cy = int(pred[0]), int(pred[1])
                cv2.drawMarker(canvas, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
        return canvas

    def _confirm_overlay(self, frame: np.ndarray) -> bool:
        win = "Check grid: ENTER - accept, R - recalibrate"
        canvas = self.draw_overlay(frame)
        cv2.rectangle(canvas, (10, 10), (900, 50), (0, 0, 0), -1)
        cv2.putText(
            canvas,
            "Grid on the floor lines? ENTER - accept | R - recalibrate",
            (20, 37),
            FONT,
            0.65,
            (255, 255, 255),
            2,
        )
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)
        cv2.imshow(win, canvas)
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 32):
                cv2.destroyWindow(win)
                return True
            if key in (ord("r"), ord("R")):
                cv2.destroyWindow(win)
                return False

    # ------------------------------------------------------------------ сохранение

    def _save_homography(self) -> None:
        assert self.homography_matrix is not None and self.frame_size is not None
        self.homography_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "frame_size": list(self.frame_size),
            "homography": self.homography_matrix.tolist(),
            "points": self.calib_points,
        }
        self.homography_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"[CourtMapper] Гомография сохранена: {self.homography_path}")

    def _load_homography(self) -> bool:
        if not self.homography_path.exists():
            return False
        data = json.loads(self.homography_path.read_text())
        if tuple(data["frame_size"]) != self.frame_size:
            print("[CourtMapper] Размер кадра изменился, нужна новая калибровка.")
            return False
        self.homography_matrix = np.array(data["homography"], dtype=np.float64)
        self.h_inv = np.linalg.inv(self.homography_matrix)
        self.calib_points = data.get("points", [])
        return True