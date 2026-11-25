import cv2
import numpy as np
import supervision as sv
from trackers import ByteTrackTracker
from ultralytics import YOLO

from ultitrack.config import (
    CONFIDENCE_THRESHOLD,
    DISC_CONFIDENCE_THRESHOLD,
    DISC_CROP_CONFIDENCE,
    IMAGE_SIZE,
    MODEL_PATH,
    TARGET_CLASSES,
    TEAM_COLORS_BGR,
    UNKNOWN_TEAM_COLOR_BGR,
)
from ultitrack.court_mapper import FIELD_L, FIELD_W, CourtMapper
from ultitrack.particle_filter import DiscParticleFilter

DISC_FIELD_MARGIN_M = 1.5  # диск за полем дальше этого отступа принимается только рядом с ожидаемой позицией
DISC_INIT_MARGIN_M = 0.5   # при первичном поиске диска (фильтр ещё не знает, где он) отступ строже
DISC_NEAR_PREV_PX = 120.0  # "рядом с ожидаемой позицией" для вылета диска за линию


class PlayerTracker:
    def __init__(self) -> None:
        self.model = YOLO(str(MODEL_PATH))
        self.tracker = ByteTrackTracker(
            lost_track_buffer=150,
            frame_rate=30,
        )
        self.disc_pf = DiscParticleFilter(num_particles=400)
        self.crop_size: int = 400
        self.last_stats = {"yolo_players": 0, "tracked": 0}

    def _snap_disc_to_click(
        self, frame: np.ndarray, click_x: int, click_y: int
    ) -> tuple[int, int, int, int] | None:
        h, w, _ = frame.shape
        half = 200
        x1, y1 = max(0, click_x - half), max(0, click_y - half)
        x2, y2 = min(w, click_x + half), min(h, click_y + half)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        res = self.model(crop, imgsz=640, conf=0.05, classes=[0], verbose=False)[0]
        detections = sv.Detections.from_ultralytics(res)

        return self._select_best_disc_from_sv(
            detections,
            expected_pos=(float(click_x), float(click_y)),
            max_jump=120.0,
            offset=(x1, y1),
        )

    def select_initial_disc(
        self, frame: np.ndarray, initial_box: tuple[int, int, int, int, str] | None = None
    ) -> tuple[int, int, int, int] | None:
        win = "UltiTrack: Click on DISC (ENTER - confirm, S - skip)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)

        state: dict = {
            "pt": None,
            "snapped_box": initial_box[:4] if initial_box is not None else None,
        }

        if state["snapped_box"] is not None:
            bx1, by1, bx2, by2 = state["snapped_box"]
            state["pt"] = ((bx1 + bx2) // 2, (by1 + by2) // 2)

        def _cb(event: int, x: int, y: int, flags: int, param: None) -> None:
            if event == cv2.EVENT_LBUTTONDOWN:
                state["pt"] = (x, y)
                state["snapped_box"] = self._snap_disc_to_click(frame, x, y)

        cv2.setMouseCallback(win, _cb)

        while True:
            canvas = frame.copy()
            cv2.rectangle(canvas, (10, 10), (780, 48), (0, 0, 0), -1)
            cv2.putText(
                canvas,
                "Click DISC (Green box = YOLO locked | Yellow = manual) -> ENTER",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 255),
                2,
            )

            if state["snapped_box"] is not None:
                bx1, by1, bx2, by2 = state["snapped_box"]
                cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                cv2.putText(
                    canvas,
                    "YOLO LOCKED",
                    (bx1, max(20, by1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )
            elif state["pt"] is not None:
                cv2.circle(canvas, state["pt"], 7, (0, 255, 255), -1)
                cv2.circle(canvas, state["pt"], 18, (0, 255, 255), 2)

            cv2.imshow(win, canvas)
            key = cv2.waitKey(20) & 0xFF

            if key in (13, 32) and (state["snapped_box"] is not None or state["pt"] is not None):
                if state["snapped_box"] is not None:
                    bx1, by1, bx2, by2 = state["snapped_box"]
                    cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                else:
                    cx, cy = float(state["pt"][0]), float(state["pt"][1])

                self.disc_pf.initialize(cx, cy)
                cv2.destroyWindow(win)
                return state["snapped_box"]

            elif key in (ord("s"), ord("S")):
                cv2.destroyWindow(win)
                return None

    def detect_frame(
        self,
        frame: np.ndarray,
        is_first_frame: bool = False,
        court_mapper: CourtMapper | None = None,
    ) -> tuple[sv.Detections, tuple[int, int, int, int, str] | None]:
        # Ищем ТОЛЬКО игроков (classes=[1]) с порогом CONFIDENCE_THRESHOLD
        results = self.model(
            frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            classes=[1],
            verbose=False,
        )[0]

        detections = sv.Detections.from_ultralytics(results)

        if len(detections) == 0 or detections.class_id is None or detections.confidence is None:
            self.last_stats = {"yolo_players": 0, "tracked": 0}
            return sv.Detections.empty(), None

        player_mask = detections.confidence >= CONFIDENCE_THRESHOLD
        players = self._track_players(detections[player_mask], is_first_frame)
        self.last_stats = {"yolo_players": int(player_mask.sum()), "tracked": len(players)}

        # Диск отключен, возвращаем None
        return players, None

    def _track_players(self, players: sv.Detections, is_first_frame: bool) -> sv.Detections:
        if len(players) > 0:
            players = players.with_nms(threshold=0.75)

            if is_first_frame:
                tracked = players
                for _ in range(3):
                    tracked = self.tracker.update(players)
                players = tracked

                if players.tracker_id is None or np.any(players.tracker_id == -1):
                    players.tracker_id = np.arange(1, len(players) + 1, dtype=int)
            else:
                players = self.tracker.update(players)
                if players.tracker_id is not None:
                    players = players[players.tracker_id != -1]
                else:
                    return sv.Detections.empty()

        return players

    def _search_disc_in_crop(
        self,
        frame: np.ndarray,
        expected_pos: tuple[float, float],
        court_mapper: CourtMapper | None = None,
    ) -> tuple[int, int, int, int] | None:
        h, w, _ = frame.shape
        cx, cy = int(expected_pos[0]), int(expected_pos[1])
        half = self.crop_size // 2
        x1, y1 = max(0, cx - half), max(0, cy - half)
        x2, y2 = min(w, cx + half), min(h, cy + half)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        res_crop = self.model(
            crop, imgsz=640, conf=DISC_CROP_CONFIDENCE, classes=[0], verbose=False
        )[0]
        crop_detections = sv.Detections.from_ultralytics(res_crop)

        return self._select_best_disc_from_sv(
            crop_detections,
            expected_pos,
            max_jump=220.0,
            offset=(x1, y1),
            court_mapper=court_mapper,
        )

    def _select_best_disc_from_sv(
        self,
        detections: sv.Detections,
        expected_pos: tuple[float, float] | None,
        max_jump: float,
        offset: tuple[int, int] = (0, 0),
        court_mapper: CourtMapper | None = None,
    ) -> tuple[int, int, int, int] | None:
        if len(detections) == 0 or detections.confidence is None:
            return None

        ox, oy = offset
        best_box = None
        best_score = -1.0

        for xyxy, conf in zip(detections.xyxy, detections.confidence):
            bx1, by1, bx2, by2 = xyxy
            bw, bh = bx2 - bx1, by2 - by1
            if not (5 <= bw <= 110 and 4 <= bh <= 110):
                continue

            gx1, gy1, gx2, gy2 = int(bx1 + ox), int(by1 + oy), int(bx2 + ox), int(by2 + oy)
            cx, cy = (gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0

            # Отбрасываем запасные диски у судей и на скамейке за бровкой.
            # Если фильтр уже ведёт диск, детекцию вне поля принимаем только рядом с ожидаемой позицией
            # (настоящий вылет за линию), иначе это чужой диск или ложное срабатывание.
            if court_mapper is not None:
                pt_2d = court_mapper.transform_point_to_2d(cx, cy, ground_offset_px=28.0)
                if pt_2d is not None:
                    m = DISC_FIELD_MARGIN_M if expected_pos is not None else DISC_INIT_MARGIN_M
                    inside = -m <= pt_2d[0] <= FIELD_W + m and -m <= pt_2d[1] <= FIELD_L + m
                    if not inside:
                        near_prev = (
                            expected_pos is not None
                            and float(np.hypot(cx - expected_pos[0], cy - expected_pos[1])) < DISC_NEAR_PREV_PX
                        )
                        if not near_prev:
                            continue

            score = float(conf)
            if expected_pos is not None:
                dist = float(np.hypot(cx - expected_pos[0], cy - expected_pos[1]))
                if dist > max_jump:
                    continue
                score += max(0.0, (max_jump - dist) / (max_jump * 1.5))

            if score > best_score:
                best_score = score
                best_box = (gx1, gy1, gx2, gy2)

        return best_box

    def draw_annotations(
        self,
        frame: np.ndarray,
        players: sv.Detections,
        player_teams: dict[int, str],
        disc_box: tuple[int, int, int, int, str] | None = None,
    ) -> np.ndarray:
        annotated = frame.copy()

        if players.tracker_id is not None:
            for xyxy, tracker_id in zip(players.xyxy, players.tracker_id):
                x1, y1, x2, y2 = map(int, xyxy)
                tid = int(tracker_id)
                team = player_teams.get(tid)

                color = TEAM_COLORS_BGR.get(team, UNKNOWN_TEAM_COLOR_BGR)
                label = f"#{tid}" + (f" {team}" if team else "")

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        if disc_box is not None:
            for px, py in self.disc_pf.get_particles_xy()[::3]:
                cv2.circle(annotated, (int(px), int(py)), 1, (180, 255, 255), -1)

            dx1, dy1, dx2, dy2, mode = disc_box
            dcx, dcy = (dx1 + dx2) // 2, (dy1 + dy2) // 2

            if mode == "YOLO":
                disc_color = (0, 255, 255)
            elif mode == "MOTION":
                disc_color = (255, 255, 0)
            else:
                disc_color = (0, 165, 255)

            cv2.rectangle(annotated, (dx1, dy1), (dx2, dy2), disc_color, 2)
            cv2.circle(annotated, (dcx, dcy), 4, disc_color, -1)
            cv2.putText(
                annotated,
                f"DISC [{mode}]",
                (dx1, max(20, dy1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                disc_color,
                2,
            )

        return annotated