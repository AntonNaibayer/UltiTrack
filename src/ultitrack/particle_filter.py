import cv2
import numpy as np
import supervision as sv


class DiscParticleFilter:
    """
    Контекстный Particle Filter для алтимат фрисби:
    Объединяет 3 сигнала:
    1. Редкие детекции YOLO (класс 0).
    2. Межкадровое движение (Motion blob) вне боксов игроков (когда диск смазан в полете).
    3. Притяжение к игроку-владельцу (когда диск пойман/перекрыт на десятки кадров).
    """

    def __init__(self, num_particles: int = 400, use_motion: bool = False) -> None:
        self.num_particles = num_particles
        # Режим MOTION (разность кадров) выключен: на паркете он ловит блики и руки, а не диск
        self.use_motion = use_motion
        self.particles = np.empty((0, 4), dtype=np.float32)  # [x, y, vx, vy]
        self.weights = np.empty(0, dtype=np.float32)
        self.initialized = False

        self.prev_gray: np.ndarray | None = None
        self.holder_id: int | None = None
        self.frames_without_obs: int = 0
        self.mode: str = "SEARCH"  # "YOLO", "MOTION", "HELD", "COAST"
        self.last_wh: tuple[int, int] = (26, 20)

    def initialize(self, x: float, y: float, w: int = 26, h: int = 20) -> None:
        self.particles = np.zeros((self.num_particles, 4), dtype=np.float32)
        self.particles[:, 0] = x + np.random.normal(0, 10.0, self.num_particles)
        self.particles[:, 1] = y + np.random.normal(0, 10.0, self.num_particles)
        self.particles[:, 2] = np.random.normal(0, 2.0, self.num_particles)
        self.particles[:, 3] = np.random.normal(0, 2.0, self.num_particles)
        self.weights = np.full(self.num_particles, 1.0 / self.num_particles, dtype=np.float32)
        self.initialized = True
        self.frames_without_obs = 0
        self.last_wh = (w, h)

    def step(
        self,
        frame: np.ndarray,
        players: sv.Detections,
        yolo_box: tuple[int, int, int, int] | None,
    ) -> tuple[int, int, int, int, str] | None:
        h, w, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Собираем словарь текущих позиций рук/корпуса игроков: {tracker_id: (cx, cy)}
        player_hands: dict[int, tuple[float, float]] = {}
        if players.tracker_id is not None and len(players) > 0:
            for xyxy, tid in zip(players.xyxy, players.tracker_id):
                px1, py1, px2, py2 = xyxy
                # Диск обычно держат на уровне груди/пояса (40% высоты бокса сверху)
                player_hands[int(tid)] = (
                    float((px1 + px2) / 2.0),
                    float(py1 + (py2 - py1) * 0.42),
                )

        # Если фильтр ещё ни разу не видел диск и YOLO молчит — привязываем старт к первому игроку с диском или ждем
        if not self.initialized:
            if yolo_box is not None:
                bx1, by1, bx2, by2 = yolo_box
                self.initialize((bx1 + bx2) / 2.0, (by1 + by2) / 2.0, bx2 - bx1, by2 - by1)
                self.mode = "YOLO"
                self.prev_gray = gray_blur
                return self._build_output()
            self.prev_gray = gray_blur
            return None

        # 1. ПРЕДСКАЗАНИЕ (Двигаем частицы)
        self._predict_particles(w, h, player_hands)

        est_x, est_y = self.estimate()

        # 2. ПОИСК ДВИЖЕНИЯ ДИСКА (если YOLO промолчала на этом кадре)
        motion_pt = None
        if self.use_motion and yolo_box is None and self.prev_gray is not None:
            motion_pt = self._detect_fast_disc_motion(
                gray_blur, self.prev_gray, players, (est_x, est_y)
            )
        self.prev_gray = gray_blur

        # 3. ОБНОВЛЕНИЕ ВЕСОВ ЧАСТИЦ (Update)
        if yolo_box is not None:
            # Есть прямой сигнал от YOLO!
            bx1, by1, bx2, by2 = yolo_box
            mx, my = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
            self.last_wh = (max(18, bx2 - bx1), max(14, by2 - by1))
            self._update_with_point(mx, my, sigma=28.0)
            self.frames_without_obs = 0
            self.mode = "YOLO"
            # Проверяем, не в руках ли у игрока этот диск
            self._check_and_assign_holder(mx, my, player_hands, max_dist=55.0)

        elif motion_pt is not None:
            # YOLO не увидела (смазан), но детектор движения засек летящий объект вне игроков!
            mx, my = motion_pt
            self._update_with_point(mx, my, sigma=35.0)
            self.frames_without_obs = 0
            self.holder_id = None  # Диск точно летит в воздухе
            self.mode = "MOTION"

        else:
            # Ни YOLO, ни свободного полёта нет -> диск либо заслонён, либо в руках у игрока
            self.frames_without_obs += 1
            cur_x, cur_y = self.estimate()

            # Если мы ещё не привязаны к игроку — ищем ближайшего игрока в радиусе 85 пикселей, который мог поймать диск
            if self.holder_id is None or self.holder_id not in player_hands:
                self._check_and_assign_holder(cur_x, cur_y, player_hands, max_dist=85.0)

            if self.holder_id is not None and self.holder_id in player_hands:
                hx, hy = player_hands[self.holder_id]
                self._update_with_point(hx, hy, sigma=25.0)
                # Гасим скорость частиц, так как диск в руках у игрока
                self.particles[:, 2] *= 0.2
                self.particles[:, 3] *= 0.2
                self.mode = f"HELD #{self.holder_id}"
            else:
                self.mode = "PF COAST"

        return self._build_output()

    def _predict_particles(
        self,
        frame_w: int,
        frame_h: int,
        player_hands: dict[int, tuple[float, float]],
    ) -> None:
        # Если диск удерживается конкретным игроком, 50% частиц следуют за этим игроком,
        # а другие 50% разлетаются вокруг него в ожидании броска (паса)
        if self.holder_id is not None and self.holder_id in player_hands:
            hx, hy = player_hands[self.holder_id]
            half_n = self.num_particles // 2
            # Первая половина плотно держит руки игрока
            self.particles[:half_n, 0] = hx + np.random.normal(0, 12.0, half_n)
            self.particles[:half_n, 1] = hy + np.random.normal(0, 12.0, half_n)
            # Вторая половина готова к вылету диска в любую сторону
            self.particles[half_n:, 0] += self.particles[half_n:, 2] + np.random.normal(
                0, 22.0, self.num_particles - half_n
            )
            self.particles[half_n:, 1] += self.particles[half_n:, 3] + np.random.normal(
                0, 22.0, self.num_particles - half_n
            )
        else:
            # Свободный полет по инерции
            pos_noise = 14.0 if self.frames_without_obs < 10 else 22.0
            self.particles[:, 0] += self.particles[:, 2] + np.random.normal(
                0, pos_noise, self.num_particles
            )
            self.particles[:, 1] += self.particles[:, 3] + np.random.normal(
                0, pos_noise, self.num_particles
            )
            self.particles[:, 2] = 0.94 * self.particles[:, 2] + np.random.normal(
                0, 4.5, self.num_particles
            )
            self.particles[:, 3] = 0.94 * self.particles[:, 3] + np.random.normal(
                0, 4.5, self.num_particles
            )

        self.particles[:, 0] = np.clip(self.particles[:, 0], 0, frame_w - 1)
        self.particles[:, 1] = np.clip(self.particles[:, 1], 0, frame_h - 1)

    def _detect_fast_disc_motion(
        self,
        gray: np.ndarray,
        prev_gray: np.ndarray,
        players: sv.Detections,
        expected_pos: tuple[float, float],
    ) -> tuple[float, float] | None:
        """
        Ищет маленькое быстро движущееся пятно (диск в полете) путем вычитания кадров,
        полностью игнорируя области внутри bounding box'ов всех игроков.
        """
        diff = cv2.absdiff(gray, prev_gray)
        _, thresh = cv2.threshold(diff, 22, 255, cv2.THRESH_BINARY)

        # Закрашиваем черным всех игроков (с запасом 10 пикселей), чтобы их бег не мешал
        if len(players) > 0:
            for xyxy in players.xyxy:
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(
                    thresh,
                    (max(0, x1 - 10), max(0, y1 - 10)),
                    (min(gray.shape[1], x2 + 10), min(gray.shape[0], y2 + 10)),
                    0,
                    -1,
                )

        # Ищем только в радиусе 260 пикселей вокруг текущего ожидания фильтра частиц
        ex, ey = int(expected_pos[0]), int(expected_pos[1])
        r = 260
        rx1, ry1 = max(0, ex - r), max(0, ey - r)
        rx2, ry2 = min(gray.shape[1], ex + r), min(gray.shape[0], ey + r)
        roi = thresh[ry1:ry2, rx1:rx2]

        if roi.size == 0:
            return None

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)
        roi = cv2.dilate(roi, kernel, iterations=2)

        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_pt = None
        min_dist = float("inf")

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Площадь летящего диска на 1080p: от 18 до 900 кв. пикселей
            if 18 <= area <= 900:
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect = max(bw, bh) / max(1, min(bw, bh))
                if aspect > 4.5:  # Слишком длинная линия — блик на паркете
                    continue
                cx = float(rx1 + x + bw / 2.0)
                cy = float(ry1 + y + bh / 2.0)
                dist = float(np.hypot(cx - expected_pos[0], cy - expected_pos[1]))
                if dist < min_dist and dist < 190.0:
                    min_dist = dist
                    best_pt = (cx, cy)

        return best_pt

    def _update_with_point(self, mx: float, my: float, sigma: float) -> None:
        old_x, old_y = self.estimate()
        dx = self.particles[:, 0] - mx
        dy = self.particles[:, 1] - my
        dist_sq = dx**2 + dy**2

        # Если новая детекция далеко от старого роя (например, YOLO вдруг нашла диск на другом конце поля)
        if np.min(np.sqrt(dist_sq)) > 180.0:
            self.initialize(mx, my, self.last_wh[0], self.last_wh[1])
            return

        likelihood = np.exp(-dist_sq / (2.0 * sigma**2)) + 1e-12
        self.weights *= likelihood
        self.weights /= np.sum(self.weights)

        if 1.0 / np.sum(self.weights**2) < self.num_particles * 0.55:
            self._resample()

        # Обновляем вектор скорости частиц по реальному смещению центра
        vx_meas = mx - old_x
        vy_meas = my - old_y
        if np.hypot(vx_meas, vy_meas) < 90.0:
            self.particles[:, 2] = 0.65 * vx_meas + 0.35 * self.particles[:, 2]
            self.particles[:, 3] = 0.65 * vy_meas + 0.35 * self.particles[:, 3]

    def _check_and_assign_holder(
        self,
        disc_x: float,
        disc_y: float,
        player_hands: dict[int, tuple[float, float]],
        max_dist: float,
    ) -> None:
        best_tid = None
        best_d = max_dist
        for tid, (hx, hy) in player_hands.items():
            d = float(np.hypot(disc_x - hx, disc_y - hy))
            if d < best_d:
                best_d = d
                best_tid = tid
        self.holder_id = best_tid

    def estimate(self) -> tuple[float, float]:
        est_x = float(np.sum(self.particles[:, 0] * self.weights))
        est_y = float(np.sum(self.particles[:, 1] * self.weights))
        return est_x, est_y

    def get_particles_xy(self) -> np.ndarray:
        if not self.initialized:
            return np.empty((0, 2), dtype=np.int32)
        return self.particles[:, :2].astype(np.int32)

    def _build_output(self) -> tuple[int, int, int, int, str]:
        cx, cy = self.estimate()
        w, h = self.last_wh
        x1, y1 = int(cx - w / 2), int(cy - h / 2)
        x2, y2 = int(cx + w / 2), int(cy + h / 2)
        return x1, y1, x2, y2, self.mode

    def _resample(self) -> None:
        positions = (np.arange(self.num_particles) + np.random.uniform()) / self.num_particles
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0
        indexes = np.searchsorted(cumulative_sum, positions)
        self.particles = self.particles[indexes].copy()
        self.weights.fill(1.0 / self.num_particles)