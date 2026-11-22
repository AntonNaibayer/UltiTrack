"""Оценка дисторсии по прямым линиям на кадре из вашего видео.

Запуск из корня проекта:
    uv run python tools/estimate_distortion.py data/raw/match3.mp4 calib/lens.json

Управление:
    ЛКМ  - добавить точку на текущую линию (кликайте вдоль линии, 6-12 точек, по всей её длине)
    ПКМ  - удалить последнюю точку
    N    - закончить линию и начать следующую (нужно минимум 4 точки)
    ENTER - закончить (нужно минимум 4 линии)

Какие линии брать: длинные, РЕАЛЬНО прямые, ближе к краям кадра и в разных направлениях
(линии пола, границы красных зон, нижний край стен, балки, края сетки).
Линии, проходящие через центр кадра, почти бесполезны: они прямые при любой дисторсии.
"""

import sys

import cv2
import numpy as np

from ultitrack.lens import fit_division_model

COLORS = [(0, 255, 0), (0, 200, 255), (255, 0, 255), (255, 255, 0), (0, 128, 255), (255, 100, 100)]


def read_frame(path: str) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Не удалось прочитать кадр из {path}")
    return frame


def collect_lines(frame: np.ndarray) -> list[np.ndarray]:
    lines: list[list[tuple[int, int]]] = [[]]
    win = "Click points along straight lines (N - next line, RMB - undo, ENTER - done)"

    def cb(event: int, x: int, y: int, flags: int, param: None) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            lines[-1].append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and lines[-1]:
            lines[-1].pop()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    cv2.setMouseCallback(win, cb)

    while True:
        canvas = frame.copy()
        for i, pts in enumerate(lines):
            color = COLORS[i % len(COLORS)]
            for p in pts:
                cv2.circle(canvas, p, 5, color, -1)
            if len(pts) >= 2:
                cv2.polylines(canvas, [np.array(pts, np.int32)], False, color, 1)

        good = sum(len(l) >= 4 for l in lines)
        cv2.rectangle(canvas, (10, 10), (760, 50), (0, 0, 0), -1)
        cv2.putText(
            canvas,
            f"Line {len(lines)} | good lines: {good} | N next, ENTER done (need 4+)",
            (20, 37),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
        )
        cv2.imshow(win, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("n"), ord("N")) and len(lines[-1]) >= 4:
            lines.append([])
        elif key in (13, 32) and good >= 4:
            break

    cv2.destroyWindow(win)
    return [np.array(l, dtype=np.float64) for l in lines if len(l) >= 4]


def undistort_image(frame: np.ndarray, lens) -> np.ndarray:
    h, w = frame.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    grid = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
    src = lens.distort(grid).astype(np.float32)
    map_x = src[:, 0].reshape(h, w)
    map_y = src[:, 1].reshape(h, w)
    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Использование: python tools/estimate_distortion.py <video> <out_lens.json>")
    video_path, out_path = sys.argv[1], sys.argv[2]

    frame = read_frame(video_path)
    h, w = frame.shape[:2]
    lines = collect_lines(frame)

    lens, rms_before, rms_after = fit_division_model(lines, w, h)
    print(f"Линий: {len(lines)}, точек: {sum(len(l) for l in lines)}")
    print(f"Отклонение от прямых: до {rms_before:.2f} px -> после {rms_after:.2f} px")
    print(f"k1 = {lens.k1:.4f}, k2 = {lens.k2:.4f}")
    if rms_after > 2.5:
        print("ВНИМАНИЕ: остаточная ошибка большая. Проверьте, что линии реально прямые, и добавьте линий.")

    lens.save(out_path)
    print(f"Сохранено: {out_path}")

    # Визуальная проверка: исходный и исправленный кадр рядом
    undist = undistort_image(frame, lens)
    side = np.hstack([frame, undist])
    cv2.imwrite("undistort_check.png", side)
    cv2.namedWindow("Original | Undistorted (any key to close)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Original | Undistorted (any key to close)", 1600, 450)
    cv2.imshow("Original | Undistorted (any key to close)", side)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()