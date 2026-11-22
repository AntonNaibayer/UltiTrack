"""Простая модель дисторсии для точек (division model) и подбор её параметров по прямым линиям.

x_u = x_d / (1 + k1 * r^2 + k2 * r^4),  r считается от центра кадра в нормированных координатах.
Работает только с точками (не с кадром), поэтому YOLO продолжает видеть исходный кадр.
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


class DivisionLens:
    def __init__(
        self,
        width: int,
        height: int,
        k1: float = 0.0,
        k2: float = 0.0,
        cx: float | None = None,
        cy: float | None = None,
    ) -> None:
        self.width, self.height = width, height
        self.k1, self.k2 = k1, k2
        self.cx = width / 2.0 if cx is None else cx
        self.cy = height / 2.0 if cy is None else cy
        self.s = width / 2.0  # масштаб нормировки

    def _norm(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        return (pts - np.array([self.cx, self.cy])) / self.s

    def _denorm(self, xy: np.ndarray) -> np.ndarray:
        return xy * self.s + np.array([self.cx, self.cy])

    def undistort(self, pts: np.ndarray) -> np.ndarray:
        """Пиксели искажённого кадра -> пиксели 'идеального' кадра без дисторсии."""
        xy = self._norm(pts)
        r2 = np.sum(xy**2, axis=1, keepdims=True)
        return self._denorm(xy / (1.0 + self.k1 * r2 + self.k2 * r2**2))

    def distort(self, pts: np.ndarray, iters: int = 30) -> np.ndarray:
        """Обратное преобразование (итерациями): нужно для remap и для рисования сетки поля на кадре."""
        xu = self._norm(pts)
        xd = xu.copy()
        for _ in range(iters):
            r2 = np.sum(xd**2, axis=1, keepdims=True)
            xd = xu * (1.0 + self.k1 * r2 + self.k2 * r2**2)
        return self._denorm(xd)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model": "division",
            "width": self.width,
            "height": self.height,
            "k1": self.k1,
            "k2": self.k2,
            "cx": self.cx,
            "cy": self.cy,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "DivisionLens":
        d = json.loads(Path(path).read_text())
        return cls(d["width"], d["height"], d["k1"], d["k2"], d["cx"], d["cy"])


def line_residuals(pts: np.ndarray) -> np.ndarray:
    """Перпендикулярные расстояния точек до лучшей прямой (метод наименьших квадратов, SVD)."""
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c)
    normal = vt[-1]
    return (pts - c) @ normal


def fit_division_model(
    lines: list[np.ndarray], width: int, height: int
) -> tuple[DivisionLens, float, float]:
    """lines: список массивов (N, 2) с точками вдоль РЕАЛЬНО прямых линий на искажённом кадре.
    Возвращает (lens, rms_до, rms_после) в пикселях."""

    def data_residuals(k1: float, k2: float) -> np.ndarray:
        lens = DivisionLens(width, height, k1, k2)
        return np.concatenate([line_residuals(lens.undistort(l)) for l in lines])

    def residuals(params: np.ndarray) -> np.ndarray:
        # небольшая регуляризация k2, чтобы не переобучаться на мало линиях
        return np.concatenate([data_residuals(*params), 20.0 * params[1:2]])

    rms_before = float(np.sqrt(np.mean(data_residuals(0.0, 0.0) ** 2)))
    sol = least_squares(residuals, x0=np.zeros(2))
    k1, k2 = float(sol.x[0]), float(sol.x[1])
    rms_after = float(np.sqrt(np.mean(data_residuals(k1, k2) ** 2)))
    return DivisionLens(width, height, k1, k2), rms_before, rms_after