"""Тепловые карты UltiTrack по телеметрии (Фаза 1).

Модуль читает telemetry JSON (v2, с мягкой поддержкой v1), собирает позиции
игроков и рисует тепловые карты в метрах: сначала по командам A/B, затем
персональные по каждому ``id`` слота. Всё считается оффлайн: ни видео, ни
моделей для этого не нужно.
"""

from __future__ import annotations

import argparse
import json
import math
import numbers
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NamedTuple

import cv2
import numpy as np

from ultitrack.config import (
    ENDZONE_LENGTH,
    FIELD_LENGTH,
    FIELD_WIDTH,
    HEATMAP_BIN_SIZE_M,
    HEATMAP_SIGMA_M,
    TEAM_COLORS_BGR,
    UNKNOWN_TEAM_COLOR_BGR,
)

# Размер полотна картинки. Ширина поля в пикселях фиксирована, высота
# подстраивается под реальные пропорции поля из meta.
_RENDER_FIELD_WIDTH_PX = 360
_RENDER_MARGIN_PX = 24
_RENDER_HEADER_PX = 64
_RENDER_MIN_FIELD_PX = 80
_RENDER_GAP_PX = 12


# ---------------------------------------------------------------------------
# Чтение телеметрии
# ---------------------------------------------------------------------------


def load_telemetry(path: str | Path) -> dict:
    """Читает telemetry JSON и возвращает словарь.

    Поддерживаются и v1, и v2: структура не валидируется жёстко, чтобы старые
    файлы без ``status``/``team``/``teams`` не ломали анализ.
    """
    telemetry_path = Path(path)
    with telemetry_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Ожидался JSON-объект в {telemetry_path}, получено {type(data).__name__}")
    return data


def _is_real_number(value: Any) -> bool:
    """True для чисел JSON, кроме bool (он тоже наследник int)."""
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _normalize_team(value: Any) -> str | None:
    """Приводит метку команды к верхнему регистру; отсутствие — к None."""
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in {"", "NONE", "NULL", "UNKNOWN", "?"}:
        return None
    return text


def _requested_team(team: str | None) -> str | None:
    """Нормализует запрошенную команду; "unknown" — особый случай."""
    if team is None:
        return None
    text = str(team).strip().upper()
    if text in {"", "NONE", "NULL", "UNKNOWN", "?"}:
        return "UNKNOWN"
    return text


def _player_status(player: dict) -> str:
    """Статус игрока; отсутствие статуса в v1 считаем active."""
    status = player.get("status")
    if status is None:
        return "active"
    text = str(status).strip().lower()
    return text or "active"


def _player_xy(player: dict) -> tuple[float, float] | None:
    """Конечные координаты игрока в метрах или None."""
    x = player.get("x")
    y = player.get("y")
    if not _is_real_number(x) or not _is_real_number(y):
        return None
    try:
        xf = float(x)
        yf = float(y)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(xf) and math.isfinite(yf)):
        return None
    return xf, yf


def _iter_player_records(
    data: dict,
    team: str | None = None,
    statuses: tuple[str, ...] = ("active",),
) -> Iterator[tuple[dict, float, float]]:
    """Генерирует ``(player, x, y)`` для записей, прошедших фильтры.

    Здесь собрана вся общая фильтрация (команда, статус, валидность
    координат), чтобы :func:`collect_positions` и
    :func:`collect_player_samples` не расходились в поведении.
    """
    frames = data.get("frames")
    if not isinstance(frames, (list, tuple)):
        frames = []

    requested = _requested_team(team)
    wanted_statuses = {
        str(item).strip().lower()
        for item in (statuses or ())
        if item is not None and str(item).strip()
    }

    for frame in frames:
        if not isinstance(frame, dict):
            continue
        players = frame.get("players")
        if not isinstance(players, (list, tuple)):
            continue
        for player in players:
            if not isinstance(player, dict):
                continue

            player_team = _normalize_team(player.get("team"))
            if requested == "UNKNOWN":
                # Не приписываем безымянные точки команде A/B и наоборот.
                if player_team in {"A", "B"}:
                    continue
            elif requested is not None and player_team != requested:
                continue

            if wanted_statuses and _player_status(player) not in wanted_statuses:
                continue

            xy = _player_xy(player)
            if xy is None:
                continue
            yield player, xy[0], xy[1]


def collect_positions(
    data: dict,
    team: str | None = None,
    statuses: tuple[str, ...] = ("active",),
) -> np.ndarray:
    """Собирает координаты игроков в метрах в массив формы (N, 2).

    Параметры:
        data: разобранный telemetry JSON.
        team: ``None`` — все команды; ``"A"``/``"B"`` — конкретная команда;
            ``"unknown"`` — игроки без команды и любые не-A/B метки (нужно
            для v1-файлов, где ``team`` отсутствует).
        statuses: какие статусы брать. Отсутствующий ``status`` считается
            ``"active"``. Пустой кортеж отключает фильтр по статусу.

    Игроки без конечных числовых ``x``/``y`` пропускаются. Точки вне поля
    не отбрасываются — это уже дело :func:`build_heatmap`.
    """
    positions = [(x, y) for _, x, y in _iter_player_records(data, team, statuses)]
    if not positions:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(positions, dtype=np.float64).reshape(-1, 2)


class PlayerSamples(NamedTuple):
    """Точки одного слота игрока и его команда."""

    player_id: int
    team: str | None
    positions: np.ndarray


def _player_id(player: dict) -> int | None:
    """Целочисленный ``id`` слота или None, если его нет/он невалиден."""
    raw = player.get("id")
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, float) and not raw.is_integer():
        return None
    try:
        return int(raw)
    except (TypeError, ValueError, OverflowError):
        return None


def collect_player_samples(
    data: dict,
    statuses: tuple[str, ...] = ("active",),
) -> list[PlayerSamples]:
    """Собирает точки отдельно по каждому ``id`` игрока.

    Команда игрока определяется большинством голосов по его же кадрам:
    до калибровки ``team`` может быть ``null``, а позже — исправиться.
    Слоты без целочисленного ``id`` (часть старых v1-файлов) пропускаются.
    Возвращает список, отсортированный по ``player_id``.
    """
    points: dict[int, list[tuple[float, float]]] = {}
    team_votes: dict[int, Counter[str]] = {}

    for player, x, y in _iter_player_records(data, team=None, statuses=statuses):
        player_id = _player_id(player)
        if player_id is None:
            continue
        points.setdefault(player_id, []).append((x, y))
        player_team = _normalize_team(player.get("team"))
        if player_team is not None:
            team_votes.setdefault(player_id, Counter())[player_team] += 1

    samples: list[PlayerSamples] = []
    for player_id in sorted(points):
        votes = team_votes.get(player_id)
        team = votes.most_common(1)[0][0] if votes else None
        positions = np.asarray(points[player_id], dtype=np.float64).reshape(-1, 2)
        samples.append(PlayerSamples(player_id=player_id, team=team, positions=positions))
    return samples


# ---------------------------------------------------------------------------
# Сетка и отрисовка
# ---------------------------------------------------------------------------


def _positive_float(value: Any, default: float) -> float:
    """Мягко достаёт положительное число из meta, иначе берёт default."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed) or parsed <= 0.0:
        return default
    return parsed


def _non_negative_float(value: Any, default: float) -> float:
    """Мягко достаёт неотрицательное число из meta, иначе берёт default."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed) or parsed < 0.0:
        return default
    return parsed


def _field_geometry(data: dict) -> tuple[float, float, float]:
    """Возвращает (width, length, endzone) из meta.field или из конфига."""
    width = FIELD_WIDTH
    length = FIELD_LENGTH
    endzone = ENDZONE_LENGTH

    meta = data.get("meta")
    field = meta.get("field") if isinstance(meta, dict) else None
    if isinstance(field, dict):
        width = _positive_float(field.get("width"), FIELD_WIDTH)
        length = _positive_float(field.get("length"), FIELD_LENGTH)
        # В telemetry v2 ключ называется endzone, в venue.json — endzone_length.
        raw_endzone = field.get("endzone", field.get("endzone_length", ENDZONE_LENGTH))
        endzone = _non_negative_float(raw_endzone, ENDZONE_LENGTH)

    # Эндзона не должна наезжать на центральную линию.
    endzone = min(endzone, length / 2.0)
    return width, length, endzone


def build_heatmap(
    positions: np.ndarray,
    field_width: float,
    field_length: float,
    bin_size_m: float = HEATMAP_BIN_SIZE_M,
    sigma_m: float = HEATMAP_SIGMA_M,
) -> np.ndarray:
    """Строит нормированную в 0..1 2D-гистограмму плотности.

    Форма результата ``(ny, nx)``, строка 0 соответствует ``y = 0``.
    Координаты клипаются к ``[0, field_width] x [0, field_length]``, поэтому
    случайные выбросы за поле не меняют форму сетки. ``sigma_m=0`` отключает
    гауссово размытие. При отсутствии точек возвращается нулевая сетка.
    """
    width = float(field_width)
    length = float(field_length)
    if not (math.isfinite(width) and math.isfinite(length)) or width <= 0 or length <= 0:
        raise ValueError("Размеры поля должны быть положительными конечными числами")
    if not math.isfinite(float(bin_size_m)) or bin_size_m <= 0:
        raise ValueError("bin_size_m должен быть > 0")
    if not math.isfinite(float(sigma_m)) or sigma_m < 0:
        raise ValueError("sigma_m должен быть >= 0")

    nx = max(1, int(math.ceil(width / float(bin_size_m))))
    ny = max(1, int(math.ceil(length / float(bin_size_m))))

    arr = np.asarray(positions, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((ny, nx), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"positions должен иметь форму (N, 2), получено {arr.shape}")

    finite = np.isfinite(arr).all(axis=1)
    arr = arr[finite]
    if arr.shape[0] == 0:
        return np.zeros((ny, nx), dtype=np.float32)

    # Клип: точки на границе попадают в крайние бины, а не ломают диапазон.
    xs = np.clip(arr[:, 0], 0.0, width)
    ys = np.clip(arr[:, 1], 0.0, length)

    histogram, _, _ = np.histogram2d(
        xs,
        ys,
        bins=(nx, ny),
        range=((0.0, width), (0.0, length)),
    )
    grid = histogram.T.astype(np.float64)  # (ny, nx): ось Y — строки

    # Плотность на квадратный метр (на нормировку не влияет, но честнее).
    bin_w = width / nx
    bin_h = length / ny
    grid /= bin_w * bin_h

    if sigma_m > 0.0:
        sigma_x = float(sigma_m) / bin_w
        sigma_y = float(sigma_m) / bin_h
        grid = cv2.GaussianBlur(
            grid,
            (0, 0),
            sigma_x,
            sigma_y,
            borderType=cv2.BORDER_CONSTANT,
        )

    peak = float(grid.max()) if grid.size else 0.0
    if peak > 0.0:
        grid = grid / peak
    else:
        grid = np.zeros_like(grid)

    return np.clip(grid, 0.0, 1.0).astype(np.float32)


def _title_accent_bgr(title: str) -> tuple[int, int, int]:
    """Цветовой акцент в шапке картинки по названию команды."""
    tokens = title.upper().replace("_", " ").replace("-", " ").split()
    if "A" in tokens:
        return TEAM_COLORS_BGR.get("A", (255, 255, 255))
    if "B" in tokens:
        return TEAM_COLORS_BGR.get("B", (255, 255, 255))
    return UNKNOWN_TEAM_COLOR_BGR


def _put_centered_text(
    image: np.ndarray,
    text: str,
    center: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """Рисует текст, центрируя его по указанной точке."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    origin = (int(center[0] - text_w / 2), int(center[1] + text_h / 2))
    cv2.putText(image, text, origin, font, scale, color, thickness, cv2.LINE_AA)


def render_heatmap_image(
    grid: np.ndarray,
    field_width: float,
    field_length: float,
    endzone_length: float,
    title: str,
    sample_count: int | None = None,
    team: str | None = None,
) -> np.ndarray:
    """Рисует BGR-картинку поля с наложенной тепловой картой.

    Ориентация: ``x`` поперёк поля (0..width), ``y`` вдоль (0..length),
    ``y = 0`` сверху. В шапке выводятся название и число точек.
    ``team`` задаёт цветовой маркер команды явно (для персональных карт,
    где в заголовке нет отдельного токена "A"/"B"). Для пустой сетки или
    нулевого sample_count на поле пишется ``no data``.
    """
    width = float(field_width)
    length = float(field_length)
    if not (math.isfinite(width) and math.isfinite(length)) or width <= 0 or length <= 0:
        raise ValueError("Размеры поля должны быть положительными конечными числами")

    endzone = float(endzone_length)
    if not math.isfinite(endzone) or endzone < 0:
        endzone = 0.0
    endzone = min(endzone, length / 2.0)

    scale = _RENDER_FIELD_WIDTH_PX / width
    field_w_px = max(_RENDER_MIN_FIELD_PX, int(round(width * scale)))
    field_h_px = max(_RENDER_MIN_FIELD_PX, int(round(length * scale)))

    margin = _RENDER_MARGIN_PX
    header = _RENDER_HEADER_PX
    image = np.full(
        (header + field_h_px + margin, field_w_px + 2 * margin, 3),
        (32, 32, 32),
        dtype=np.uint8,
    )
    field = np.full((field_h_px, field_w_px, 3), (45, 110, 65), dtype=np.uint8)

    grid_arr = np.asarray(grid, dtype=np.float64)
    has_heat = grid_arr.size > 0 and np.isfinite(grid_arr).any() and float(np.nanmax(grid_arr)) > 0.0
    if has_heat:
        if grid_arr.ndim != 2:
            raise ValueError(f"grid должен быть 2D, получено {grid_arr.shape}")
        clean = np.nan_to_num(grid_arr, nan=0.0, posinf=0.0, neginf=0.0)
        clean = np.clip(clean, 0.0, None)
        peak = float(clean.max())
        if peak > 0.0:
            clean = clean / peak
        heat = cv2.resize(clean, (field_w_px, field_h_px), interpolation=cv2.INTER_LINEAR)
        heat_u8 = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        alpha = np.clip(heat * 0.75, 0.0, 0.75)[:, :, None]
        blended = field.astype(np.float32) * (1.0 - alpha) + colored.astype(np.float32) * alpha
        field = np.clip(blended, 0, 255).astype(np.uint8)

    # Линии поля рисуем поверх тепловой карты, чтобы разметка читалась.
    line_color = (255, 255, 255)
    thin_color = (200, 230, 200)
    thickness = max(1, int(round(scale * 0.10)))
    cv2.rectangle(field, (0, 0), (field_w_px - 1, field_h_px - 1), line_color, thickness, cv2.LINE_AA)

    if endzone > 0.0:
        far_y = int(round(endzone * scale))
        near_y = int(round((length - endzone) * scale))
        far_y = min(max(far_y, 1), field_h_px - 2)
        near_y = min(max(near_y, 1), field_h_px - 2)
        cv2.line(field, (0, far_y), (field_w_px - 1, far_y), line_color, thickness, cv2.LINE_AA)
        cv2.line(field, (0, near_y), (field_w_px - 1, near_y), line_color, thickness, cv2.LINE_AA)

    middle_y = min(max(int(round(length / 2.0 * scale)), 1), field_h_px - 2)
    cv2.line(field, (0, middle_y), (field_w_px - 1, middle_y), thin_color, 1, cv2.LINE_AA)

    no_data = (sample_count is not None and sample_count <= 0) or not has_heat
    if no_data:
        _put_centered_text(
            field,
            "no data",
            (field_w_px // 2, field_h_px // 2),
            scale=max(0.8, min(1.4, scale * 0.09)),
            color=(180, 180, 180),
            thickness=2,
        )

    image[header : header + field_h_px, margin : margin + field_w_px] = field

    # Шапка: цветовой маркер команды, название и число точек.
    if team is None:
        accent = _title_accent_bgr(str(title))
    else:
        accent = TEAM_COLORS_BGR.get(str(team).strip().upper(), UNKNOWN_TEAM_COLOR_BGR)
    cv2.rectangle(image, (margin, 18), (margin + 14, 34), accent, -1, cv2.LINE_AA)
    cv2.putText(
        image,
        str(title),
        (margin + 24, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if sample_count is None:
        subtitle = ""
    elif sample_count <= 0:
        subtitle = "points: 0 (no data)"
    else:
        subtitle = f"points: {int(sample_count)}"
    if subtitle:
        cv2.putText(
            image,
            subtitle,
            (margin + 24, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

    return image


# ---------------------------------------------------------------------------
# Сохранение набора карт
# ---------------------------------------------------------------------------


def _side_by_side(images: list[np.ndarray], gap: int = _RENDER_GAP_PX) -> np.ndarray:
    """Склеивает картинки по горизонтали, выравнивая их по высоте."""
    if not images:
        raise ValueError("Нет изображений для склейки")
    height = max(image.shape[0] for image in images)
    width = sum(image.shape[1] for image in images) + gap * (len(images) - 1)
    canvas = np.full((height, width, 3), (32, 32, 32), dtype=np.uint8)
    x_offset = 0
    for image in images:
        y_offset = (height - image.shape[0]) // 2
        canvas[
            y_offset : y_offset + image.shape[0],
            x_offset : x_offset + image.shape[1],
        ] = image
        x_offset += image.shape[1] + gap
    return canvas


def _tile_images(
    images: list[np.ndarray],
    columns: int,
    gap: int = _RENDER_GAP_PX,
) -> np.ndarray:
    """Раскладывает картинки сеткой слева направо и сверху вниз."""
    if not images:
        raise ValueError("Нет изображений для склейки")
    columns = max(1, min(int(columns), len(images)))
    rows = [_side_by_side(images[start : start + columns], gap) for start in range(0, len(images), columns)]
    width = max(row.shape[1] for row in rows)
    height = sum(row.shape[0] for row in rows) + gap * (len(rows) - 1)
    canvas = np.full((height, width, 3), (32, 32, 32), dtype=np.uint8)
    y_offset = 0
    for row in rows:
        canvas[y_offset : y_offset + row.shape[0], : row.shape[1]] = row
        y_offset += row.shape[0] + gap
    return canvas


def _write_png(path: Path, image: np.ndarray) -> None:
    """Сохраняет BGR-картинку и падает с понятной ошибкой, если не вышло."""
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise OSError(f"Не удалось сохранить изображение: {path}")


def save_team_heatmaps(
    telemetry_path: str | Path,
    out_dir: str | Path,
    bin_size_m: float = HEATMAP_BIN_SIZE_M,
    sigma_m: float = HEATMAP_SIGMA_M,
) -> list[Path]:
    """Сохраняет тепловые карты команд A, B (и unknown, если он есть).

    Файлы: ``heatmap_<stem>_team_<T>.png`` и ``heatmap_<stem>_combined.png``
    (A и B рядом). Для команды без active-точек всё равно рисуется корректная
    картинка с надписью ``no data``. Возвращает список созданных файлов.
    """
    source = Path(telemetry_path)
    data = load_telemetry(source)
    field_width, field_length, endzone = _field_geometry(data)

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    created: list[Path] = []

    per_team_images: dict[str, np.ndarray] = {}
    for team in ("A", "B"):
        positions = collect_positions(data, team=team)
        grid = build_heatmap(
            positions,
            field_width,
            field_length,
            bin_size_m=bin_size_m,
            sigma_m=sigma_m,
        )
        image = render_heatmap_image(
            grid,
            field_width,
            field_length,
            endzone,
            title=f"Team {team}",
            sample_count=int(positions.shape[0]),
        )
        target = output_dir / f"heatmap_{stem}_team_{team}.png"
        _write_png(target, image)
        created.append(target)
        per_team_images[team] = image
        if positions.shape[0] == 0:
            print(f"[Heatmaps] Team {team}: нет active-точек, сохраняю пустую карту")
        else:
            print(f"[Heatmaps] Team {team}: {positions.shape[0]} точек -> {target}")

    # unknown появляется только если реально есть безымянные/не-A/B точки.
    unknown_positions = collect_positions(data, team="unknown")
    if unknown_positions.shape[0] > 0:
        grid = build_heatmap(
            unknown_positions,
            field_width,
            field_length,
            bin_size_m=bin_size_m,
            sigma_m=sigma_m,
        )
        image = render_heatmap_image(
            grid,
            field_width,
            field_length,
            endzone,
            title="Team unknown",
            sample_count=int(unknown_positions.shape[0]),
        )
        target = output_dir / f"heatmap_{stem}_team_unknown.png"
        _write_png(target, image)
        created.append(target)
        print(f"[Heatmaps] Team unknown: {unknown_positions.shape[0]} точек -> {target}")

    combined = _side_by_side([per_team_images["A"], per_team_images["B"]])
    combined_path = output_dir / f"heatmap_{stem}_combined.png"
    _write_png(combined_path, combined)
    created.append(combined_path)
    print(f"[Heatmaps] Combined A+B -> {combined_path}")

    return created


def _team_slug(team: str | None) -> str:
    """Короткая метка команды для имени файла."""
    return team if team in {"A", "B"} else "unknown"


def save_player_heatmaps(
    telemetry_path: str | Path,
    out_dir: str | Path,
    bin_size_m: float = HEATMAP_BIN_SIZE_M,
    sigma_m: float = HEATMAP_SIGMA_M,
    columns: int = 5,
) -> list[Path]:
    """Сохраняет персональные тепловые карты по ``id`` слота.

    Игрок определяется стабильным ``id`` слота из telemetry v2 (1..10),
    команда — большинством голосов по его кадрам. Для каждого игрока с
    active-точками пишется ``heatmap_<stem>_player_<NN>_team_<T>.png``,
    затем мозаики ``heatmap_<stem>_players_team_<T>.png`` (по ``columns``
    карт в ряд) и общая ``heatmap_<stem>_players_combined.png``.
    Возвращает список созданных файлов.
    """
    source = Path(telemetry_path)
    data = load_telemetry(source)
    field_width, field_length, endzone = _field_geometry(data)

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    created: list[Path] = []

    samples = collect_player_samples(data)
    if not samples:
        print("[Heatmaps] Игроков с active-точками и id нет — персональные карты пропущены")
        return created

    per_team: dict[str, list[np.ndarray]] = {}
    for sample in samples:
        slug = _team_slug(sample.team)
        grid = build_heatmap(
            sample.positions,
            field_width,
            field_length,
            bin_size_m=bin_size_m,
            sigma_m=sigma_m,
        )
        image = render_heatmap_image(
            grid,
            field_width,
            field_length,
            endzone,
            title=f"Player {sample.player_id} (Team {sample.team or 'unknown'})",
            sample_count=int(sample.positions.shape[0]),
            team=sample.team,
        )
        target = output_dir / f"heatmap_{stem}_player_{sample.player_id:02d}_team_{slug}.png"
        _write_png(target, image)
        created.append(target)
        per_team.setdefault(slug, []).append(image)
        print(
            f"[Heatmaps] Player {sample.player_id} (team {sample.team or 'unknown'}): "
            f"{sample.positions.shape[0]} точек -> {target}"
        )

    mosaics: list[np.ndarray] = []
    for team in sorted(per_team):
        mosaic = _tile_images(per_team[team], columns)
        target = output_dir / f"heatmap_{stem}_players_team_{team}.png"
        _write_png(target, mosaic)
        created.append(target)
        mosaics.append(mosaic)
        print(f"[Heatmaps] Мозаика игроков team {team} -> {target}")

    if len(mosaics) > 1:
        combined_path = output_dir / f"heatmap_{stem}_players_combined.png"
        _write_png(combined_path, _tile_images(mosaics, 1))
        created.append(combined_path)
        print(f"[Heatmaps] Все игроки -> {combined_path}")

    return created


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI: python -m ultitrack.analytics.heatmaps <telemetry.json> ..."""
    parser = argparse.ArgumentParser(
        description="Командные и персональные тепловые карты UltiTrack по telemetry JSON",
    )
    parser.add_argument("telemetry", type=Path, help="путь к telemetry_*.json")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="куда сохранять PNG (по умолчанию <папка телеметрии>/heatmaps)",
    )
    parser.add_argument(
        "--bin-size",
        type=float,
        default=HEATMAP_BIN_SIZE_M,
        help=f"размер бина в метрах (по умолчанию {HEATMAP_BIN_SIZE_M})",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=HEATMAP_SIGMA_M,
        help=f"sigma гауссова размытия в метрах (по умолчанию {HEATMAP_SIGMA_M})",
    )
    parser.add_argument(
        "--players",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="строить также персональные карты игроков (по умолчанию включено)",
    )
    parser.add_argument(
        "--player-columns",
        type=int,
        default=5,
        help="сколько персональных карт в ряд в мозаике (по умолчанию 5)",
    )
    args = parser.parse_args(argv)

    telemetry_path: Path = args.telemetry
    if not telemetry_path.is_file():
        print(f"[Heatmaps] Файл не найден: {telemetry_path}", file=sys.stderr)
        return 1

    output_dir = args.out_dir if args.out_dir is not None else telemetry_path.parent / "heatmaps"
    try:
        created = save_team_heatmaps(
            telemetry_path,
            output_dir,
            bin_size_m=args.bin_size,
            sigma_m=args.sigma,
        )
        if args.players:
            created += save_player_heatmaps(
                telemetry_path,
                output_dir,
                bin_size_m=args.bin_size,
                sigma_m=args.sigma,
                columns=args.player_columns,
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[Heatmaps] Ошибка: {exc}", file=sys.stderr)
        return 1

    for path in created:
        print(f"[Heatmaps] Готово: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
