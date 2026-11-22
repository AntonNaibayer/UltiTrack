"""Метрики качества диска и игроков по telemetry_*.json.

Запуск из корня проекта:
    python tools/analyze_telemetry.py data/processed/telemetry_match4.json
    python tools/analyze_telemetry.py старый.json новый.json      # сравнение
"""

import collections
import json
import sys

import numpy as np

FIELD_MARGIN = 1.0  # м: "вне поля" для диска и игроков


def runs(seq: list[str]) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    cur, n = None, 0
    for x in seq:
        if x == cur:
            n += 1
        else:
            if cur is not None:
                out.append((cur, n))
            cur, n = x, 1
    if cur is not None:
        out.append((cur, n))
    return out


def analyze(path: str) -> dict:
    data = json.load(open(path, encoding="utf-8"))
    frames = data["frames"]
    fps = float(data["meta"]["fps"])
    w = float(data["meta"]["field"]["width"])
    length = float(data["meta"]["field"]["length"])
    n = len(frames)

    # ---- диск
    states: list[str] = []
    pos: list[tuple[float, float] | None] = []
    for f in frames:
        d = f.get("disc")
        if d:
            st = d["state"]
            states.append("HELD" if st.startswith("HELD") else st)
            pos.append((d["x"], d["y"]))
        else:
            states.append("NONE")
            pos.append(None)

    jumps = []
    for a, b in zip(pos[:-1], pos[1:]):
        if a and b:
            jumps.append(float(np.hypot(b[0] - a[0], b[1] - a[1])))
    jumps_arr = np.array(jumps) if jumps else np.zeros(1)

    out_field = sum(
        1
        for p in pos
        if p and not (-FIELD_MARGIN <= p[0] <= w + FIELD_MARGIN and -FIELD_MARGIN <= p[1] <= length + FIELD_MARGIN)
    )
    r = runs(states)
    counts = collections.Counter(states)

    # ---- игроки
    present: dict[int, np.ndarray] = collections.defaultdict(lambda: np.zeros(n, dtype=bool))
    for i, f in enumerate(frames):
        for p in f["players"]:
            present[p["id"]][i] = True
    gap_lens: list[int] = []
    for mask in present.values():
        g = 0
        for v in mask:
            if not v:
                g += 1
            else:
                if g:
                    gap_lens.append(g)
                g = 0
        if g:
            gap_lens.append(g)
    gaps = np.array(gap_lens) if gap_lens else np.zeros(1, dtype=int)
    slots = max(len(present), 1)
    presence = float(np.mean([m.mean() for m in present.values()])) if present else 0.0

    return {
        "fps": fps,
        "frames": n,
        "disc_state_changes": len(r) - 1,
        "disc_states": {k: f"{v / n:.0%}" for k, v in counts.most_common()},
        "disc_jumps_gt_1m_per_frame": int((jumps_arr > 1.0).sum()),
        "disc_max_plausible_m_per_frame": round(30.0 / fps, 2),
        "disc_out_of_field_frames": f"{out_field}/{n} ({out_field / n:.0%})",
        "players_slots": slots,
        "players_mean_presence": f"{presence:.0%}",
        "players_gaps_total": int(len(gap_lens)),
        "players_gaps_le_5_frames": int((gaps <= 5).sum()) if gap_lens else 0,
        "players_gaps_gt_30_frames": int((gaps > 30).sum()) if gap_lens else 0,
    }


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit("Использование: python tools/analyze_telemetry.py a.json [b.json]")
    results = [analyze(p) for p in paths]
    keys = list(results[0].keys())
    width = max(len(k) for k in keys) + 2
    header = " " * width + "".join(f"{p.split('/')[-1][:28]:<32}" for p in paths)
    print(header)
    for k in keys:
        print(f"{k:<{width}}" + "".join(f"{str(r[k]):<32}" for r in results))


if __name__ == "__main__":
    main()