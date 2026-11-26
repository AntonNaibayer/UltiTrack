"""CLI: python -m ultitrack.cvat {extract,label,export,import-cvat,info}."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultitrack.cvat.tracks import (
    LABEL_DISC,
    LABEL_PLAYER,
    TrackSet,
    default_output_dir,
    extract_tracks,
    resolve_video_path,
)
from ultitrack.cvat.teams import TeamBindings, apply_bindings
from ultitrack.cvat.xml_io import (
    parse_cvat_video_xml,
    write_cvat_video_xml,
    write_import_csv,
    write_import_json,
    write_import_teams,
    write_labels_json,
    write_summary_csv,
)


def _tracks_path(args: argparse.Namespace, out_dir: Path) -> Path:
    return Path(args.tracks) if getattr(args, "tracks", None) else out_dir / "tracks.json"


def _teams_path(args: argparse.Namespace, out_dir: Path) -> Path:
    return Path(args.teams) if getattr(args, "teams", None) else out_dir / "teams.json"


def _add_label_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--player-label",
        default=LABEL_PLAYER,
        help=f"имя метки игрока в CVAT (по умолчанию {LABEL_PLAYER})",
    )
    parser.add_argument(
        "--disc-label",
        default=LABEL_DISC,
        help=f"имя метки диска в CVAT (по умолчанию {LABEL_DISC})",
    )
    parser.add_argument(
        "--team-attr",
        default="team",
        help="имя атрибута команды в CVAT (по умолчанию team)",
    )


def cmd_extract(args: argparse.Namespace) -> int:
    video_path = resolve_video_path(args.video)
    out_dir = Path(args.out_dir) if args.out_dir else default_output_dir(video_path)
    tracks_path = _tracks_path(args, out_dir)
    if tracks_path.exists() and not args.force:
        track_set = TrackSet.load(tracks_path)
        counts = track_set.counts()
        print(f"[CVAT] Уже есть {tracks_path} ({counts['players']} игроков, {counts['discs']} диск).")
        print("[CVAT] Перезапустить: --force")
        return 0

    device = None if args.device in (None, "", "auto") else args.device
    track_set = extract_tracks(
        video_path,
        conf=args.conf,
        disc_conf=args.disc_conf,
        imgsz=args.imgsz,
        device=device,
        field_filter=not args.no_field_filter,
        field_margin=args.field_margin,
        min_track_frames=args.min_track_frames,
        max_frames=args.max_frames,
    )
    track_set.save(tracks_path)
    print(f"[CVAT] Треки сохранены: {tracks_path}")
    print(f"[CVAT] Дальше: python -m ultitrack.cvat label {video_path}")
    return 0


def cmd_label(args: argparse.Namespace) -> int:
    from ultitrack.cvat.server import run_label_server

    video_path = resolve_video_path(args.video)
    out_dir = Path(args.out_dir) if args.out_dir else default_output_dir(video_path)
    tracks_path = _tracks_path(args, out_dir)
    if not tracks_path.exists():
        print(f"[CVAT] Нет {tracks_path}. Сначала выполните:")
        print(f"  python -m ultitrack.cvat extract {video_path}")
        return 1
    track_set = TrackSet.load(tracks_path)
    if not track_set.player_tracks:
        print("[CVAT] В tracks.json нет игроков — проверьте порог conf/фильтр поля.")
        return 1

    bindings_path = _teams_path(args, out_dir)
    if bindings_path.exists():
        bindings = TeamBindings.load(bindings_path, video=track_set.meta.video_name)
        apply_bindings(track_set, bindings)
        print(f"[CVAT] Загружены команды: {bindings_path}")

    run_label_server(
        video_path,
        track_set,
        out_dir,
        port=args.port,
        open_browser=not args.no_browser,
        teams_path=bindings_path,
        player_label=args.player_label,
        disc_label=args.disc_label,
        team_attr=args.team_attr,
        stride=args.stride,
        disc_stride=args.disc_stride,
        bridge_gap=args.bridge_gap,
    )
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    video_path = resolve_video_path(args.video)
    out_dir = Path(args.out_dir) if args.out_dir else default_output_dir(video_path)
    tracks_path = _tracks_path(args, out_dir)
    if not tracks_path.exists():
        print(f"[CVAT] Нет {tracks_path}. Сначала выполните extract/label.")
        return 1

    track_set = TrackSet.load(tracks_path)
    bindings_path = _teams_path(args, out_dir)
    if bindings_path.exists():
        bindings = TeamBindings.load(bindings_path, video=track_set.meta.video_name)
    else:
        print(f"[CVAT] {bindings_path} не найден: все команды будут unknown.")
        bindings = TeamBindings(video=track_set.meta.video_name)
    known_ids = {track.id for track in track_set.tracks}
    stale = sorted(
        track_id for track_id in bindings.assignments if track_id not in known_ids
    )
    if stale:
        print(
            f"[CVAT] В {bindings_path} есть ID, которых нет в tracks.json: "
            f"{stale[:10]}{'…' if len(stale) > 10 else ''}. "
            "Они будут проигнорированы."
        )
    apply_bindings(track_set, bindings)

    xml_path = write_cvat_video_xml(
        track_set,
        Path(args.xml) if args.xml else out_dir / "annotations.xml",
        player_label=args.player_label,
        disc_label=args.disc_label,
        team_attr=args.team_attr,
        stride=args.stride,
        disc_stride=args.disc_stride,
        bridge_gap=args.bridge_gap,
    )
    csv_path = write_summary_csv(track_set, out_dir / "tracks.csv")
    labels_path = write_labels_json(
        out_dir / "cvat_labels.json",
        player_label=args.player_label,
        disc_label=args.disc_label,
        team_attr=args.team_attr,
    )
    counts = track_set.counts()
    print(
        f"[CVAT] XML: {xml_path}\n"
        f"[CVAT] CSV: {csv_path}\n"
        f"[CVAT] Метки: {labels_path}\n"
        f"[CVAT] Игроков {counts['players']} (A={counts['A']}, B={counts['B']}, "
        f"unknown={counts['unknown']}, excluded={counts['excluded']}), "
        f"диск {counts['discs']}."
    )
    return 0


def cmd_import_cvat(args: argparse.Namespace) -> int:
    parsed = parse_cvat_video_xml(Path(args.xml), team_attr=args.team_attr)
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.xml).parent
    json_path = write_import_json(
        parsed, out_dir / "cvat_tracks.json", per_frame=args.per_frame
    )
    csv_path = write_import_csv(parsed, out_dir / "cvat_tracks.csv")
    teams_path = Path(args.teams_out) if args.teams_out else out_dir / "teams_from_cvat.json"
    write_import_teams(parsed, teams_path, player_label=args.player_label)
    print(
        f"[CVAT] Импорт: {len(parsed.tracks)} треков, {parsed.total_frames} кадров.\n"
        f"[CVAT] JSON: {json_path}\n"
        f"[CVAT] CSV: {csv_path}\n"
        f"[CVAT] Команды: {teams_path}"
    )
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    video_path = resolve_video_path(args.video)
    out_dir = Path(args.out_dir) if args.out_dir else default_output_dir(video_path)
    tracks_path = _tracks_path(args, out_dir)
    if not tracks_path.exists():
        print(f"[CVAT] Нет {tracks_path}.")
        return 1
    track_set = TrackSet.load(tracks_path)
    bindings_path = _teams_path(args, out_dir)
    bindings = (
        TeamBindings.load(bindings_path, video=track_set.meta.video_name)
        if bindings_path.exists()
        else TeamBindings(video=track_set.meta.video_name)
    )
    apply_bindings(track_set, bindings)
    meta = track_set.meta
    counts = track_set.counts()
    print(
        f"{meta.video_name}: {meta.width}x{meta.height}, {meta.fps:.2f} fps, "
        f"{meta.total_frames} кадров\n"
        f"Игроков: {counts['players']}, диск: {counts['discs']}; "
        f"A={counts['A']}, B={counts['B']}, unknown={counts['unknown']}, "
        f"excluded={counts['excluded']}\n"
        f"Файлы: {out_dir}"
    )
    for track in track_set.player_tracks:
        print(
            f"  #{track.id:<3} {track.team:<8} {track.team_source:<7} "
            f"f{track.first_frame}–{track.last_frame} ({len(track.frames)} кадр.)"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ultitrack.cvat",
        description=(
            "Предразметка видео для CVAT: YOLO + ByteTrack, ручные команды "
            "и импорт/экспорт CVAT for video 1.1."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser(
        "extract", help="прогнать модель и сохранить tracks.json"
    )
    extract.add_argument("video", help="путь к видео или имя файла в data/raw")
    extract.add_argument("--out-dir", default=None, help="папка результатов")
    extract.add_argument("--tracks", default=None, help="путь к tracks.json")
    extract.add_argument("--conf", type=float, default=0.15, help="порог игроков")
    extract.add_argument("--disc-conf", type=float, default=0.22, help="порог диска")
    extract.add_argument("--imgsz", type=int, default=1280, help="размер инференса YOLO")
    extract.add_argument("--device", default=None, help="cpu / cuda:0 (по умолчанию авто)")
    extract.add_argument(
        "--no-field-filter",
        action="store_true",
        help="не отбрасывать людей и диски вне поля",
    )
    extract.add_argument(
        "--field-margin", type=float, default=1.0, help="допуск поля, м"
    )
    extract.add_argument(
        "--min-track-frames",
        type=int,
        default=3,
        help="выбросить треки короче N кадров (ложные срабатывания)",
    )
    extract.add_argument(
        "--max-frames", type=int, default=0, help="ограничить число кадров (отладка)"
    )
    extract.add_argument("--force", action="store_true", help="перезаписать tracks.json")
    extract.set_defaults(func=cmd_extract)

    label = subparsers.add_parser(
        "label", help="открыть браузер и разметить команды кликами"
    )
    label.add_argument("video", help="путь к видео или имя файла в data/raw")
    label.add_argument("--out-dir", default=None, help="папка результатов")
    label.add_argument("--tracks", default=None, help="путь к tracks.json")
    label.add_argument("--teams", default=None, help="путь к teams.json")
    label.add_argument("--port", type=int, default=8766, help="порт локального сервера")
    label.add_argument("--no-browser", action="store_true", help="не открывать браузер")
    label.add_argument("--stride", type=int, default=3, help="шаг keyframe игроков")
    label.add_argument("--disc-stride", type=int, default=1, help="шаг keyframe диска")
    label.add_argument("--bridge-gap", type=int, default=10, help="склеивать пропуски до N кадров")
    _add_label_options(label)
    label.set_defaults(func=cmd_label)

    export = subparsers.add_parser(
        "export", help="собрать annotations.xml без интерактивного UI"
    )
    export.add_argument("video", help="путь к видео или имя файла в data/raw")
    export.add_argument("--out-dir", default=None, help="папка результатов")
    export.add_argument("--tracks", default=None, help="путь к tracks.json")
    export.add_argument("--teams", default=None, help="путь к teams.json")
    export.add_argument("--xml", default=None, help="путь к выходному XML")
    export.add_argument("--stride", type=int, default=3, help="шаг keyframe игроков")
    export.add_argument("--disc-stride", type=int, default=1, help="шаг keyframe диска")
    export.add_argument("--bridge-gap", type=int, default=10, help="склеивать пропуски до N кадров")
    _add_label_options(export)
    export.set_defaults(func=cmd_export)

    import_cvat = subparsers.add_parser(
        "import-cvat", help="разобрать XML, выгруженный из CVAT, обратно в JSON/CSV"
    )
    import_cvat.add_argument("xml", help="annotations.xml из CVAT")
    import_cvat.add_argument("--out-dir", default=None, help="папка результатов")
    import_cvat.add_argument(
        "--per-frame",
        action="store_true",
        help="развернуть keyframe в покадровые рамки с интерполяцией",
    )
    import_cvat.add_argument("--teams-out", default=None, help="куда сохранить команды")
    import_cvat.add_argument("--player-label", default=LABEL_PLAYER)
    import_cvat.add_argument("--team-attr", default="team")
    import_cvat.set_defaults(func=cmd_import_cvat)

    info = subparsers.add_parser("info", help="показать треки и текущие команды")
    info.add_argument("video", help="путь к видео или имя файла в data/raw")
    info.add_argument("--out-dir", default=None)
    info.add_argument("--tracks", default=None)
    info.add_argument("--teams", default=None)
    info.set_defaults(func=cmd_info)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
