from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET

from ultitrack.cvat.teams import TeamBindings
from ultitrack.cvat.tracks import (
    LABEL_DISC,
    LABEL_PLAYER,
    TEAM_A,
    TEAM_B,
    TEAM_UNKNOWN,
    Track,
    TrackSet,
    VideoMeta,
)
from ultitrack.cvat.xml_io import (
    ParsedBox,
    ParsedTrack,
    expand_present_boxes,
    parse_cvat_video_xml,
    write_cvat_video_xml,
    write_import_teams,
)


def make_track(
    track_id: int,
    label: str = LABEL_PLAYER,
    frames: range | list[int] = range(0, 6),
    team: str = TEAM_UNKNOWN,
    box: tuple[float, float, float, float] = (10.0, 20.0, 50.0, 90.0),
) -> Track:
    data = {
        int(frame): (box[0], box[1], box[2], box[3], 0.9)
        for frame in frames
    }
    return Track(id=track_id, label=label, frames=data, team=team)


def make_track_set(total_frames: int = 120) -> TrackSet:
    meta = VideoMeta(
        video="/tmp/r1.mp4",
        video_name="r1.mp4",
        width=1920,
        height=1080,
        fps=30.0,
        total_frames=total_frames,
        model="ulti.pt",
        conf=0.15,
        disc_conf=0.22,
        imgsz=1280,
        field_filter=True,
        field_margin=1.0,
    )
    return TrackSet(meta=meta, tracks=[])


class CvatXmlWriteTest(unittest.TestCase):
    def test_roundtrip_labels_team_and_boxes(self) -> None:
        track_set = make_track_set(60)
        track_set.tracks = [
            make_track(1, frames=range(0, 10), team=TEAM_A),
            make_track(2, frames=range(0, 4), team=TEAM_B),
            make_track(0, label=LABEL_DISC, frames=range(0, 3)),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotations.xml"
            write_cvat_video_xml(
                track_set, path, stride=3, disc_stride=1, bridge_gap=2
            )
            root = ET.parse(path).getroot()
            self.assertEqual(root.findtext("version"), "1.1")
            self.assertEqual(root.findtext("meta/task/size"), "60")
            self.assertEqual(
                [el.text for el in root.findall("meta/task/labels/label/name")],
                ["player", "disc"],
            )
            values = root.findtext("meta/task/labels/label/attributes/attribute/values")
            self.assertEqual(values, "A\nB\nunknown")

            tracks = root.findall("track")
            self.assertEqual([el.get("id") for el in tracks], ["0", "1", "2"])
            player = next(el for el in tracks if el.get("id") == "1")
            self.assertEqual(player.get("label"), "player")
            self.assertEqual(player.get("source"), "auto")
            frames = [int(box.get("frame")) for box in player.findall("box")]
            self.assertEqual(frames[0], 0)
            self.assertEqual(frames[-1], 10)  # outside на 10-м кадре
            self.assertEqual(player.findall("box")[-1].get("outside"), "1")
            self.assertEqual(
                player.find("box/attribute").text, TEAM_A
            )

            parsed = parse_cvat_video_xml(path)
            self.assertEqual(parsed.total_frames, 60)
            self.assertEqual(parsed.width, 1920)
            by_id = {track.id: track for track in parsed.tracks}
            self.assertEqual(by_id[1].team, TEAM_A)
            self.assertEqual(by_id[2].team, TEAM_B)
            present = [box for box in by_id[1].boxes if not box.outside]
            self.assertEqual([box.frame for box in present], [0, 3, 6, 9])
            self.assertEqual(by_id[1].keyframe_count, 4)

    def test_long_gap_is_closed_and_reopened(self) -> None:
        track_set = make_track_set(40)
        track_set.tracks = [
            make_track(1, frames=[0, 1, 2, 20, 21], team=TEAM_A),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotations.xml"
            write_cvat_video_xml(track_set, path, stride=1, bridge_gap=3)
            parsed = parse_cvat_video_xml(path)
            boxes = parsed.tracks[0].boxes
            self.assertEqual(boxes[3].frame, 3)
            self.assertTrue(boxes[3].outside)
            self.assertEqual(boxes[4].frame, 20)
            self.assertFalse(boxes[4].outside)
            self.assertTrue(boxes[-1].outside)
            self.assertEqual(boxes[-1].frame, 22)

    def test_excluded_track_is_not_exported(self) -> None:
        track_set = make_track_set(20)
        excluded = make_track(1, frames=range(0, 5), team=TEAM_A)
        excluded.excluded = True
        track_set.tracks = [excluded, make_track(2, frames=range(0, 5), team=TEAM_B)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotations.xml"
            write_cvat_video_xml(track_set, path)
            root = ET.parse(path).getroot()
            self.assertEqual([track.get("id") for track in root.findall("track")], ["2"])

    def test_short_gap_is_interpolated(self) -> None:
        track_set = make_track_set(20)
        track_set.tracks = [
            make_track(1, frames=[0, 1, 2, 4, 5], team=TEAM_A),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotations.xml"
            write_cvat_video_xml(track_set, path, stride=1, bridge_gap=2)
            parsed = parse_cvat_video_xml(path)
            boxes = parsed.tracks[0].boxes
            # Внутри короткого пропуска outside быть не должно; закрытие
            # в конце ролика — ожидаемое.
            self.assertFalse(any(box.outside for box in boxes if box.frame <= 5))
            self.assertEqual([box.frame for box in boxes], [0, 1, 2, 4, 5, 6])
            self.assertTrue(boxes[-1].outside)

    def test_expand_present_boxes_interpolates(self) -> None:
        track_set = make_track_set(20)
        track_set.tracks = [
            Track(
                id=1,
                label=LABEL_PLAYER,
                team=TEAM_A,
                frames={
                    0: (0.0, 0.0, 10.0, 10.0, 0.9),
                    5: (10.0, 0.0, 20.0, 10.0, 0.9),
                },
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotations.xml"
            write_cvat_video_xml(track_set, path, stride=5, bridge_gap=10)
            parsed = parse_cvat_video_xml(path)
            expanded = expand_present_boxes(parsed.tracks[0])
            self.assertEqual([frame for frame, _ in expanded], [0, 1, 2, 3, 4, 5])
            self.assertAlmostEqual(expanded[3][1][0], 6.0)

    def test_expand_holds_box_until_outside(self) -> None:
        track = ParsedTrack(
            id=1,
            label=LABEL_PLAYER,
            team=TEAM_A,
            boxes=[
                ParsedBox(0, (0.0, 0.0, 10.0, 10.0)),
                ParsedBox(3, (0.0, 0.0, 10.0, 10.0), outside=True),
                ParsedBox(6, (20.0, 0.0, 30.0, 10.0)),
                ParsedBox(9, (20.0, 0.0, 30.0, 10.0), outside=True),
            ],
        )
        expanded = expand_present_boxes(track, 20)
        self.assertEqual([frame for frame, _ in expanded], [0, 1, 2, 6, 7, 8])

    def test_expand_extends_last_box_to_end(self) -> None:
        track = ParsedTrack(
            id=1,
            label=LABEL_PLAYER,
            team=TEAM_A,
            boxes=[
                ParsedBox(0, (0.0, 0.0, 10.0, 10.0)),
                ParsedBox(5, (20.0, 0.0, 30.0, 10.0)),
            ],
        )
        expanded = expand_present_boxes(track, 10)
        self.assertEqual([frame for frame, _ in expanded], list(range(10)))
        self.assertAlmostEqual(expanded[7][1][0], 20.0)

    def test_import_teams(self) -> None:
        track_set = make_track_set(30)
        track_set.tracks = [
            make_track(1, team=TEAM_A),
            make_track(2, team=TEAM_B),
            make_track(0, label=LABEL_DISC, frames=range(0, 2)),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotations.xml"
            write_cvat_video_xml(track_set, path)
            parsed = parse_cvat_video_xml(path)
            teams_path = Path(tmp) / "teams_from_cvat.json"
            write_import_teams(parsed, teams_path)
            bindings = TeamBindings.load(teams_path)
            self.assertEqual(bindings.assignments[1], TEAM_A)
            self.assertEqual(bindings.assignments[2], TEAM_B)
            self.assertEqual(bindings.manual, {1, 2})


if __name__ == "__main__":
    unittest.main()
