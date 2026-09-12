"""Метрическая ground truth и географическое разбиение RobotCar."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tgseqloc.data.robotcar import (
    Track,
    build_geographic_split,
    build_radius_positives,
    load_ins_track,
)


class TrackTests(unittest.TestCase):
    def track(self) -> Track:
        return Track((100, 200, 300), ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0)))

    def test_position_is_interpolated_between_samples(self) -> None:
        self.assertEqual(self.track().position_at(150), (5.0, 0.0))

    def test_exact_sample_is_returned(self) -> None:
        self.assertEqual(self.track().position_at(200), (10.0, 0.0))

    def test_frame_outside_the_log_has_no_position(self) -> None:
        """Экстраполяция здесь выдумала бы ground truth."""

        self.assertIsNone(self.track().position_at(50))
        self.assertIsNone(self.track().position_at(400))

    def test_unsorted_rows_are_ordered_on_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ins.csv"
            path.write_text(
                "timestamp,northing,easting\n300,10,10\n100,0,0\n200,10,0\n",
                encoding="utf-8",
            )
            track = load_ins_track(path)
            self.assertEqual(track.timestamps, (100, 200, 300))
            self.assertEqual(track.position_at(150), (5.0, 0.0))


class PositivesTests(unittest.TestCase):
    def test_every_database_frame_within_the_radius_is_a_positive(self) -> None:
        database = {0: (0.0, 0.0), 1: (10.0, 0.0), 2: (100.0, 0.0)}
        positives = build_radius_positives({7: (5.0, 0.0)}, database, radius=25.0)
        self.assertEqual(positives[7], [0, 1])

    def test_a_query_with_nothing_near_is_absent(self) -> None:
        """Отсутствие положительных — утверждение о покрытии, а не о файле."""

        positives = build_radius_positives({1: (0.0, 0.0)}, {0: (500.0, 0.0)}, 25.0)
        self.assertEqual(positives, {})

    def test_radius_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            build_radius_positives({}, {}, radius=0.0)


class SplitTests(unittest.TestCase):
    def positions(self) -> dict[int, tuple[float, float]]:
        # Прямой проезд на 900 м, затем возврат в начальную точку.
        forward = {i: (float(i * 100), 0.0) for i in range(10)}
        forward[9] = (0.0, 0.0)
        return forward

    def test_test_segment_comes_from_the_end_of_the_route(self) -> None:
        positions = {i: (float(i * 100), 0.0) for i in range(10)}
        positives = {i: [0] for i in positions}
        split = build_geographic_split(positions, positives, test_ratio=0.2,
                                       validation_ratio=0.0, radius=25.0)
        self.assertEqual(split["test_query_indices"], [8, 9])
        self.assertEqual(split["excluded_test_indices"], [])

    def test_a_revisit_is_dropped_from_the_test_set(self) -> None:
        """Маршрут возвращается на пройденное; иначе тест отвечался бы обучением."""

        positions = self.positions()
        positives = {i: [0] for i in positions}
        split = build_geographic_split(positions, positives, test_ratio=0.2,
                                       validation_ratio=0.0, radius=25.0)
        self.assertEqual(split["excluded_test_indices"], [9])
        self.assertEqual(split["test_query_indices"], [8])

    def test_validation_is_its_own_segment_between_training_and_test(self) -> None:
        positions = {i: (float(i * 100), 0.0) for i in range(20)}
        positives = {i: [0] for i in positions}
        split = build_geographic_split(positions, positives, test_ratio=0.2,
                                       validation_ratio=0.2, radius=25.0)
        train, validation, test = (split[f"{part}_query_indices"]
                                   for part in ("train", "validation", "test"))
        self.assertEqual(train, list(range(12)))
        self.assertEqual(validation, [12, 13, 14, 15])
        self.assertEqual(test, [16, 17, 18, 19])
        self.assertTrue(set(train).isdisjoint(validation))
        self.assertTrue(set(validation).isdisjoint(test))

    def test_a_revisit_is_dropped_from_validation_too(self) -> None:
        """Валидация выбирает эпоху, поэтому её отделяют тем же правилом, что и тест."""

        positions = {i: (float(i * 100), 0.0) for i in range(20)}
        positions[13] = (500.0, 0.0)   # возврат на землю обучения
        positions[17] = (1400.0, 0.0)  # возврат на землю валидации
        positives = {i: [0] for i in positions}
        split = build_geographic_split(positions, positives, test_ratio=0.2,
                                       validation_ratio=0.2, radius=25.0)
        self.assertEqual(split["excluded_validation_indices"], [13])
        self.assertEqual(split["excluded_test_indices"], [17])
        self.assertNotIn(13, split["validation_query_indices"])
        self.assertNotIn(17, split["test_query_indices"])

    def test_ratios_must_leave_room_for_training(self) -> None:
        positions = {i: (float(i * 100), 0.0) for i in range(10)}
        with self.assertRaises(ValueError):
            build_geographic_split(positions, {i: [0] for i in positions},
                                   test_ratio=0.6, validation_ratio=0.4, radius=25.0)

    def test_queries_without_positives_are_not_scored(self) -> None:
        positions = {i: (float(i * 100), 0.0) for i in range(10)}
        positives = {i: [0] for i in positions if i % 2 == 0}
        split = build_geographic_split(positions, positives, test_ratio=0.2,
                                       validation_ratio=0.0, radius=25.0)
        self.assertEqual(split["test_query_indices"], [8])

    def test_radius_is_recorded_in_the_split(self) -> None:
        positions = {i: (float(i * 100), 0.0) for i in range(10)}
        split = build_geographic_split(positions, {i: [0] for i in positions},
                                       test_ratio=0.2, validation_ratio=0.0, radius=30.0)
        self.assertEqual(split["split_radius_m"], 30.0)


class FrameListTests(unittest.TestCase):
    """Какие кадры входят в эксперимент — записанный список, а не шаг по каталогу."""

    def test_comments_blank_lines_and_role_prefixes(self) -> None:
        from tgseqloc.data.robotcar import load_frame_list

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frames.txt"
            path.write_text("# подмножество\n\n1439388259735205\nquery/1441186652690741\n",
                            encoding="utf-8")
            self.assertEqual(load_frame_list(path),
                             frozenset({"1439388259735205", "1441186652690741"}))

    def test_an_empty_list_is_an_error(self) -> None:
        """Пустой список означает ошибку в конфиге, а не просьбу о нуле кадров."""

        from tgseqloc.data.robotcar import load_frame_list

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frames.txt"
            path.write_text("# пусто\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_frame_list(path)

    def test_discovery_keeps_only_listed_frames_with_contiguous_indices(self) -> None:
        """Индексы идут подряд: по ним адресуются положительные пары."""

        from tgseqloc.data.robotcar import discover_robotcar_records

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "graphs" / "base").mkdir(parents=True)
            for stem in ("100", "200", "300"):
                (root / "ocr" / "t1" / stem).mkdir(parents=True)
                (root / "graphs" / "base" / f"{stem}.json").write_text("{}", encoding="utf-8")
            ocr, graphs = str(root / "ocr" / "{traversal}"), str(root / "graphs" / "{sequence}")
            everything = discover_robotcar_records(ocr, graphs, {"base": "t1"})
            listed = discover_robotcar_records(ocr, graphs, {"base": "t1"}, frame_list={"100", "300"})
            self.assertEqual([r.stem for r in everything], ["100", "200", "300"])
            self.assertEqual([r.stem for r in listed], ["100", "300"])
            self.assertEqual([r.index for r in listed], [0, 1])


if __name__ == "__main__":
    unittest.main()
