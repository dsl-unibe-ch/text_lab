"""From detected line polygons to boxes and crops."""

import pytest

from textlab.features.handwriting import lines


def test_polygon_box_is_the_bounding_box():
    points = [[10, 5], [50, 8], [48, 30], [12, 28]]
    assert lines.polygon_box(points) == (10, 5, 50, 30)
    with pytest.raises(ValueError):
        lines.polygon_box([])


def square(left, top, right, bottom):
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


def test_line_boxes_keep_the_detectors_reading_order():
    # kraken sorts the lines; a marginal note may come before a lower line.
    polygons = [
        square(10, 100, 300, 130),
        square(900, 12, 1000, 40),
        square(10, 10, 300, 40),
    ]
    assert lines.line_boxes(polygons) == [
        (10, 100, 300, 130),
        (900, 12, 1000, 40),
        (10, 10, 300, 40),
    ]


def test_tiny_and_empty_lines_are_dropped():
    polygons = [square(10, 10, 300, 40), square(500, 500, 503, 502), []]
    assert lines.line_boxes(polygons) == [(10, 10, 300, 40)]
    assert lines.line_boxes([]) == []


def test_crop_box_adds_a_margin_within_the_page():
    assert lines.crop_box((10, 20, 110, 70), 1000, 1000) == (5, 15, 115, 75)
    assert lines.crop_box((0, 0, 50, 20), 52, 21) == (0, 0, 52, 21)


def test_fit_pixels_keeps_the_aspect_ratio():
    assert lines.fit_pixels(100, 50, None) == (100, 50)
    assert lines.fit_pixels(100, 50, 10_000) == (100, 50)
    width, height = lines.fit_pixels(4000, 200, 262_144)
    assert width * height <= 262_144
    assert abs(width / height - 20) < 0.1
