"""Joining detected text pieces into lines, reading order and crops."""

import pytest

from textlab.features.handwriting import lines


def test_polygon_box_is_the_bounding_box():
    points = [[10, 5], [50, 8], [48, 30], [12, 28]]
    assert lines.polygon_box(points) == (10, 5, 50, 30)
    with pytest.raises(ValueError):
        lines.polygon_box([])


def test_pieces_on_one_line_are_joined():
    pieces = [(200, 10, 300, 40), (10, 12, 180, 42), (320, 11, 400, 39)]
    assert lines.group_lines(pieces) == [(10, 10, 400, 42)]


def test_lines_above_each_other_stay_apart_and_are_ordered():
    pieces = [(10, 100, 300, 130), (10, 10, 300, 40), (10, 55, 300, 85)]
    assert lines.group_lines(pieces) == [
        (10, 10, 300, 40),
        (10, 55, 300, 85),
        (10, 100, 300, 130),
    ]


def test_a_slanted_line_joins_piece_by_piece():
    # Each piece is a little lower than the one before; the first and the
    # last do not overlap, but neighbours do.
    pieces = [(0, 0, 100, 30), (110, 10, 210, 40), (220, 20, 320, 50)]
    assert lines.group_lines(pieces) == [(0, 0, 320, 50)]


def test_a_wide_gap_starts_a_new_line_read_left_to_right():
    # A marginal note far to the right of the body text, at its height.
    pieces = [(900, 12, 1000, 40), (10, 10, 300, 40)]
    assert lines.group_lines(pieces) == [
        (10, 10, 300, 40),
        (900, 12, 1000, 40),
    ]


def test_tiny_boxes_are_dropped():
    pieces = [(10, 10, 300, 40), (500, 500, 503, 502)]
    assert lines.group_lines(pieces) == [(10, 10, 300, 40)]
    assert lines.group_lines([]) == []


def test_crop_box_adds_a_margin_within_the_page():
    assert lines.crop_box((10, 20, 110, 70), 1000, 1000) == (5, 15, 115, 75)
    assert lines.crop_box((0, 0, 50, 20), 52, 21) == (0, 0, 52, 21)


def test_fit_pixels_keeps_the_aspect_ratio():
    assert lines.fit_pixels(100, 50, None) == (100, 50)
    assert lines.fit_pixels(100, 50, 10_000) == (100, 50)
    width, height = lines.fit_pixels(4000, 200, 262_144)
    assert width * height <= 262_144
    assert abs(width / height - 20) < 0.1
