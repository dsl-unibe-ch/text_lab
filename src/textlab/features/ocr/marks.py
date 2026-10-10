"""Checkbox states and response marks on a recognized page.

Checkbox regions are classified, and mark glyphs (such as ``[x]`` or
ticks) in text and tables are checked against the ink on the page, using
:mod:`textlab.features.ocr.markup_detect`. A debug overlay shows every mark
found, for validating the detector.
"""

from __future__ import annotations

from textlab.features.ocr import doc_ir, markup_detect
from textlab.features.ocr.rasters import crop_region, encode_png_b64


def apply_markup(page: doc_ir.Page, page_bgr=None, debug_collect=None):
    """Classify checkbox regions and verify mark glyphs against page ink.

    ``page_bgr`` and region boxes must use full-resolution coordinates.
    Located marks are appended to *debug_collect* when supplied.
    """
    for region in page.regions:
        if region.type == doc_ir.CHECKBOX:
            block_text = (
                region.content.get("text")
                or region.content.get("markdown")
                or ""
            )
            crop_b64 = (region.asset or {}).get("b64")
            crop = markup_detect.decode_crop_b64(crop_b64)
            if crop is None and page_bgr is not None and len(region.bbox) >= 4:
                crop = crop_region(page_bgr, region.bbox)
            verdict = markup_detect.classify_checkbox(block_text, crop)
            region.markup = verdict
            if (
                verdict.get("state") == "uncertain"
                and "unverified checkbox state" not in region.warnings
            ):
                region.warnings.append("unverified checkbox state")
            if verdict.get("status") == "geometry_disagreement":
                region.warnings.append("checkbox OCR and geometry disagree")
            if debug_collect is not None and len(region.bbox) >= 4:
                debug_collect.append(
                    {
                        "bbox": [int(v) for v in region.bbox[:4]],
                        "state": verdict.get("state", "uncertain"),
                        "override": False,
                        "region_id": region.id,
                    }
                )
            continue

        # Glyph-borne marks inside tables / text
        source_text = (
            region.content.get("html")
            if region.type == doc_ir.TABLE
            else (
                region.content.get("text")
                or region.content.get("markdown")
                or ""
            )
        ) or ""
        glyph_items = markup_detect.extract_mark_glyphs(source_text)
        if not glyph_items:
            continue

        geo_marks = []
        crop = None
        origin = (0, 0)
        if page_bgr is not None and len(region.bbox) >= 4:
            origin = (
                max(0, int(round(region.bbox[0]))),
                max(0, int(round(region.bbox[1]))),
            )
            crop = crop_region(page_bgr, region.bbox)
            if crop is not None:
                geo_marks = markup_detect.find_marks(
                    crop, n_expected=len(glyph_items)
                )

        items, status = markup_detect.reconcile_marks(glyph_items, geo_marks)
        counts = markup_detect.summarize_items(items)

        if debug_collect is not None:
            _collect_mark_debug(
                debug_collect, region, origin, geo_marks, items, status
            )

        # Evidence thumbnails for marks a human should be able to verify at a
        # glance: geometry disagreements and uncertain states.
        for item in items:
            geo_bbox = item.pop("geo_bbox", None)
            if crop is None or not geo_bbox:
                continue
            if item.get("needs_review") or item.get("state") == "uncertain":
                x1, y1, x2, y2 = [int(v) for v in geo_bbox]
                sub = crop[max(0, y1) : y2, max(0, x1) : x2]
                if sub.size:
                    item["crop_b64"] = encode_png_b64(sub)

        region.markup = {
            "kind": "glyph-marks",
            "status": status,
            "items": [
                {
                    k: item.get(k)
                    for k in (
                        "glyph",
                        "state",
                        "method",
                        "score",
                        "geometry",
                        "needs_review",
                        "crop_b64",
                    )
                }
                for item in items
            ],
            **counts,
        }
        if status == "count_mismatch":
            region.warnings.append(
                "mark count mismatch between transcription and geometry"
            )
        if status == "geometry_saturated":
            region.warnings.append(
                "geometry found ink in every mark (non-discriminative); "
                "overrides skipped"
            )
        if status == "geometry_disagreement":
            region.warnings.append(
                "mark transcription and geometry disagree; OCR left unchanged"
            )
        if counts["n_uncertain"]:
            region.warnings.append("uncertain mark state(s)")


_DEBUG_STATE_COLORS = {  # BGR
    "checked": (0, 170, 0),
    "unchecked": (200, 130, 0),
    "uncertain": (0, 140, 255),
}


def _collect_mark_debug(
    debug_collect, region, origin, geo_marks, items, status
):
    """Record every mark found by geometry, in page coordinates."""
    ox, oy = origin
    # When reconciliation aligned marks 1:1, colour by the final (possibly
    # overridden) state; otherwise fall back to the raw geometric verdict.
    aligned = status == "matched" and len(items) == len(geo_marks)
    for i, mark in enumerate(geo_marks):
        gb = mark.get("bbox")
        if not gb:
            continue
        if aligned:
            state = items[i].get("state", mark["state"])
            override = items[i].get("method") == "geometric-override"
        else:
            state, override = mark["state"], False
        debug_collect.append(
            {
                "bbox": [
                    ox + int(gb[0]),
                    oy + int(gb[1]),
                    ox + int(gb[2]),
                    oy + int(gb[3]),
                ],
                "state": state,
                "override": override,
                "region_id": region.id,
            }
        )


def write_mark_debug_overlay(page_bgr, page, debug_marks, out_path):
    """Draw region boxes + every located mark (coloured by state) and save."""
    try:
        import cv2
    except Exception:
        return
    if page_bgr is None:
        return
    vis = page_bgr.copy()
    for region in page.regions:
        if not (region.markup and len(region.bbox) >= 4):
            continue
        x1, y1, x2, y2 = [int(v) for v in region.bbox[:4]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (170, 170, 170), 1)
        cv2.putText(
            vis,
            region.id,
            (x1, max(10, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )
    for mark in debug_marks:
        x1, y1, x2, y2 = mark["bbox"]
        color = _DEBUG_STATE_COLORS.get(mark["state"], (0, 140, 255))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        if mark.get("override"):
            cv2.putText(
                vis,
                "OVR",
                (x1, max(9, y1 - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )
    cv2.imwrite(str(out_path), vis)
