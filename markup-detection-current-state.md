# Survey Markup Detection — Current State & Design Review Request

**Status:** working but not reliable enough to ship. Seeking a second opinion on
whether to keep iterating on the current geometric approach or switch strategy.

**Branch:** `feature/auto-ocr` (uncommitted). **Test doc:** `split_1_2.pdf`
(2-page German population survey, ~170 marks). **Container:** `text_lab.sif`
(paddleocr 3.7 / paddlex 3.7.2 in `paddle_vl_backend`). GPU node has an RTX 4090.

---

## 1. Context — the automatic OCR pipeline this lives in

Text Lab's OCR page was reworked into a single automatic pipeline (no engine
picker). Flow:

```
upload → auto_ocr.process_document() → typed IR (doc_ir.Document) → UI tabs / downloads
```

- **Routing** (`src/core/auto_ocr.py`): born-digital PDF pages with a real text
  layer and no math take a native PyMuPDF fast lane; scanned pages, images, and
  math pages go to the **PaddleOCR-VL** worker (`src/core/paddle_vl_worker.py`,
  runs in an isolated conda env via subprocess) rasterised at 200 DPI.
- **IR** (`src/core/doc_ir.py`): `Document → Page → Region`, region types
  (text/title/table/figure/formula/checkbox/…), exporters (markdown, JSON,
  per-table CSV, ZIP bundles).
- **Markup detection** (`src/core/markup_detect.py`): the subject of this doc.

The rest of the pipeline (layout, tables, figures, equations, markdown/JSON
export) works well. **Only survey-mark (checkbox/circle) reading is unreliable.**

---

## 2. The problem

The surveys use **printed circles or boxes** as options; respondents answer with
a **hand-drawn X** (or filled box `☒`, or a tick). Examples from the real doc:

- A 1–6 rating scale: a row of 11 printed circles `○`, one crossed.
- `○ Ja   ⊗ Nein` — two circles, one crossed.
- A matrix (`p2_r8`): ~96 circles in a grid, one crossed per row.
- Checkbox lists: `☐ … ☒ …`.

**What the VL model does:** PaddleOCR-VL transcribes the *content* of a region
(often a table or text block), and *usually* renders marks as glyphs inside that
content — `☒`/`☐`, `✗`/`○`. It gets **most** marks right on its own. But it
frequently transcribes a **crossed** circle as a plain `○` (drops the mark), and
occasionally the reverse. It does **not** emit dedicated `checkbox` layout
regions for these — the marks are buried in table/text content.

**Goal:** recover the marks the VL model drops, without corrupting the ones it
got right, and flag anything uncertain for human review.

---

## 3. Approach taken (geometry + reconciliation)

Because marks are buried in table/text content, the pipeline:

1. **Glyph scan** (`extract_mark_glyphs`): pull every mark glyph (`○☐☒✗…`) out of
   a region's transcribed content, in reading order, with a provisional state.
2. **Geometric mark finder** (`find_marks`): crop the region from the 200 DPI
   raster, locate the individual printed circles/boxes via OpenCV contours, and
   classify each (checked/unchecked/uncertain).
3. **Reconcile** (`reconcile_marks`): align the two lists positionally. Only
   override in the **unchecked→checked** direction (a VL `○` with confident ink
   under it = a dropped cross). A VL `☒` is never downgraded.
4. **Rewrite content** (`apply_states_to_content`): if geometry finds a dropped
   cross, rewrite the glyph in the region content to `☒` so every downstream
   export (markdown, table, CSV, JSON) carries the correction.
5. **Surface uncertainty**: uncertain marks and count-mismatch regions are
   flagged in the Markup tab with evidence crops; nothing is silently dropped.

### Single-mark classifier (`detect_markup_geometric`)

Interior **fill ratio** (fraction of ink inside the mark, border trimmed) plus a
**Hough line "strike" score** (a thin pen X has a tiny fill ratio but a long
crossing stroke). Thresholds (empirically hand-tuned):
`CHECKED_FILL=0.18`, `UNCHECKED_FILL=0.045`, `STRIKE_CHECKED=0.45`,
`STRIKE_AMBIGUOUS=0.30`, `INTERIOR_MARGIN=0.24`.

### Mark finder heuristics (`find_marks`)

- contour bbox size/aspect gates; contour must be a closed outline (area ≥ 30%
  of bbox);
- **shape filter**: must look like a box (ink on all 4 bbox sides) or a circle
  (contour points lie on the min-enclosing circle);
- **isolation**: reject shapes with ink flush left/right (letters inside words);
- **dominant alignment group**: keep only the largest subset sharing an x-column
  or y-row (survey marks are laid out in regular rows/columns; stray letters are
  scattered);
- **gap recovery**: a heavy X destroys the printed circle's outline, so the
  *marked* option is the one that vanishes from detection. With n−1 marks found
  in a regular pattern and one double-width interior gap, synthesize the missing
  slot and classify it.
- **saturation guard**: if geometry claims *every* mark in a ≥4-mark region is
  inked, it's non-discriminative — discard and keep the VL transcription.

---

## 4. What was tried, in order (on the real doc)

| Iteration | Change | Overrides | False overrides | Uncertain |
|---|---|---|---|---|
| v1 | fill-ratio only, fixed threshold | — | — | many |
| v2 | + stroke-aware (Hough), glyph scan, reconciliation | **32** | ~all | 69 |
| v3 | + background-anchored binarization (not Otsu), min-ink floor, saturation guard | 5 | 5 (letters `g`,`r`,`6`) | 23 |
| v4 | + shape filter (box/circle), dominant-alignment group, gap recovery, wider isolation | **2** | **0 (both verified real)** | 25 |

The trajectory is real progress on **false positives** (32 → 0) but it exposed
that the remaining problem is **misses**, and the classifier is fragile.

### Key finding: Otsu was catastrophic on real scans
The first "almost everything became crossed" failure was **Otsu thresholding**:
on a sparse, noisy scan crop Otsu bisects the *paper texture* into "ink", so
every empty circle read as filled. Fixed by anchoring the dark threshold to the
estimated paper background (high percentile of gray) + a minimum-ink floor.

### Key finding: extreme size/resolution sensitivity
Marks are ~11–15 px. The interior fill ratio is dominated by the **printed ring
itself** at small sizes. Measuring the same empty circles on the *downscaled
preview* raster gives `fill≈0.25–0.37` → "checked"; on the 200 DPI raster they
read differently again. The metric has no stable operating point across
mark size / scan resolution — this is the core fragility.

---

## 5. Current results on `split_1_2.pdf` (v4)

- 2 geometric overrides, **both genuine** crossed circles the VL model dropped
  (verified by eye) — the original bug ("crosses not returned") is fixed for
  those.
- 0 false overrides.
- 25 marks flagged uncertain.
- **But** the debug overlay (`tests/validate_real_doc.py` →
  `split_1_2_validation/page_N_marks.png`, every located mark boxed and coloured)
  shows two persistent problems (§6).

---

## 6. Unsolved / unaddressed failure modes

1. **Destroyed-outline crosses (the important one).** A heavy hand-drawn X breaks
   the printed circle's closed contour, so `find_marks` (which requires a closed
   circle/box shape) filters the *marked* option out entirely. Both the VL model
   and the geometry can miss the same mark. In the overlay, the actual answer in
   Q1 (crossed "4"), Q2 and Q3 (crossed "Nein") has **no box at all**. Gap
   recovery only fires with a clean n−1 pattern and a single interior gap — it
   fails for **edge positions** and **small groups** (Ja/Nein has no pattern to
   recover from).

2. **Empty vs marked at small sizes.** Empty circles frequently land as
   "uncertain" (orange in the overlay) instead of confident "unchecked", because
   fill conflates the ring with ink and the Hough strike fires on ring arcs.
   Noisy and threshold-dependent.

3. **Changed answers / deletions — not handled at all.** When a respondent marks
   one option, crosses it out, and marks another, we have no notion of a
   *cancelled* mark. Both would register as checked. Distinguishing a struck-out
   answer (scribble / multiple cancellation strokes) from an active one is a
   semantic problem the current geometry doesn't model.

4. **No glyph anchor.** If the VL model transcribes a circle as the letter "O" or
   drops the glyph entirely, `find_marks` never runs (we only process
   glyph-bearing regions), so those marks are invisible to the geometric layer.

5. **Big grids.** The ~96-mark matrix (`p2_r8`) goes count-mismatch (geometry vs
   transcription counts disagree) → no override, relies entirely on VL.

6. **VL non-determinism.** The same region transcribes `☒` in one run and `□` in
   another, so the glyph baseline itself shifts between runs.

---

## 7. Options on the table (want Codex's read)

- **A. Positive X-detection on a reconstructed grid.** Extrapolate the full
  option grid from the *clean* circles (pitch + extent), then at each slot —
  including where the circle was destroyed — test for crossing pen strokes /
  excess ink. More robust to destroyed outlines than gap recovery, but complex
  and false-positive-prone near handwriting/signatures.
- **B. Dedicated form/checkbox model.** A small trained checkbox-state detector
  (YOLO-ish), or a layout model with a checkbox-state head. Robust but needs
  training data / a model to source.
- **C. Targeted VL re-prompt per survey region.** Crop each mark-bearing region
  and ask the VLM specifically "which options are marked?" — leans on the model
  that already gets most of them, with a focused prompt instead of relying on
  incidental transcription. Extra inference cost; still model-dependent.
- **D. Template registration.** If surveys come from known blank templates,
  register blank-vs-filled and diff. Very robust, but requires templates.
- **E. Stop overriding; trust VL + human review.** Drop the geometric override
  entirely, keep the glyph transcription as source of truth, and invest in the
  review UI (evidence crops, uncertain flags). Simplest and honest; accepts that
  some crosses are missed.
- **F. Explicit "Survey mode" toggle.** Only run the purpose-built form pipeline
  when the user declares the document is a survey, so the general OCR path stays
  clean.

My lean: **C or E** for the near term (the VL model already gets most marks; the
geometric layer keeps trading FPs for misses), with **A** only if geometry must
stay. But this is exactly what I'd like a second opinion on.

---

## 8. Files & how to reproduce

Code:
- `src/core/markup_detect.py` — glyph scan, geometric classifier, mark finder,
  reconciliation (the detector).
- `src/core/auto_ocr.py` — `_apply_markup()` wires it in; `debug_dir` writes the
  audit overlay; math routing; native/VL lanes.
- `src/core/paddle_vl_worker.py` — PaddleOCR-VL subprocess worker.
- `src/core/doc_ir.py` — IR + exporters.
- `src/pages/OCR.py` — UI (Markup tab shows per-region marks + evidence crops).

Tests (`tests/`, run with cv2/lxml on path via `TEXTLAB_TESTDEPS`):
- `test_markup.py`, `test_noisy_scan.py`, `test_regression.py` — all green, but
  note they use **synthetic** marks with clean backgrounds, which is exactly why
  they missed the Otsu and size-sensitivity failures. Synthetic coverage is not
  representative of real scans.
- `validate_real_doc.py` — runs the real pipeline on a PDF, prints a per-region
  markup report, and writes `page_N_marks.png` overlays (every located mark,
  coloured by state) + flagged-mark evidence crops.

Run the real-doc audit (GPU node, inside the container):

```bash
apptainer exec --nv --bind /storage:/storage \
  --env PADDLE_PDX_CACHE_HOME=/storage/research/dsl_shared/solutions/ondemand/text_lab/container/models/paddlex \
  --env PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True --env DISABLE_MODEL_SOURCE_CHECK=True \
  ./text_lab.sif /opt/conda/envs/text_lab_main/bin/python tests/validate_real_doc.py split_1_2.pdf
# → split_1_2_validation/page_N_marks.png  (green=checked, blue=unchecked, orange=uncertain, OVR=corrected)
```

The overlay is the fastest way to see the state: correct greens/blues, but empty
circles bleeding to orange, and the genuinely crossed answers with **no box**.
