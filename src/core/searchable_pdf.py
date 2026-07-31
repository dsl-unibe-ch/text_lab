"""Searchable-PDF export: the original pages with an invisible text layer.

PaddleOCR-VL returns one bounding box per laid-out *block*, which is too coarse
to drive a usable text layer — selecting a word would highlight a whole
paragraph. Tesseract's ``image_to_data`` is the only engine in the container
that reports per-word geometry, so it runs here as a **geometry-only second
pass**: its boxes are used, its transcription is not.

The authoritative text stays the VL lane's, matching the rest of the exports
(``.md`` / ``.docx`` / ``.txt`` / ``.json`` all agree with the PDF). The two
transcriptions never match character-for-character, so :func:`align_tokens`
maps VL tokens onto Tesseract boxes and falls back to a coarser box wherever
the two disagree — search always works, only the highlight gets less precise.

The pass runs during ``auto_ocr.process_document`` because it needs the
full-resolution rasters, which live in the job workspace and are deleted
afterwards; ``Region.bbox`` is rescaled to preview space once the page is
finalised.
"""

from __future__ import annotations

import difflib
import io
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:  # pragma: no cover - exercised only by container availability
    from core import doc_ir
except ImportError:  # pragma: no cover
    import doc_ir  # type: ignore


#: Tesseract language codes installed in the container, keyed by UI language.
TESSERACT_LANGUAGES = {
    "English": "eng",
    "German": "deu",
    "French": "fra",
    "Italian": "ita",
    "Spanish": "spa",
}

DEFAULT_TESSERACT_LANG = "eng"

#: Word boxes below this Tesseract confidence are treated as noise.
MIN_WORD_CONFIDENCE = 30.0

#: Region types whose text belongs in the searchable layer.
#:
#: Formulas are excluded: the VL lane exports LaTeX, which is not what is
#: printed, so aligning ``E = mc^2`` to a rendered equation would index text the
#: reader cannot see. Tables *are* included — their cell values genuinely appear
#: on the page — but via :func:`region_layer_text`, which reads the cells rather
#: than the HTML wrapper.
LAYER_TYPES = {
    doc_ir.TEXT,
    doc_ir.TITLE,
    doc_ir.FOOTNOTE,
    doc_ir.HEADER,
    doc_ir.FOOTER,
    doc_ir.LIST,
    doc_ir.REFERENCE,
    doc_ir.TABLE,
}


def region_layer_text(region: "doc_ir.Region") -> str:
    """The printed text of *region*, as it should appear in the layer.

    Tables carry HTML, so the cells are flattened in reading order; every other
    layer type is already plain text.
    """
    if region.type != doc_ir.TABLE:
        return region.text
    try:
        frame = doc_ir.extract_html_table(region.content.get("html", ""))
    except Exception:
        frame = None
    if frame is None:
        return ""
    cells = _printed_header(frame)
    for row in frame.itertuples(index=False):
        cells.extend("" if value is None else str(value) for value in row)
    return " ".join(cell for cell in cells if cell and cell.lower() != "nan")


def _printed_header(frame) -> List[str]:
    """Column names, but only when they were really printed on the page.

    A table without ``<th>`` gets pandas' positional column names (``0``, ``1``,
    ... or ``Unnamed: 0``). Emitting those as layer tokens injects text that
    exists nowhere in the image, and because alignment is positional it shifts
    every following cell onto the wrong box — the whole table lands misplaced.
    """
    columns = list(frame.columns)
    if not columns:
        return []
    if all(isinstance(column, (int, bool)) or str(column).isdigit() for column in columns):
        return []
    return [
        str(column)
        for column in columns
        if str(column).strip() and not str(column).startswith("Unnamed:")
    ]


# ==========================================
#        TOKENISATION / ALIGNMENT
# ==========================================


def _normalise(token: str) -> str:
    """Fold a token to a comparable key (case, punctuation and accents aside)."""
    return re.sub(r"[^\w]", "", token, flags=re.UNICODE).lower()


#: Mark/box glyphs the VL lane transcribes inline on forms. They are not words,
#: nobody searches for them, and the base-14 PDF fonts cannot encode them — so
#: they are stripped from the layer rather than silently dropped by the writer.
_GLYPH_CHARS = "□■○●◯☐☑☒✓✔✗✘"


def tokenize(text: str) -> List[str]:
    """Split VL text into searchable tokens.

    Whitespace-delimited, with mark glyphs stripped and tokens that carry no
    word characters at all discarded: indexing "□" helps nobody, and a token the
    PDF font cannot encode would be dropped by the writer anyway, taking its
    box with it.
    """
    tokens = []
    for raw in re.split(r"\s+", text or ""):
        cleaned = raw.strip(_GLYPH_CHARS).strip()
        if cleaned and _normalise(cleaned):
            tokens.append(cleaned)
    return tokens


def align_tokens(
    vl_tokens: Sequence[str],
    tess_words: Sequence[Dict[str, Any]],
    fallback_bbox: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    """Map VL tokens onto Tesseract word boxes.

    ``tess_words`` are dicts with ``text`` and ``bbox`` ``[x1, y1, x2, y2]``.
    Matching runs on normalised keys via :class:`difflib.SequenceMatcher`, so
    ordinary OCR differences (``rn``/``m``, dropped umlauts, stray punctuation)
    only cost precision, never correctness:

    * equal runs   -> VL token placed at its Tesseract box (exact highlight);
    * replacements -> VL tokens spread across the boxes they displaced;
    * insertions   -> VL tokens with no ink to anchor to fall back to the
      surrounding box, so the text is still searchable.

    Returns ``[{"text", "bbox", "exact"}]``; ``exact`` marks a one-to-one box.
    """
    vl_tokens = list(vl_tokens)
    tess_words = list(tess_words)
    if not vl_tokens:
        return []
    if not tess_words:
        if fallback_bbox is None:
            return []
        return [
            {"text": " ".join(vl_tokens), "bbox": list(fallback_bbox), "exact": False}
        ]

    vl_keys = [_normalise(t) for t in vl_tokens]
    tess_keys = [_normalise(w.get("text", "")) for w in tess_words]

    placed: List[Dict[str, Any]] = []
    matcher = difflib.SequenceMatcher(a=vl_keys, b=tess_keys, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                word = tess_words[j1 + offset]
                placed.append(
                    {
                        "text": vl_tokens[i1 + offset],
                        "bbox": list(word["bbox"]),
                        "exact": True,
                        "line": word.get("line"),
                    }
                )
        elif tag in ("replace", "delete"):
            # VL says something the boxes disagree with (or Tesseract missed the
            # ink entirely). Emit the VL text across whatever boxes it displaced,
            # or the surrounding region when there are none.
            boxes = [w["bbox"] for w in tess_words[j1:j2]]
            span = _union(boxes) if boxes else _neighbour_box(tess_words, j1, fallback_bbox)
            if span is not None:
                line = tess_words[j1].get("line") if j1 < len(tess_words) else None
                placed.append(
                    {
                        "text": " ".join(vl_tokens[i1:i2]),
                        "bbox": list(span),
                        "exact": False,
                        "line": line,
                    }
                )
        # 'insert' = Tesseract found ink the VL lane did not transcribe. The VL
        # text is authoritative, so that ink is deliberately left unindexed.
    return placed


def _union(boxes: Sequence[Sequence[float]]) -> Optional[List[float]]:
    boxes = [b for b in boxes if b]
    if not boxes:
        return None
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def _neighbour_box(
    tess_words: Sequence[Dict[str, Any]],
    index: int,
    fallback_bbox: Optional[Sequence[float]],
) -> Optional[List[float]]:
    """Nearest real box to an unanchored token, else the region box."""
    for candidate in (index, index - 1, index + 1):
        if 0 <= candidate < len(tess_words):
            return list(tess_words[candidate]["bbox"])
    return list(fallback_bbox) if fallback_bbox is not None else None


# ==========================================
#        LANGUAGE DETECTION
# ==========================================


#: ISO-639-1 code per supported Tesseract pack, for the stopword lexicons.
_ISO_FOR_TESSERACT = {"eng": "en", "deu": "de", "fra": "fr", "ita": "it", "spa": "es"}

#: A clear stopword-ratio win needs no second opinion.
_STOPWORD_MARGIN = 0.04

_STOPWORD_CACHE: Dict[str, frozenset] = {}


def _stopwords(iso: str) -> frozenset:
    if iso not in _STOPWORD_CACHE:
        try:
            import stopwordsiso

            _STOPWORD_CACHE[iso] = frozenset(stopwordsiso.stopwords(iso))
        except Exception:
            _STOPWORD_CACHE[iso] = frozenset()
    return _STOPWORD_CACHE[iso]


def _stopword_scores(tokens: Sequence[str]) -> Dict[str, float]:
    """Fraction of *tokens* that are stopwords in each supported language."""
    if not tokens:
        return {}
    scores = {}
    for tess, iso in _ISO_FOR_TESSERACT.items():
        words = _stopwords(iso)
        if words:
            scores[tess] = sum(1 for t in tokens if t in words) / len(tokens)
    return scores


def detect_language(text: str, default: str = DEFAULT_TESSERACT_LANG) -> str:
    """Pick a Tesseract language code from the VL text.

    Deliberately constrained to the packs installed in the container: this is a
    five-way choice, not open-set language ID, which makes a stopword-frequency
    vote both accurate and deterministic. ``langdetect`` is consulted only to
    break a close call, and is ignored when it names a language we cannot run.

    Falls back to *default* on short or unrecognisable text, where guessing
    wrong costs more precision than the default does.
    """
    tokens = [t for t in re.findall(r"[^\W\d_]+", (text or "").lower(), re.UNICODE) if t]
    if len(tokens) < 20:  # too little evidence to beat the default
        return default

    scores = _stopword_scores(tokens)
    if not scores:
        return default
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best, best_score = ranked[0]
    if best_score < 0.05:  # no language's function words show up: not prose
        return default
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    if best_score - runner_up >= _STOPWORD_MARGIN:
        return best

    try:  # close call — let langdetect arbitrate, but only within our packs
        from langdetect import DetectorFactory, detect_langs

        DetectorFactory.seed = 0  # deterministic
        iso_to_tess = {v: k for k, v in _ISO_FOR_TESSERACT.items()}
        for guess in detect_langs(text):
            if guess.lang in iso_to_tess:
                return iso_to_tess[guess.lang]
    except Exception:
        pass
    return best


#: Language detection reads running prose only. Table cells are mostly numbers,
#: codes and one-word labels — weak evidence that skews a stopword vote.
PROSE_TYPES = LAYER_TYPES - {doc_ir.TABLE}


def document_language(document: "doc_ir.Document", default: str = DEFAULT_TESSERACT_LANG) -> str:
    """Detect once over the whole document's prose regions."""
    parts = [
        region.text
        for _, region in document.all_regions()
        if region.type in PROSE_TYPES and region.text.strip()
    ]
    return detect_language(" ".join(parts), default=default)


def page_language(page: "doc_ir.Page", default: str = DEFAULT_TESSERACT_LANG) -> str:
    """Detect from one page's own text.

    Detection runs per page because the word-geometry pass is per page: a
    sparse page simply keeps *default* rather than dragging a whole document
    onto one page's guess.
    """
    parts = [
        region.text
        for region in page.ordered_regions()
        if region.type in PROSE_TYPES and region.text.strip()
    ]
    return detect_language(" ".join(parts), default=default)


# ==========================================
#        TESSERACT WORD GEOMETRY
# ==========================================


def tesseract_available() -> bool:
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def engine_citation(lang: str = DEFAULT_TESSERACT_LANG) -> str:
    """Name and version of the geometry engine, for the provenance summary."""
    try:
        import pytesseract

        return f"Tesseract {pytesseract.get_tesseract_version()} ({lang}), word geometry only"
    except Exception:
        return f"Tesseract ({lang}), word geometry only"


def tesseract_words(image_bgr, lang: str = DEFAULT_TESSERACT_LANG) -> List[Dict[str, Any]]:
    """Per-word boxes for a crop, in the crop's own pixel coordinates."""
    import pytesseract
    from PIL import Image

    rgb = image_bgr[:, :, ::-1] if getattr(image_bgr, "ndim", 0) == 3 else image_bgr
    # psm 4 ("single column of variable-sized text") rather than psm 6 ("one
    # uniform block"): psm 6 collapses a ruled table into a few bogus lines,
    # reading the rules themselves as text. On a real ruled table psm 6 found
    # 4 of 20 cell tokens where psm 4 found all 20, at no cost to prose.
    data = pytesseract.image_to_data(
        Image.fromarray(rgb),
        lang=lang,
        config="--psm 4",
        output_type=pytesseract.Output.DICT,
    )
    words: List[Dict[str, Any]] = []
    for index, text in enumerate(data.get("text", [])):
        if not str(text).strip():
            continue
        # Table rules read as "|" or "_". They are not words, and letting them
        # act as anchors drags real cell text onto the ruling line.
        if not _normalise(str(text)):
            continue
        try:
            conf = float(data["conf"][index])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < MIN_WORD_CONFIDENCE:
            continue
        left, top = data["left"][index], data["top"][index]
        words.append(
            {
                "text": str(text),
                "bbox": [
                    float(left),
                    float(top),
                    float(left + data["width"][index]),
                    float(top + data["height"][index]),
                ],
                "conf": conf,
                # Tesseract's own line grouping. Font size and baseline are
                # derived per line: a word's own ink box is the wrong basis,
                # because "we" (no ascender or descender) is much shorter than
                # "Frühjahr" on the very same line.
                "line": (
                    data.get("block_num", [0] * len(data["text"]))[index],
                    data.get("par_num", [0] * len(data["text"]))[index],
                    data.get("line_num", [0] * len(data["text"]))[index],
                ),
            }
        )
    return words


def page_text_layer(
    page: "doc_ir.Page",
    page_bgr,
    lang: str = DEFAULT_TESSERACT_LANG,
    word_provider: Optional[Callable[[Any, str], List[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """Build the invisible-layer entries for one page, in raster pixel space.

    ``word_provider`` is injectable so the alignment can be tested without a
    Tesseract binary present.
    """
    provider = word_provider or tesseract_words
    if page_bgr is None:
        return []
    height, width = page_bgr.shape[:2]

    # One engine call for the whole page, not one per region: pytesseract spawns
    # the tesseract binary on every call, so per-region cost scaled with layout
    # complexity (35 calls on a dense page). Whole-page is ~2x faster *and*
    # slightly more accurate — the engine gets full-page context for line
    # modelling, and region crops sometimes clip glyphs at their edges.
    try:
        all_words = provider(page_bgr, lang)
    except Exception:
        all_words = []

    targets = [
        region
        for region in page.ordered_regions()
        if region.type in LAYER_TYPES and region.bbox and region_layer_text(region).strip()
    ]
    buckets = _bucket_words(targets, all_words, width, height)

    entries: List[Dict[str, Any]] = []
    for region, words in zip(targets, buckets):
        x1, y1, x2, y2 = (float(v) for v in region.bbox[:4])
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(width), x2), min(float(height), y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        text = region_layer_text(region).strip()
        placed = align_tokens(tokenize(text), words, fallback_bbox=[x1, y1, x2, y2])
        # Line metrics are scoped to the region on purpose. Applied to raw
        # whole-page output they would be wrong: on a two-column page most
        # engine "lines" merge both columns, so a shared baseline derived from
        # that union suits neither. Bucketing first removes the merge entirely.
        _apply_line_metrics(placed, words)
        entries.extend(placed)
    return entries


#: Slack when testing whether a word belongs to a region, as a fraction of the
#: region's size: layout boxes often sit a hair inside the glyphs they contain.
_BUCKET_MARGIN = 0.02


def _bucket_words(
    regions: Sequence["doc_ir.Region"],
    words: Sequence[Dict[str, Any]],
    width: int,
    height: int,
) -> List[List[Dict[str, Any]]]:
    """Assign each word to the first region (reading order) holding its centre.

    First-match-wins so overlapping layout boxes cannot index the same word
    twice. Words outside every region are dropped, which is intended: the VL
    lane did not transcribe that area, so there is nothing to align them to.
    Engine order is preserved within a region, since alignment is positional.
    """
    buckets: List[List[Dict[str, Any]]] = [[] for _ in regions]
    if not words:
        return buckets

    boxes = []
    for region in regions:
        x1, y1, x2, y2 = (float(v) for v in region.bbox[:4])
        mx = max(1.0, (x2 - x1) * _BUCKET_MARGIN)
        my = max(1.0, (y2 - y1) * _BUCKET_MARGIN)
        boxes.append((x1 - mx, y1 - my, x2 + mx, y2 + my))

    for word in words:
        box = word["bbox"]
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        for index, (x1, y1, x2, y2) in enumerate(boxes):
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                buckets[index].append(word)
                break
    return buckets


#: How far Helvetica's *ink* reaches above and below the baseline, as em
#: fractions (cap/ascender height and descender depth).
#:
#: These deliberately are not the font's bbox metrics (``fitz.Font("helv")``
#: reports 1.075/-0.299): those describe the font's design envelope, while a
#: Tesseract line box measures the ink actually printed. Sizing ink with bbox
#: metrics makes the font ~30% too small and drops every baseline. Measured
#: against known glyph positions, the ink ratios below land within ~0.3pt.
_ASCENT, _DESCENT = 0.718, 0.207


def _font_metrics() -> Tuple[float, float]:
    """``(ascent, descent)`` used to map an ink box to a size and baseline."""
    return _ASCENT, _DESCENT


def _apply_line_metrics(placed: List[Dict[str, Any]], words: Sequence[Dict[str, Any]]):
    """Attach a per-line font size and baseline to each placed entry.

    A text line's ink box spans ascender to descender across *all* its words, so
    it yields one consistent font size and one shared baseline — which is what
    makes a selection highlight track the line instead of jittering word to word.
    """
    line_boxes: Dict[Any, List[float]] = {}
    for word in words:
        key = word.get("line")
        if key is None:
            continue
        box = word["bbox"]
        current = line_boxes.get(key)
        line_boxes[key] = (
            [box[1], box[3]] if current is None
            else [min(current[0], box[1]), max(current[1], box[3])]
        )

    for entry in placed:
        span = line_boxes.get(entry.get("line"))
        if span is None:
            box = entry["bbox"]
            span = [box[1], box[3]]
        ascent, descent = _font_metrics()
        height = max(1.0, span[1] - span[0])
        fontsize = height / (ascent + descent)
        entry["fontsize"] = fontsize
        entry["baseline"] = span[1] - descent * fontsize


# ==========================================
#        PDF ASSEMBLY
# ==========================================


#: Typographic characters that have a safe Latin-1 equivalent. Anything else
#: outside Latin-1 is dropped per-character rather than losing the whole word.
_LATIN1_SUBSTITUTIONS = {
    "‘": "'", "’": "'", "‚": "'",
    "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "−": "-",
    "…": "...", " ": " ", "‹": "<", "›": ">",
}


def _encodable(text: str) -> str:
    """Fold *text* to something the base-14 PDF fonts can actually write."""
    folded = "".join(_LATIN1_SUBSTITUTIONS.get(ch, ch) for ch in text)
    return "".join(ch for ch in folded if ord(ch) < 256).strip()


def build_searchable_pdf(
    layers: Dict[int, List[Dict[str, Any]]],
    source_pdf: Optional[str] = None,
    rasters: Optional[Dict[int, bytes]] = None,
    raster_dpi: int = 200,
    page_sizes: Optional[Dict[int, Tuple[int, int]]] = None,
) -> Optional[bytes]:
    """Write the invisible text layer onto the pages and return PDF bytes.

    ``layers`` maps 1-based page number to entries from :func:`page_text_layer`
    (raster pixel coordinates). When ``source_pdf`` is given the original file is
    reused, preserving scan quality; otherwise pages are built from ``rasters``.

    ``page_sizes`` gives the pixel dimensions of the raster each layer was
    measured on. When present the pixel->point scale is derived from real
    geometry per page, which is the only reliable way: a PNG carrying a dpi tag
    (screenshots are usually 96) makes PyMuPDF build a page at 72/96 of its
    pixel size, and a raster rendered at a different DPI than assumed shifts
    every word progressively further from the origin. ``raster_dpi`` is only the
    fallback when the true size is unknown.
    """
    import fitz

    if source_pdf:
        doc = fitz.open(source_pdf)
    elif rasters:
        doc = fitz.open()
        for page_number in sorted(rasters):
            data = rasters[page_number]
            pix = fitz.Pixmap(data)
            # Build the page at the raster's true pixel size rather than letting
            # an embedded dpi tag decide it, so pixels are points 1:1.
            page = doc.new_page(width=pix.width, height=pix.height)
            page.insert_image(page.rect, stream=data)
    else:
        return None

    try:
        for index in range(doc.page_count):
            entries = layers.get(index + 1)
            if not entries:
                continue
            page = doc.load_page(index)
            size = (page_sizes or {}).get(index + 1)
            if size and size[0] and size[1]:
                scale_x = page.rect.width / float(size[0])
                scale_y = page.rect.height / float(size[1])
            else:
                scale_x = scale_y = 72.0 / raster_dpi if raster_dpi else 1.0
            for entry in entries:
                x1, y1, x2, y2 = entry["bbox"]
                rect = fitz.Rect(x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
                if rect.is_empty or rect.height <= 0 or rect.width <= 0:
                    continue
                # The base-14 fonts are Latin-1. A character outside it makes
                # PyMuPDF drop the whole string — and its box with it — so
                # sanitise here rather than lose the surrounding words.
                text = _encodable(entry["text"])
                if not text:
                    continue
                # Size and baseline come from the whole text line (see
                # _apply_line_metrics); only fall back to this entry's own box
                # when the line is unknown.
                if entry.get("fontsize") and entry.get("baseline") is not None:
                    fontsize = max(1.0, entry["fontsize"] * scale_y)
                    baseline = entry["baseline"] * scale_y
                else:
                    ascent, descent = _font_metrics()
                    fontsize = max(1.0, rect.height / (ascent + descent))
                    baseline = rect.y1 - descent * fontsize
                # Condense to the measured width so the highlight matches the
                # printed word. insert_textbox is not usable here: it drops the
                # string entirely when it does not fit its rect.
                width = fitz.get_text_length(text, fontname="helv", fontsize=fontsize)
                if width > rect.width and width > 0:
                    fontsize = max(1.0, fontsize * rect.width / width)
                # render_mode=3 -> drawn but invisible: selectable and
                # searchable, never covering the scan underneath.
                page.insert_text(
                    fitz.Point(rect.x0, baseline),
                    text,
                    fontsize=fontsize,
                    fontname="helv",
                    render_mode=3,
                )
        buf = io.BytesIO()
        doc.save(buf, garbage=3, deflate=True)
        return buf.getvalue()
    finally:
        doc.close()
