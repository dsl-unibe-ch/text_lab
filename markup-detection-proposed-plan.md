# TextLab Form and Survey Response Extraction — Proposed Plan

> **Status:** accepted direction; implementation in progress  
> **Date:** 2026-07-20  
> **Companion documents:** `ocr-page-implementation-plan.md`,
> `markup-detection-current-state.md`  
> **Target branch:** `feature/auto-ocr`

## 1. Decision summary

Keep the normal document path simple and cheap: PyMuPDF for suitable
born-digital pages and PaddleOCR-VL for scans, complex layout, or the
highest-quality path. Add two explicit, independent enrichments: **Describe
figures and images** and **Extract survey/form responses**. Only the second
action runs targeted, question-level response extraction; ordinary OCR incurs
no additional large-VLM cost.

The initial implementation uses the already-staged local `gemma4:26b` model on
complete high-resolution question sections, not entire pages. The
locally staged, vision-capable `qwen3.6:35b` must be included in the benchmark
on an A100/H100; its roughly 24 GB weights leave too little margin for the 23 GB
RTX 4090. University GPUStack Qwen is a contingent option only if local models
do not clear the quality gate.

The geometric detector remains available for diagnostics and supporting
evidence, but it must no longer silently change OCR content. For batches using
the same questionnaire, v1 reuses normalized crop locations and printed
schemas. Registered template differencing is promoted into the early benchmark
because it may become the preferred high-throughput batch path.

In short:

```text
explicit survey action
  -> PaddleOCR-VL: layout, printed text, tables, initial mark observations
  -> form-region proposal and high-resolution cropping
  -> targeted response VLM: selected/cancelled/ambiguous answers
  -> conservative reconciliation and validation
  -> question-grouped IR, review UI and semantic exports
```

This retains the simple OCR-page experience. Users choose outcomes, not OCR or
VLM engine names.

## 2. Why the current approach should not be extended

The current implementation has reduced false overrides, but the remaining
failures are caused by its input assumptions rather than threshold tuning:

1. `find_marks()` looks for closed circle/box contours. A heavy handwritten X
   destroys the contour, so the most important marked option can disappear
   before classification.
2. Non-checkbox regions are processed only if Paddle has already emitted a mark
   glyph. A dropped glyph or a circle transcribed as `O` prevents detection from
   running at all.
3. Positional reconciliation requires the OCR and geometric candidate counts to
   match. Large grids are therefore rejected precisely where positional
   recovery is most useful.
4. Fill and Hough thresholds are unstable across scan resolution, mark size,
   paper noise and downscaling.
5. A flat sequence of `checked`/`unchecked` glyphs does not represent the user
   concept: a response belongs to a question, row and labelled option.
6. Rewriting OCR text or table HTML from `○` to `☒` destroys provenance and can
   propagate a wrong inference into every export.
7. Changed answers are not binary checkbox states. A form can contain an active
   answer, a cancelled earlier answer, or genuinely ambiguous intent.

Further geometric work may still be useful for a dedicated, trained detector
or a registered template, but the current generic contour pipeline should not
remain an authoritative correction layer.

## 3. Evidence from the supplied survey

A small diagnostic was run against 300-DPI crops from `split_1_2.pdf` using the
local Ollama `gemma4:26b` model already staged for TextLab.

Observed results:

- Focused crops correctly recovered Q1=`4`, Q2=`Nein`, Q2=`Andere Gründe`, and
  Q3=`Nein`, including marks that Paddle's incidental transcription missed.
- A single crop containing all 16 rows of Q10 produced several wrong column
  assignments because the image was resized too aggressively by the visual
  encoder.
- Splitting Q10 into three overlapping horizontal crops recovered all visible
  marked positions in this spot check, including both marks in the row with a
  changed answer.
- For the changed-answer example in Q9, the model saw both marks but did not
  reliably distinguish the dense scribble as a cancellation. This case must be
  reviewed rather than guessed.
- After model loading, focused calls took roughly 1–2 seconds each. The model
  used approximately 18.4 GiB on the RTX 4090.
- `think: false` was required for concise structured extraction. With thinking
  enabled, reasoning could consume the output budget before a final answer was
  emitted.
- A later end-to-end run through 300-DPI Paddle plus the implemented structured
  crop pipeline took **229 seconds for two pages** on the RTX 4090. Gemma still
  produced plausible false selections on several wide rating/matrix questions,
  occasionally treated clean outlines as selected, and returned one invalid
  structured tile. Geometry/rule validation successfully forced these groups
  into review, but the result is not suitable for automatic acceptance.
- Auditing showed that the original response schema asked Gemma to repeat
  question/row structure it did not need to own. Replacing it with one ID-only
  contract (`visible_row_ids`, `marks`, `unmapped_marks`) for every call reduced
  the same audit from 224 to **175 seconds**, produced 43/43 parseable responses
  instead of 42/43, and removed variable question/row labels. It also corrected
  obvious over-selection on Q5 and Q7. Visual ambiguity/false-mark problems remain on
  examples such as Q8 and Q13, so the benchmark gate still applies.
- The call-by-call audit also showed that independent horizontal/vertical tiles
  removed the visual relationship between questions, labels, and marks. The
  implemented policy now makes one request per complete Paddle-bounded question
  section. Ordinary tables and single geometric contours are no longer form
  candidates; geometry may propose a section only when it finds an aligned
  repeated mark pattern.

These are feasibility checks, not accuracy results. They demonstrate that the
local model is usable as an experimental observation source and that crop
construction and visual resolution are critical. No survey model is approved
for automatic acceptance; a labelled multi-document benchmark remains a hard
release gate.

## 4. Scope

### In scope

- Extract form/survey content only after the explicit user action.
- Extract selected responses from circles, radio buttons and boxes.
- Support single-choice, multiple-choice, rating-scale and matrix questions.
- Represent multiple visible marks, cancellation candidates and ambiguity.
- Associate every result with question/row/option labels.
- Preserve Paddle output and all model observations for auditability.
- Surface disagreements and uncertain cases for review.
- Export semantic form responses in canonical JSON and tabular formats.
- Evaluate local Gemma, local Qwen, and template differencing using identical
  labelled inputs where applicable.
- Retain a one-button default OCR workflow.

### Out of scope for the first implementation

- Claiming perfect interpretation of crossed-out or corrected answers.
- Training a custom detector before representative real data exists.
- Automatically aggregating an arbitrary batch into one respondent row without
  a reviewed question schema.
- Silently sending documents to GPUStack.
- Replacing PaddleOCR-VL for ordinary text, layout, tables, figures or formulas.
- Supporting every kind of handwritten free-text answer semantically. Normal
  OCR transcription continues to handle free text.

## 5. Proposed architecture

### 5.1 Preserve the primary document parser

PaddleOCR-VL remains responsible for:

- page layout and reading order;
- printed question and option text;
- table structure and matrix row/column labels;
- existing Markdown/HTML/LaTeX content;
- an initial, non-authoritative observation of mark glyphs.

The VLM response extractor consumes Paddle's structure but does not replace or
mutate it.

### 5.2 Explicit activation and region proposal

V1 does not calculate form likelihood on every document. The user explicitly
enables **Extract survey/form responses**, which routes those pages through the
300-DPI Paddle path and then proposes candidate question regions. Proposal is
Paddle-first and requires at least one of:

- Paddle content containing a circle, square, checked-mark glyph, or recognised
  mark notation;
- a checkbox/radio layout region;
- a compact binary option row such as `Ja`/`Nein`; or
- at least two aligned circle/box shapes found by non-authoritative image
  analysis when Paddle omitted every control glyph.

A generic table and a single geometric contour are explicitly insufficient.
This prevents ordinary tables, diagrams, prize lists, and free-text questions
from consuming VLM calls. Automatic survey detection remains a later
enhancement, not a v1 dependency.

### 5.3 Split pages before extracting answers

The original page raster must remain available until response extraction has
finished. Preview images and preview-scaled bounding boxes must never be used as
model input.

Processing order:

1. Rasterise likely form pages at 300 DPI.
2. Detect scanned spreads or strong column gutters and split them into panels.
3. Group adjacent Paddle blocks into question sections, using question/title
   boundaries, table boundaries and whitespace.
4. Add modest context around every crop so question and option labels remain
   visible.
5. Submit the complete section in one request. Do not issue independent
   horizontal or vertical snippet calls.
6. If a later benchmark proves that an exceptionally large matrix needs zoomed
   details, send the complete overview and its details together in one
   multi-image request so every answer is produced with full visual context.

The exact point at which an additional detail image is useful must be chosen
from benchmark results, not hardcoded from the single test document. The
complete section must always remain the primary evidence image.

### 5.4 Build an explicit question schema

Before asking the response VLM to read marks, derive the printed form structure
from Paddle whenever possible:

```json
{
  "question_id": "p2_q10",
  "question": "Für welche Themen ...?",
  "selection_rule": "one_per_row",
  "columns": ["0", "1", "2", "3", "4", "5"],
  "rows": [
    {"row_id": "r1", "label": "Produktion und Vermarktung lokaler Produkte"}
  ]
}
```

The prompt should give the model stable IDs and an allow-list of labels. The
model should return IDs/states rather than freely re-OCRing long labels. This
reduces hallucinations and column shifts and allows deterministic validation.

If Paddle cannot derive a complete structure, the crop extractor may ask the
response VLM to return both structure and observations, but these results start
as `needs_review` until validated.

### 5.5 Targeted response VLM

Add `src/core/form_extract.py` with a provider-neutral interface, for example:

```python
class FormExtractor(Protocol):
    def extract(self, crop, schema, *, context=None) -> FormGroup: ...
```

Initial providers:

- `OllamaGemmaFormExtractor` using `gemma4:26b` locally.
- `GPUStackQwenFormExtractor` using the OpenAI-compatible university endpoint.

Required local Gemma settings for the initial implementation:

- `think: false`;
- `temperature: 0`;
- a fixed seed where supported;
- structured JSON output;
- a bounded output token budget;
- keep the model loaded while all crops for a document/batch are processed;
- run after the Paddle subprocess exits so the two models do not compete for
  GPU memory.

The provider must validate JSON against a local schema. Invalid JSON, unknown
IDs, duplicate rows, missing required rows and impossible states must produce a
warning/review result, not a guessed repair.

The model prompt must distinguish visual observation from semantic intent:

- clean printed outline -> no visible mark;
- ordinary handwritten X/tick/fill -> selected candidate;
- dense scribble/multiple cancellation strokes -> cancellation candidate;
- visible but unclear handwriting -> ambiguous;
- never infer an answer from question wording or expected survey logic.

### 5.6 New question-level IR

Markup can span several Paddle regions, so it should not be stored only as an
untyped `Region.markup` dictionary. Add a question-level structure to `Page`,
while retaining `Region.markup` temporarily for backward compatibility during
migration.

Suggested shape:

```text
Page.form_groups[]
  id
  bbox
  question_text
  question_type: single | multiple | rating | matrix | unknown
  selection_rule: zero_or_one | exactly_one | zero_or_more | one_per_row
  rows[]
    id
    label
    options[]
      id
      label
      bbox (when available)
      visual_mark: none | x | tick | filled | scribbled | other | uncertain
      state: selected | unselected | cancelled | ambiguous
      observations[]
        source: paddle | gemma4-26b | qwen3-vl-30b | geometric | template | human
        value
        raw
      evidence_crop
  status: accepted | recovered | needs_review | failed
  warnings[]
  source_crop
```

Important invariants:

- OCR content is immutable after parsing.
- Every inferred state retains its observations and method.
- Model-reported confidence is not treated as calibrated probability.
- `cancelled` is used only when evidence is clear; otherwise use `ambiguous`.
- Human corrections are stored as another observation and become authoritative
  for the current document.

### 5.7 Conservative reconciliation

Use deterministic rules rather than a single blended confidence score:

| Situation | Result |
|---|---|
| Paddle and targeted VLM agree | `accepted` |
| Paddle missed a mark; VLM finds it consistently in overlapping crops | `recovered` |
| Paddle and VLM disagree | `needs_review` |
| Overlapping crops disagree | `needs_review` |
| More than one visible mark in a single-choice row | `needs_review` |
| VLM returns an unknown option/row ID | reject result and `needs_review` |
| Required row is missing | `needs_review` |
| Geometry alone proposes a correction | evidence only; never authoritative |

Repeating the same model with the same crop is not independent evidence.
Agreement between differently framed/overlapping crops is a useful consistency
signal, but should not be presented as calibrated confidence.

Optional remote verification may be applied only to local-model disagreements
or review cases, which limits latency, cost and data transfer.

### 5.8 Template-assisted mode for repeated forms

For batches containing the same questionnaire revision, template registration
is likely to outperform a general VLM and should be designed as a later provider
behind the same IR:

1. Obtain a blank template, or reconstruct a printed baseline from the aligned
   median of a sufficiently large batch.
2. Deskew and register each response page to the template using robust features
   and a homography/affine transform.
3. Detect printed option positions once on the template.
4. Compare each response against the template at those positions.
5. Use the response VLM only for multiple marks, poor registration and unusual
   corrections.

This mode still cannot always infer which of two marked answers the respondent
intended to cancel; such cases remain review items.

## 6. Model selection and privacy

### Local Gemma 4 26B

Proposed default because it:

- is already staged and available to TextLab through Ollama;
- accepts images;
- fitted on the available RTX 4090 during the diagnostic;
- recovered the important missed marks when given focused crops;
- preserves the current local/offline privacy model.

`gemma4:31b` should not be prioritised for the 4090. It has a tighter memory
margin, while crop design already fixed the observed 26B spatial errors.

### Local Qwen 35B and university GPUStack

The staged `qwen3.6:35b` has vision metadata and should be tested locally on a
larger GPU because its weights do not fit the 4090 with a safe runtime margin.
It must use the same crops, schemas, prompts and labelled answers as Gemma.

GPUStack is considered only if it materially improves the held-out result.
Remote use changes TextLab's offline boundary even though the service is private
and university-hosted. Therefore, if it is later added:

- it must never be enabled silently;
- the UI/documentation must say that form crops leave the local job;
- credentials must not be stored in document JSON or logs;
- timeouts and service failures must fall back to local extraction/review;
- only necessary form crops should be sent, not entire documents, unless the
  user explicitly chooses otherwise.

Relevant upstream references:

- Gemma image understanding and variable-resolution vision:
  <https://ai.google.dev/gemma/docs/capabilities/vision/image>
- Qwen3-VL 30B model card:
  <https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct>

## 7. User experience

### Default single-document flow

Keep one primary action: **Parse document**.

The automatic pipeline should:

1. parse the document normally;
2. run no additional large VLM by default;
3. run figure descriptions only when **Describe figures and images** is enabled;
4. run 300-DPI targeted form extraction only when **Extract survey/form
   responses** is enabled;
5. show a Responses tab when form groups or review candidates exist.

The UI does not expose Gemma/Paddle/Qwen as ordinary engine choices. Model
selection is an internal benchmarked deployment decision.

If remote Qwen is later offered, it should be phrased as an explicit privacy
choice such as **Use university GPUStack to verify uncertain responses**, not as
another OCR engine.

### Responses tab

Replace the flat mark-chip display with question-level cards:

- question text and page number;
- selected option(s), grouped by row for matrices;
- clear status badge: accepted, recovered or needs review;
- the high-resolution evidence crop;
- source agreement/disagreement summary;
- editable controls for ambiguous or disputed states;
- filter to show review items only.

Do not show hundreds of unchecked cells by default. For a large matrix, show a
compact response table and expand individual evidence only when requested.

### Exports

Add semantic outputs while preserving existing document exports:

- canonical form-response JSON with provenance;
- one row per question/matrix row CSV for a single document;
- review report containing only uncertain/disputed cases and evidence;
- later: one row per respondent for batches sharing a reviewed schema.

Markdown and Paddle table HTML remain faithful to the original parser output.
They should not be mutated to encode response-extraction corrections.

## 8. Benchmark before integration

Build a standalone benchmark harness before wiring the new extractor into the
Streamlit page.

### Dataset

- At least 10–20 real pages for the first decision, preferably more.
- Several form designs, not only `split_1_2.pdf`.
- Circles, boxes, ticks, thin Xs, heavy Xs, fills and colored/black pens.
- Grayscale scans, noisy paper, skew, compression and multiple DPIs.
- Small binary questions, rating scales and large matrices.
- Deliberate changed answers and ambiguous examples.
- Blank controls and pages with form-like shapes that are not response fields.

Every row/option must be hand-labelled with:

- visible mark type;
- semantic state (`selected`, `cancelled`, `ambiguous`, `unselected`);
- question type/selection rule;
- whether human intent is genuinely decidable.

Keep a small development set for prompt/crop iteration and a separate held-out
set for the model decision.

### Compared systems

1. Paddle glyph transcription only.
2. Current geometric approach, frozen as the baseline.
3. Local Gemma on full regions.
4. Local Gemma on proposed crops/tiles.
5. GPUStack Qwen on the same crops/tiles.
6. Paddle + Gemma reconciliation.
7. Paddle + Qwen reconciliation.
8. Optional: local extractor with remote verification only on disagreements.

### Metrics

- selected-mark precision and recall;
- exact answer-set accuracy per question/row;
- false selected answers (highest-risk error);
- missed selected answers;
- cancellation/ambiguity accuracy, reported separately;
- proportion automatically accepted;
- review recall: fraction of wrong automatic answers that were flagged;
- form-region proposal recall;
- invalid/malformed structured outputs;
- run-to-run consistency;
- seconds per page/crop and first-model-load time;
- peak VRAM;
- remote bytes/calls and failure rate where relevant.

### Initial release gates

Exact numeric thresholds should be agreed after seeing label prevalence, but the
following principles are release requirements:

- zero known silent false selections on the small release audit;
- every disagreement or invalid output is surfaced for review;
- the combined system materially improves selected-mark recall over Paddle;
- review recall is prioritised over the percentage auto-accepted;
- no OCR text/HTML is mutated by an inferred response;
- local mode works with network blocked;
- GPUStack is opt-in and fails safely;
- the 4090 completes a representative document without OOM.

## 9. Implementation phases

### Phase 0 — safety correction

- Disable geometric overrides and `apply_states_to_content()` in the automatic
  pipeline.
- Retain glyph extraction, geometry, overlays and evidence for baseline/debug
  reporting.
- Stop presenting geometry-derived rewrites as corrected ground truth.
- Adjust UI wording so TextLab does not claim reliable automatic survey-state
  detection before the replacement is validated.

### Phase 1 — benchmark spike

- Obtain representative labelled pages before making the provider decision.
- Create labelled ground-truth format and evaluation script.
- Add deterministic crop generation at 300 DPI.
- Implement matrix tiling with overlap.
- Compare local Gemma, local Qwen on a suitable GPU, the frozen geometry
  baseline, and registered template differencing for repeated forms.
- Freeze prompts and structured schemas.
- Measure seconds/document, batch wall-clock, first-load time, peak VRAM, and
  GPU lifecycle behavior as release gates.
- Produce a comparison report and choose the internal default provider.

No production UI integration should occur before this phase's decision.

### Phase 2 — IR and local extraction

- Add typed `FormGroup`, `FormRow`, `FormOption` and `Observation` structures.
- Extend `Page` serialization and bundle exporters.
- Add the explicit survey action and high-recall region proposal.
- Implement the selected local provider behind `FormExtractor`.
- Add strict response validation and conservative reconciliation.
- Keep original full-resolution rasters until extraction completes.
- Add unit tests for schemas, crop mapping, complete-section requests,
  candidate filtering and validation.

### Phase 3 — review-focused UI and exports

- Replace the current flat Markup tab with question/row cards and matrix tables.
- Add evidence viewing and editable review states.
- Add canonical form JSON, response CSV and review-report downloads.
- Add progress messages for page parsing, form extraction and verification.
- Update privacy and OCR documentation.

### Phase 4 — registered template batch path

- Add robust registration and template differencing for repeated survey batches.
- Use the response VLM for poor registration, multiple marks, and corrections.
- Add batch respondent aggregation only after question schemas are stable.

### Phase 5 — optional GPUStack verification

- Add an explicit opt-in control and privacy disclosure.
- Verify only local disagreements/review items by default.
- Add credential handling, timeout, retry limits and safe fallback.
- Record provider/source metadata without credentials or sensitive request logs.

### Phase 6 — learning loop

- Store user-reviewed corrections in an exportable annotation format.
- Reassess a small trained mark detector only after enough representative real
  corrections exist.

## 10. Testing strategy

### Unit tests

- question schema parsing and stable IDs;
- JSON validation and unknown-ID rejection;
- crop coordinate transforms between full raster and preview;
- spread/column splitting;
- matrix tiling, overlap and deduplication;
- reconciliation state table;
- immutable Paddle content;
- IR round-trip serialization;
- remote-client timeout and credential redaction.

### Integration tests

- supplied two-page German survey;
- born-digital form with raster handwriting;
- scanned form without any recognised mark glyphs;
- large matrix with one mark per row;
- multi-select question;
- changed-answer example;
- ordinary document with circles/boxes that are not form fields;
- batch ZIP containing mixed documents and surveys;
- offline execution with staged local models.

### Visual audit

Retain a validation command that writes:

- proposed question/group boxes;
- complete question-section boundaries;
- selected/cancelled/ambiguous overlays;
- per-option observation disagreements;
- evidence crops for every review item.

Synthetic tests remain useful for mechanics but cannot substitute for the real
scan benchmark.

## 11. Operational considerations

- Paddle and Gemma run sequentially on the 4090. The implementation serializes
  local vision use, unloads other Ollama models before analysis, keeps the
  selected model warm for a document/batch, and unloads it afterward.
- Batch all Paddle pages as today, then process form crops while keeping Gemma
  alive across the document/batch.
- Bound maximum candidate crops per page and warn rather than hanging on a
  pathological document.
- Cache extraction results within the job directory using a key derived from
  crop bytes, schema version, provider, model and prompt version.
- Include `model`, `provider`, `prompt_version`, `schema_version` and crop
  metadata in observations for reproducibility.
- Never include document images/base64 evidence in normal application logs.
- Continue using the existing self-cleaning OCR job directories.

## 12. Open review questions

These should be resolved after the benchmark, not assumed in implementation:

1. Does Qwen materially outperform tiled local Gemma on held-out real forms?
2. Is remote verification acceptable under TextLab's privacy requirements?
3. What review-recall and false-selection thresholds are acceptable for the
   support use case?
4. Can the requester provide blank templates or multiple responses using the
   same questionnaire revision?
5. Should human corrections be session-only initially, or persisted/exported as
   reusable annotations from the first version?

## 13. Recommended approval

Approve Phases 0 and 1 first. Do not commit to Gemma or Qwen as the production
authority until both have been run against the same held-out labelled crops.

The expected production direction is nevertheless clear:

- keep Paddle for document parsing;
- use question-sized, high-resolution VLM extraction for form responses;
- default to local processing unless a benchmark and privacy decision justify
  GPUStack;
- treat disagreement and changed answers as review work;
- preserve provenance instead of rewriting OCR;
- add template differencing when repeated-form batches justify it.
