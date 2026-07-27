# qwen3-vl instruct benchmark — results (2026-07-27)

Executed the handoff's "step 1": pulled `qwen3-vl:30b-a3b-instruct` (19.6 GB) into the shared
Ollama store and re-ran the real `process_document` pipeline on `split_1_2.pdf`, with a per-call
audit dir capturing every request/response.

Baseline for comparison is `split_1_2.json` (qwen3-vl:30b, the thinking build, num_predict=8000).

---

## 1. The handoff's root cause was wrong

The handoff claimed the staged `qwen3-vl:30b` had a **botched chat template** ("whoever
`ollama create`d this tag omitted the official Qwen3-VL template"). That is not the case:

| | model layer digest |
|---|---|
| registry `qwen3-vl:30b` | `sha256:b1da6f96a2e40e5d…` |
| **local staged** `qwen3-vl:30b` | `sha256:b1da6f96a2e40e5d…` — **identical** |
| registry `qwen3-vl:30b-a3b-instruct` | `sha256:8088c24b807ccac3…` |

The local tag is byte-identical to the official registry build. Neither manifest contains a
`template` layer, and **both** report `template: {{ .Prompt }}` via `/api/show` — that is simply how
Ollama ships qwen3-vl, whose prompt templating lives in the new-engine Go implementation rather than
in a Modelfile layer. So the stub template was never a defect and never explains anything.

The real difference is the capability list:

- `qwen3-vl:30b` → `[completion, vision, tools, **thinking**]`
- `qwen3-vl:30b-a3b-instruct` → `[completion, vision, tools]`

They are different models, and `think:false` was ignored because the thinking build simply cannot
be switched off — not because a template was broken.

## 2. The instruct model does remove the thinking tax — completely

**0 thinking characters across all 13 calls** (baseline: ~10.5k chars per call). Normal sections now
finish in 177–766 eval tokens, so `TEXTLAB_VISION_NUM_PREDICT=8000` is far larger than needed for
ordinary questions.

## 3. But on its own it is a regression: failed groups went 1 → 5

| metric | baseline (thinking) | instruct |
|---|---|---|
| groups | 17 | 18 |
| rowless / failed groups | **1** (`p2_s6`) | **5** (`p2_s4`, `p2_s5`, `p2_s6`, `p2_s7`, `p2_s9`) |
| calls with thinking | 13/13 | 0/13 |

The failure mode changed from "thinking exhausts the budget before any JSON" to **degenerate
repetition**. With no thinking to absorb the budget, the model pads JSON arrays until `num_predict`
truncates the response mid-string and the whole section is lost:

- `p2_s7` "**Geschlecht**" (a two-option question): 1 real question + **63 empty question objects**, 27,941 chars
- `p2_s4`: 1 real + **54 empty questions**, 30,280 chars
- `p2_s9`: correct question and all 8 villages, then **490 empty rows**, 30,418 chars
- `p2_s6` (big table): the same 16 rows repeated **7×** under hallucinated question titles, 27,425 chars
- `p2_s5`: a newline loop *inside* a string

## 4. Root cause of the repetition: every array in the schema was unbounded

`UNIVERSAL_RESPONSE_SCHEMA` set `minItems` but **no `maxItems`** on `questions`, `rows`, `choices`,
`extra_choices`, `marked_answers` and `unmapped_marks`. Constrained decoding never forces an
unbounded array to close, so the model can legally keep appending forever — and it does.
(`question_text` also has no `minLength`, which is why empty-string questions are schema-valid.
Note `_prune_section_groups` — the existing "drop empty-question groups" patch — was treating this
symptom downstream.)

Probe on the captured crops, re-sending the identical image and prompt with the array bounded:

| section | unbounded | `maxItems` |
|---|---|---|
| `p2_s7` Geschlecht | **FAIL** after 77.3 s (invalid JSON) | **OK in 4.8 s** |
| `p2_s6` big table | 19 questions / truncated in-pipeline | **OK in 9.6 s, all 16 rows** |

**Applied:** bounds on all six arrays in `src/core/form_extract.py`
(`questions` 6, `rows` 24, `choices` 24, `marked_answers` 24, `extra_choices` 8, `unmapped_marks` 12).
The caps are generous versus any real paper form, so they only bite on runaway generation.
Existing `test_regression.py` and `test_markup.py` still pass.

### End-to-end validation of the bounded schema

| metric | baseline (thinking) | instruct, unbounded | **instruct + bounds** |
|---|---|---|---|
| rowless / failed groups | 1 (`p2_s6`) | 5 | **1** (`p2_s5`) |
| **big table `p2_s6`** | FAIL | FAIL | **OK — all 16 rows with answers** |
| thinking chars per call | ~10.5k | 0 | 0 |
| groups | 17 | 18 | 20 |
| wall clock | — | 380 s | **233 s** |

The headline is that **`p2_s6` — the "big table" — extracts for the first time**, with all 16
statement rows and their marks. Individually failing calls were rescued outright: `p2_s4` went from
a 30,280-char truncation to a clean 3,552-char answer, and `p2_s6` from truncation to a clean stop
at 6,834 tokens.

Two caveats, both visible in the table above:

- **`p2_s5` still fails**, deterministically, with the newline-inside-a-string loop. `maxItems`
  bounds arrays, not strings, so this one is untouched. It is also the section whose crop is
  truncated (§6), so it needs fixing regardless.
- **Group count rose 17 → 20.** The repetition is now *bounded* rather than *eliminated*: `p2_s6`
  is emitted twice, the second copy under a hallucinated title ("Was ist Ihre Meinung zu den
  folgenden Aussagen?") carrying an identical 16-row body. `_prune_section_groups` misses it because
  it only drops *exact-text* duplicates — it should dedupe on row/option structure too.

## 5. Paddle already reads the big table better than the VLM does

Ground truth for `p2_s6` was read by eye from the 300-DPI crop (`ground_truth_p2_s6.json`):

| source | exact rows | notes |
|---|---|---|
| **PaddleOCR-VL table parse** (`p2_r8`) | **15 / 16** | only misses row 1; correctly captures the ambiguous double-mark on row 12 |
| Qwen instruct, first pass | 12 / 16 | 3 wrong, all in the **bottom rows** (14–16) — positional drift down the matrix |

Paddle already emits this table as an **18×7 dataframe with the marks intact** (`✗` vs `○`), and it
already renders in the **Tables** tab. The data the user wants exists today; it is just missing from
the Responses tab. Meanwhile the schema-free path computes `_table_schema` — Paddle-derived rows,
options and `paddle_state` — and then **discards it**, using it only in the error fallback.

## 6. Separate bug found: section crops are truncated at the column boundary

`p2_s5`'s crop cuts off the table's last response column:

- table region `p2_r5` spans x = 94 → **730**
- section crop `p2_s5_q1` spans x = 31 → **645**

`_question_sections` sets the column boundary to the midpoint between question-anchor *centres*, and
`_section_bbox` then **clamps** the crop to it. A wide table is assigned to the left column by its
centre (x=412) but extends past the boundary, so ~85 px is sliced off. Both models therefore missed
row 3's answer; Paddle, which sees the whole page, found it. The crop should expand to contain its
own section's regions rather than clamp to a geometric midpoint.

## 7. Two smaller operational notes

- **Paddle OOMs when an Ollama model is resident.** The card is 23 GB and the model is 21 GB.
  `_unload_other_models()` runs inside `prepare()`, i.e. at the *first* `analyze()` call — which is
  *after* the Paddle stage. A user who used chat before the OCR page can OOM the Paddle lane.
- **Nothing is release-approved.** `TEXTLAB_APPROVED_SURVEY_MODELS` is set nowhere (not in
  `script.sh.erb`), so *every* group is force-flagged `needs_review` regardless of quality. That
  alone is most of why the Responses tab feels heavy.

## 8. Recommendation

Switching model is necessary but **not sufficient**, and must ship together with the schema bounds —
the instruct model alone is worse than what is deployed. Together they are a real improvement: the
thinking tax is gone, the run is 39 % faster, and the big table finally extracts.

Beyond that, the evidence points away from asking the VLM to re-derive table structure it is worse
at than the parser already in the pipeline: let Paddle own table-shaped matrix questions (reusing
the `_table_schema` hint that is currently discarded) and keep the VLM for free-form choice
questions. Do **not** delete the Paddle-ID machinery — that recommendation from the earlier
brainstorm is withdrawn.

### Final state after the follow-up changes

| metric | baseline (thinking) | instruct alone | instruct + bounds | **+ the four changes** |
|---|---|---|---|---|
| groups | 17 | 18 | 20 | **17** |
| rowless / failed | 1 | 5 | 1 | **1** (`p2_s5` only) |
| big table `p2_s6` | FAIL | FAIL | OK | **OK** |
| duplicate matrix | — | — | yes | **gone** |
| groups needing review | 17 / 17 | 18 / 18 | 20 / 20 | **13 / 17** |
| wall clock | — | 380 s | 233 s | **214 s** |

Four groups now come out **`accepted` with zero warnings**; previously every group was flagged
regardless of quality. The remaining 13 are flagged for substantive reasons, dominated by
*geometric ink evidence disagrees with the OCR transcription* (8 of 17) — that heuristic is now the
single largest source of review load and is the obvious next thing to measure.

### Applied after the benchmark

1. **Structure-aware dedupe** in `_prune_section_groups`. The text-keyed signature could not catch a
   matrix re-emitted under an invented title, so a second body-only signature was added, scoped to
   groups with **two or more labelled rows**. That targets duplicated matrices while leaving two
   legitimately identical simple questions (e.g. two Ja/Nein items in one section) intact.
2. **Crop clamp fixed** in `_section_bbox`: the column limits may now only shrink the *padding*,
   never cut into the section's own regions. `p2_s5`'s crop goes from x2=645 back to x2=730.
3. **Launcher** now defaults `TEXTLAB_VISION_MODEL` to `qwen3-vl:30b-a3b-instruct` and sets
   `TEXTLAB_APPROVED_SURVEY_MODELS` to the same, so groups can come out `accepted`. The old
   thinking tag deliberately stays unapproved, so overriding back to it still forces review.
4. **`num_predict` stays at 8000** — this reverses the earlier suggestion to lower it. Ordinary
   sections use 177–949 tokens, but the big matrix legitimately needs **6,834**, so 3000 would
   re-break the very thing that was just fixed.
5. **A malformed question no longer costs its whole section.** Widening the crop (change 2) made the
   model see more of the prize/legal paragraph on `p1_s2` and emit it as a 240+ character "choice",
   which raised and discarded *every* question in that section — including the good one. Per-question
   field validation moved into `_question_defect()`, so an unusable question is dropped on its own
   and recorded as a warning; the section only fails if nothing survives.

### Remaining, in order

1. Re-do the Responses tab as summary-first, with editing only on genuinely flagged questions.
2. `p2_s5`'s newline-inside-a-string loop still needs a mitigation (`repeat_penalty`, or a differing
   seed on retry) — array bounds cannot reach it.
3. Consider the Paddle-owns-tables hybrid if matrix rows remain the dominant error source.

Reproduction: `bench_survey.py` / `score_run.py` / `probe_bounded.py` and the full per-call audit
dirs are in this session's scratchpad; `ground_truth_p2_s6.json` holds the hand-read ground truth.
