# Survey mark-extraction — handoff (2026-07-22)

Branch: `feature/auto-ocr`. All changes below are **in the dev tree `src/`, NOT git-committed.**

## Goal
Make the survey/form mark-extraction feature usable: a local VLM reads which options a
respondent marked on scanned questionnaires, behind a human review step. Test doc: `split_1_2.pdf`.

## Decisions already settled (don't relitigate)
- **Model = `qwen3-vl:30b`** (local Ollama, shared store), long detailed prompt, `needs_review`
  gate. Chosen over gemma4:26b/gemma3:27b/glm-ocr in the 2026-07-22 benchmark (best perception,
  fixed over-selection + glyph leakage). gemma4:26b is the fast fallback.
- Short prompts and a 2nd restructuring call were tested and rejected (hurt or no help).
  See `prompt-complexity-experiment-results.md`, `qwen-two-pass-survey-experiment-results.md`.

## ROOT CAUSE — the staged `qwen3-vl:30b` has a BROKEN chat template (this is the big lead)
Verified 2026-07-27: container **Ollama = 0.24.0** (NOT 0.22.1 — that earlier number was a different
server on :11434). `ollama show qwen3-vl:30b` (via /api/show) reports:
- `capabilities: [completion, vision, tools, thinking]`, family `qwen3vlmoe`, 31.1B, Q4_K_M
- **`template` is literally `{{ .Prompt }}`** — a stub. No `<|im_start|>` role markers, no
  system/vision structure, no thinking-control logic.

That stub template is almost certainly the core problem: whoever `ollama create`d this tag omitted the
official Qwen3-VL chat template, so the model gets raw un-templated text. Consequences observed:
- `think:false` (top-level AND in `options`) and `/no_think` are ALL ignored — there is no template
  logic to honor them, so the model always thinks (~10.5k chars) → on the dense matrix `p2_s6`
  thinking exhausts even `num_predict=8000` before any JSON → rowless fallback group.
- Very likely also inflates the over-generation / single-as-matrix mis-structuring, since the model
  is not receiving a properly-formatted chat prompt.

Splitting the crop does NOT help (a half-band thinks just as much). Confirmed via scratchpad probes.

FIX PATH — confirmed by Qwen + Ollama docs (web search 2026-07-27):
- Qwen split thinking/non-thinking into SEPARATE models; for non-thinking use the Instruct variant.
- For Qwen3-VL the in-prompt soft-switches (/think, /no_think) are unreliable/ignored (matches probes).
- Ollama `think:false`/`--nothink` is TEMPLATE-DRIVEN — it toggles a variable in the model's chat
  template. Our tag's template is the stub `{{ .Prompt }}`, so the toggle has nothing to act on →
  silently ignored. That is the exact mechanism behind every failed think:false/options/no_think probe.

1. **PRIMARY: `ollama pull qwen3-vl:30b-a3b-instruct`** (~20 GB; the non-thinking model the remote
   GPUStack audit used). Non-thinking by design + ships a proper template → fixes both the thinking
   tax and the stub-template issue. Then re-run split_1_2 and re-benchmark validity/structure/matrix.
2. Secondary/optional: if a reasoning model is ever wanted, re-create the current tag with the correct
   official Qwen3-VL template (the stub also likely degrades quality via raw un-templated input) — but
   for THIS task, Instruct is the answer.
NOTE: "upgrade Ollama" is OFF the table — already 0.24.0. Sources in the chat log / handoff.

## Changes made this session (dev tree, uncommitted)
- `src/pages/OCR.py` — rewrote the Responses tab into an **interactive review editor**:
  `_render_form_review` (radio for single/rating/matrix rows, checkboxes for multiple; pre-filled
  from `FormOption.state`; per-question "Show scan"; 💾 Save corrections), `_apply_form_corrections`
  (writes picks back; keeps the model's original in `group.provenance["model_answer"]` +
  `human_reviewed`; regenerates CSV/JSON/bundle downloads), `_render_legacy_mark_summary`
  (image-free). Removed the confusing circle crops/chips + the read-only dataframe. `clear_results`
  now drops `rev_*` widget keys. Shows group warnings + a graceful message for rowless groups.
  Correction logic is unit-tested (single + multi-select).
- `src/core/vision_enrich.py` — default model `gemma4:26b`→`qwen3-vl:30b`; `TEXTLAB_VISION_NUM_PREDICT`
  default `3000`→`8000` (accommodates thinking so the matrix stops raising).
- `src/core/form_extract.py` — `_prune_section_groups(groups)`: within a section, drop exact-duplicate
  groups and groups with empty question text (never a conditional parent; never prune to empty).
  Called just before the return in `_universal_to_form_groups`. Validated **21→17** on `split_1_2`
  (dropped exactly `p1_s2_q3` dup phone, `p1_s6_q2`, `p2_s7_q2`, `p2_s9_q2`).
- `template/script.sh.erb` — adds `--env TEXTLAB_VISION_MODEL="${TEXTLAB_VISION_MODEL:-qwen3-vl:30b}"`.

## CURRENT STATUS (start here in the new chat)
User runs the **dev sandbox OnDemand app** (confirmed on the dev tree: they SEE the new interactive
review UI; the deployed `src_main280526` still has the OLD OCR page). So my hypothesis "wrong tree /
no reload" is WRONG — the edited code IS what runs. Code verified correct on disk and NOT overridden:
`vision_enrich.py:83` num_predict default=8000 (passed into every call), `form_extract.py:1541` calls
`_prune_section_groups`, both `OllamaVisionClient()` use the defaults, nothing sets
`TEXTLAB_VISION_NUM_PREDICT`.

RESOLVED with a fresh run (`split_1_2.json`, 2026-07-27 11:17, fixes confirmed live):
- **Prune works.** 21 → **17 groups**; the 4 junk groups are gone; no cross-group duplicate
  question text remains.
- **The perceived "duplicate questions" is NOT separate groups — it is single questions mis-typed as
  multi-row matrices.** `p1_s2_q1` ("Möchtest du an der Verlosung teilnehmen?", a simple Ja/Nein) is
  emitted as a 3-row matrix with row1 Ja=selected AND row2 Nein=selected (contradictory); `p1_s5_q1`
  merges its conditionals in as rows. The review UI renders each row as its own widget, so one
  question looks repeated 3×. (`p2_s5_q1` is a *legitimate* 3-row matrix.) The prune cannot fix this
  — it is within a single group.
- **`num_predict=8000` does NOT fix the live matrix.** `p2_s6` STILL fails with "Vision model
  returned no final content" — i.e. even at 8000 the live crop's thinking exhausts the budget before
  any JSON. So the 8000 bump (validated on the smaller saved crop) does not generalize to the real
  pipeline crop. Raising the budget further is a losing game; the matrix needs **thinking removed**.

Net: the two real problems left are both driven by the forced thinking model — (a) the matrix
never emits content, and (b) qwen over-structures single questions into contradictory matrices.
Both point away from more normalizer patching and toward removing thinking / the mark-only contract.

## Remaining issues even once the code IS live (qwen structural noise)
The prune only removes exact dups + empty-text strays. qwen still:
- mis-emits a single Ja/Nein question as a multi-row "matrix" with contradictory rows
  (`p1_s2_q1`: row1 Ja✓, row2 Nein✓);
- emits conditionals both merged into the parent's rows AND as separate groups;
- keeps unanswered conditionals as visible groups.
Likely tied to the thinking model over-elaborating. Candidate fixes: remove thinking (option 1/2
above), add a **"remove this question" control** to the review UI (quick, model-independent
backstop), and/or reconsider the `PADDLE_ID` mark-only contract so TextLab owns structure and the
VLM only reports marks.

## Repro / environment notes
- Shared Ollama store: `/storage/research/dsl_shared/solutions/ondemand/text_lab/container/models/ollama`
  has `qwen3-vl:30b` (20 GB), `gemma4:26b`, `gemma3:27b`, `glm-ocr`, etc. No local qwen-VL besides
  qwen3-vl:30b; the other `qwen3.x` are text-only.
- Headless qwen probes this session ran via `apptainer exec --nv --bind /storage:/storage <sif>`
  launching `ollama serve` (OLLAMA_MODELS=shared store) then a Python driver — pattern saved in the
  session scratchpad runners.
- The `gemma_survey_audit_*` saved-crop dirs were deleted from the working tree (untracked scratch);
  reconstruct crops from `split_1_2.pdf` + bboxes in `split_1_2.json` if needed.
- Container Ollama is 0.22.1 (client warned 0.24.0). `process_document(extract_survey=True,
  vision_client=…)` is the headless entry point; it renders at SURVEY_DPI=300 and shells out to the
  `paddle_vl_backend` env for layout/OCR.

## Suggested first moves next chat
1. Confirm the running app is on the edited tree (blocker above); if not, fix that and re-test —
   many of the reported symptoms may simply vanish.
2. If code is live but structure is still messy → decide the thinking-tax fix (Ollama upgrade vs
   instruct pull) and add the "remove question" review control.
