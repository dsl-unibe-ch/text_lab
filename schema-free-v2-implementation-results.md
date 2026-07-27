# Schema-free survey VLM v2 — implementation results

Date: 2026-07-21

## Outcome

The v2 contract and deterministic safety layer are implemented. The compact
Gemma audit completed all 13 calls with valid JSON, but Gemma is not reliable
enough to approve for unattended survey extraction. It should remain behind
TextLab's existing `needs_review` release gate. Qwen3-VL-30B should be run
through the same saved-crop audit before choosing the production provider.

## Implemented changes

- Crops are clamped to Paddle question/column boundaries, so ordinary padding
  cannot include the previous or next numbered question.
- Explicitly numbered questions that disagree with the Paddle crop anchor are
  excluded as adjacent-question spill and recorded in provenance.
- The universal prompt is provider-neutral and remains close to the stable v1
  prompt length. It requires one empty-label row for simple questions and
  shared choices plus statement rows for matrices.
- Free-text-only conditional fields are not separate response groups;
  respondent text linked to a selected choice is stored as `associated_text`.
- Selection rules are derived by TextLab. A multiple-answer cue must be present
  in Paddle's printed section and scoped to a model-reconstructed multiple
  subquestion; a later follow-up cannot weaken an earlier yes/no question.
- One-row/matrix structural violations, empty required matrix rows, multiple
  marks in a single-choice row, uncertain visual marks, positional echo
  mismatches, and zero-mark sections are review triggers.
- A simple question whose reconstructed choices are fewer than independently
  detected controls is a review case.
- Duplicate labels and labels containing response glyphs or serialized-output
  leakage are review cases. This catches valid-JSON corruption such as
  `"Frau',"` and age labels that absorb printed circle glyphs.
- `schema-free-v2` is the default. The old `schema-free-v1` command spelling is
  accepted as an alias to v2 rather than silently invoking the weaker path.
- Audit metadata preserves the exact crop, prompt, response schema, raw Ollama
  envelope, parsed content, generation settings, and normalized document.

## Audit runs

| Directory | Valid JSON | Output tokens | Model time | Purpose |
|---|---:|---:|---:|---|
| `gemma_survey_audit_schema_free_v1/` | 12/13 | 6,855 | 66.4 s | Previous baseline |
| `gemma_survey_audit_schema_free_v2/` | 13/13 | 4,724 | 70.0 s | Repetition-penalty experiment; rejected |
| `gemma_survey_audit_schema_free_v2_final/` | 12/13 | 7,464 | 90.8 s | Overlong prompt experiment; rejected |
| `gemma_survey_audit_schema_free_v2_compact/` | 13/13 | 4,421 | 65.5 s | Current compact prompt/schema |

The first two v2 experiments are retained because they show why explicit
repetition penalties and a longer, more prescriptive Gemma prompt were removed.
The compact directory is the relevant review artifact for the current prompt.

## What improved

- Q10's crop no longer contains the Q11 heading.
- Q3's crop no longer includes handwriting from the preceding question.
- Q15 returns valid JSON and associates the phone number with the selected
  `Ja` choice instead of looping over legal text.
- Empty matrix rows are localized for review instead of looking like confident
  unanswered rows.
- The Q9 double-mark/correction and Q10 positional shifts are surfaced through
  rule and echo disagreements.

## Remaining Gemma failures

The compact run demonstrates run-to-run semantic instability even with
temperature 0 and seed 42:

- one complete conditional block in Q2 was omitted despite being clearly
  visible in the crop;
- Q8 duplicated/collapsed the full vertical label list into each choice;
- Q10 split endpoint labels from their numeric values, shifting positions, and
  missed two visible rows;
- Q11 returned only a corrupted `Frau` label;
- Q12 included printed circle glyphs inside every age label;
- the corrected/overwritten marks in Q3/Q9/Q10 were not interpreted
  consistently across runs.

The safety layer now flags these observed classes, but flagging bad extraction
is not the same as producing a usable answer dataset. This is why Gemma should
not be approved merely because every call parses.

## Recommended provider decision

Run Qwen3-VL-30B against the exact 13 PNG files in
`gemma_survey_audit_schema_free_v2_compact/vlm_calls/`, using the exact compact
prompt and response schema saved beside each image. Compare semantic accuracy,
structure accuracy, missed-mark rate, review recall, output validity, latency,
and GPU cost. If the user's manual Qwen result reproduces across all sections,
make Qwen the survey provider and retain Gemma only as an explicitly
experimental fallback whose output always requires review.

## Verification

The deterministic enrichment, markup, noisy-scan, and regression suites pass.
The enrichment suite includes new coverage for boundary spill, scoped
multiple-answer cues, simple/matrix structural invariants, missing matrix
marks, uncertain visual marks, control-count mismatch, and choice-label format
leakage.
