# Prompt complexity experiment: does a short prompt help the survey VLM?

Date: 2026-07-22
Hypothesis (raised in review): the current ~430-word prescriptive prompt may confuse
the VLM; a very short natural-language prompt + a second LLM restructuring call might do
better.

Method: reused the 13 saved crops + response schemas (`gemma_survey_audit_schema_free_v2_compact/`).
Only the prompt varied; schema (grammar-constrained), model settings (temp0/seed42,
num_ctx 8192, num_predict 3000), and 3 reps were held constant. Ollama local, RTX 4090.
Second LLM call was NOT re-tested — the earlier two-pass experiment
(`qwen-two-pass-survey-experiment-results.md`) already showed the normalization call
*reduced* validity (87.2% -> 82.1%) and nearly doubled latency.

Prompts:
- LONG = current textlab-survey-schema-free-v2 prompt.
- SHORT-bad = "...a choice that is crossed out ... is NOT a real answer -> cancelled..."
- SHORT-corrected = "...an X, tick, or filled circle over an option means SELECTED;
  use cancelled only if struck through or overwritten..."

## Results

| Model | Prompt | Valid JSON | Mean s | selected | cancelled | call_0007 |
|---|---|---:|---:|---:|---:|---|
| qwen3-vl:30b | LONG | 35/39 | 14.6 | 56 | 3 | selected/x (correct) |
| qwen3-vl:30b | SHORT-bad | 30/39 | 13.7 | 9 | 53 | cancelled/x (WRONG) |
| qwen3-vl:30b | SHORT-corrected | 30/39 | 11.7 | 22 | 37 | selected/x (correct) |
| gemma4:26b | LONG | 37/39 | 5.2 | 88 | 0 | — |
| gemma4:26b | SHORT-bad | 35/39 | — | 15 | 36 | — |
| gemma4:26b | SHORT-corrected | 36/39 | — | 44 | 0 | — |

## Findings

1. **The short prompt does not help; the detailed prompt is load-bearing.** Neither model
   beat the long prompt on JSON validity. The long prompt's explicit mark-semantics
   paragraph ("a normal X/tick/fill over a printed circle is *selected*; a dense overwrite
   *may* be cancelled; if unclear use ambiguous") is what keeps the model calibrated to this
   survey's confusing convention where **an X drawn through a circle is the SELECTION**.

2. **"Deleted-answer" instructions are dangerous on X-as-selection forms.** Telling the
   model that crossed-out marks are not answers made BOTH models relabel legitimate X
   selections as `cancelled` (gemma 0->36, qwen 3->53). The corrected wording recovered
   gemma (cancelled back to 0) but qwen stayed unstable (still 37 false cancellations).
   Any deletion guard must precisely separate a selection-X from a retraction, and must be
   validated on crops that actually contain a corrected/struck-through answer — the current
   13-crop set has none (all marks are clean single X selections).

3. **call_0010 fails in every prompt condition** because it is a 16-row x 6-column rating
   matrix: the output overruns num_predict=3000 (done_reason=length). This is a token-budget
   / structure problem, independent of prompt style. Fix = scale num_predict with expected
   rows (and/or split matrices into row bands) plus a retry on done_reason=length.

## Conclusion

Keep the configuration from `vlm_model_bench_20260722/VERDICT.md`: **qwen3-vl:30b +
the detailed prompt + grammar-constrained schema**, gemma4:26b as the fast fallback,
`needs_review` gate retained. Do NOT switch to a short prompt, and do NOT add the second
restructuring call. Do NOT add a naive "ignore deleted answers" instruction. Separately,
fix matrix runaways with a row-scaled num_predict + length-retry.
