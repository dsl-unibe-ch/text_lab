# Review — `schema-free-survey-vlm-proposed-plan.md`

Date: 2026-07-21
Reviewer: Claude (Opus 4.8)
Reviewed: `schema-free-survey-vlm-proposed-plan.md` (2026-07-21), against
`ocr-page-implementation-plan.md`, `markup-detection-current-state.md`,
`markup-detection-proposed-plan.md`, `gemma-survey-audit-comparison.md`, the
`feature/auto-ocr` working tree, and the
`gemma_survey_audit_complete_sections_v2/` audit output.

---

## 0. Verdict

**Approve the direction. Do not approve Phase 1 as written.**

The diagnosis in §2 is correct and I verified it independently against the code
and the audit CSV — the Paddle-derived semantic schema is genuinely broken in
ways that threshold-tuning cannot fix, and it is corrupting the very
provider-comparison the previous plan depends on. Dropping it is right.

But the proposal removes the Paddle schema without replacing the one thing that
schema provided: **positional grounding that TextLab can independently verify.**
Under the new contract, `choices` and `marked_answers` come from the same
forward pass, so `choice_position` is an ordinal into a list the model invented
in the same breath. A mis-segmented choice list produces a *consistently*
mis-indexed mark, and nothing downstream can detect it. The current contract's
failures are ugly but *visible* (`unknown_option_frau`, `option 4`) — that is
how Q4/Q11/Q15 got flagged. The new contract converts detectable failures into
silent, plausible-looking ones.

That trades away the property `markup-detection-proposed-plan.md` §8 names as
the top release gate: *"review recall is prioritised over the percentage
auto-accepted."* Four blocking changes (§3) restore it. With those, this is the
right design.

---

## 1. What I verified

Every claim in §2.2 checks out. Sources: `gemma_survey_audit_complete_sections_v2/form_responses.csv`
and [form_extract.py](src/core/form_extract.py).

| Claim | Verified | Evidence |
|---|---|---|
| Q7 answer leakage into prompt | **Yes — worst one** | CSV `question` column literally contains `✗ Bevölkerung hat eher Nachteile`. The observed answer is in the prompt text. |
| Q8 five choices collapsed to two | Yes | CSV `selected` = `"Er ist wichtig Ich bin unschlüssig / habe keine Meinung Er ist unwichtig Er ist absolut unwichtig…"` — four labels in one option. |
| Q9/Q10 `option N` column labels | Yes, and I found the root cause | [form_extract.py:408-412](src/core/form_extract.py#L408-L412): the backward header search breaks on `any(header[i].strip())`, but a preceding *response* row's cells contain `○` — non-empty. A response row is accepted as the header, its `○` labels fail `_label_is_resolved`, so every later row falls back to `option N`. |
| Q4/Q11/Q15 `rows: []` | Yes | [form_extract.py:442-443](src/core/form_extract.py#L442-L443): `_text_schema` returns `[]` when Paddle transcribed no glyph. The section is still sent (candidate evidence came from `paddle-binary-options` or geometry), so the model gets an empty allow-list → `unknown_option_ja`, `unknown_option_frau`, `unknown_option_mittelm_ssig`. |
| Q12 labels-before-controls | Yes | [form_extract.py:446-448](src/core/form_extract.py#L446-L448): labels are taken from text *after* each glyph, so age ranges printed before their circles all land in the row label. CSV row label = `"16-24 25-34 35-44 45-54 55-64 65-74 75 und älter"`. |
| Q3 subquestions merged | Yes | CSV shows one 13-choice row plus `unknown_option_p1_r31_o1_radio_button_nein` — the model synthesized an ID for the missing primary `Ja/Nein`. |

Two things the plan does not say that a reader needs:

- **Every row in that CSV is `needs_review`.** No model is in
  `TEXTLAB_APPROVED_SURVEY_MODELS`, so [form_extract.py:1134](src/core/form_extract.py#L1134)
  gates the whole output. Nothing is auto-accepted today. State this — the plan
  reads as closer to shipping than it is.
- **Section boundaries are also still wrong.** Q2 emits rows `p1_r25/r26/r27`
  with everything empty — the section swallowed following content. §3 assigns
  boundaries to Paddle and moves on, but boundaries are now *more* load-bearing
  than before (§3.3 below).

---

## 2. What I approve without change

- **Retiring the Paddle semantic schema.** Correct, and the evidence supports it.
- **One complete question section per call.** The 43→13 call reduction with
  95.2s→50.4s model time is the most solid empirical result in the whole
  document set. Don't relitigate it.
- **The model never constructs IDs; TextLab assigns them post-hoc** (§5). Right
  division of labour.
- **Shared `choices` + `rows` for matrices** as a *wire format*. Q10 as six
  choices × 16 rows instead of 96 labelled objects is the correct call.
- **§7.3 failure policy** — no repair-guessing, no Paddle fallback as accepted
  answer, no retry-as-independent-evidence. Keep verbatim.
- **§14 audit/privacy requirements**, and audit mode off by default.
- **§16's refusal to delete the old path or crown a provider before A/B.**

---

## 3. Blocking changes

### 3.1 `choice_position` needs external grounding — add an echo check now, boxes if they work

This is the central problem. Two concrete fixes, in order of cost:

**(a) Free, do it unconditionally.** Require `marked_answers[].choice_text` as an
echo, and validate it against `choices[choice_position - 1].choice_text` after
parsing. Mismatch → `needs_review`. This costs a handful of output tokens and
catches the entire off-by-one / mis-segmentation class — exactly the Q8 failure
mode — which position-only output cannot catch at all.

**(b) Benchmark, then decide.** This is my answer to your Q8 (§4.9 below): ask
for a normalized bbox per choice, and validate positionally rather than trusting
it as coordinates. Concretely — take the frozen geometric detector and use it
for what it is actually good at. `find_marks` is unreliable at deciding
*whether* a circle is crossed (`markup-detection-current-state.md` §6 is
emphatic about this) but reasonably good at finding *that* there are N aligned
circles with a regular pitch. So:

- count agreement: |aligned contours| vs. |choices| → mismatch is a review reason;
- ordering agreement: model choice order vs. reading-order sort of the contours;
- ink-at-position: does the marked choice's box contain ink above the
  background-anchored floor?

That repurposes geometry from "second opinion on state" (fragile, where it kept
trading false positives for misses) to "verifier of positional grounding"
(robust). It is a much better use of `markup_detect.py` than §7.2's unquantified
*"geometric evidence strongly disagrees with the selected position."*

If Gemma's boxes turn out unusable — likely; small VLMs are weak at coordinates —
fall back to (a) plus count/order agreement, which needs no boxes from the model
at all.

### 3.2 `selection_rule` must not be model-reported when it gates review

§4 has the model return `selection_rule`, and §7.2 makes the review triggers
depend on it: *"a `zero_or_one` row has multiple selected answers"*, *"a
`one_per_row` matrix row has multiple selected answers."*

A model that returns `zero_or_more` for a single-choice question suppresses its
own review trigger. Self-reported constraints cannot gate self-reported answers.

Today this is derived deterministically by
[`_infer_question_contract`](src/core/form_extract.py#L641) from structure plus
printed cues (`"Mehrere Antworten möglich"`). **Keep deriving it in TextLab.**
Accept the model's value as an observation only, and when the two disagree, that
disagreement is itself a review reason. This is a soundness bug, not a
preference.

### 3.3 Add a section-boundary metric — the crop is now the only evidence

Under the old contract a clipped crop still had the full Paddle schema as a
safety net; the model could map to an option even if its pixels were cut off.
Under the new contract, **what is outside the crop does not exist.** A section
that clips two of five choices yields a confident, internally consistent,
schema-valid, completely wrong answer, and none of §7.1's checks fire.

`markup-detection-proposed-plan.md` §8 had *"form-region proposal recall"* as a
metric. §11.3 dropped it. Put it back, strengthened:

- boundary precision/recall against hand-drawn section boxes;
- **choice-clipping rate** — sections where a printed control is cut by the crop
  edge. This is the one that silently poisons results.

Q2's `r25/r26/r27` in the current CSV says this is not hypothetical.

### 3.4 Add an under-selection check

`state` correctly drops `unselected`, so `marked_answers` holds only marked
choices. The consequence: **a section that returns zero marks is perfectly
valid.** §7.1 catches "a large visible section unexpectedly returns no questions
or rows" but not "returns rows with no marks at all."

This matters because the prompt is heavily biased toward *not* emitting marks —
the current production prompt repeats the anti-false-positive instruction four
times ([form_extract.py:769-784](src/core/form_extract.py#L769-L784)), and the
proposed §6 prompt keeps that pressure. The current CSV already shows Q10 rows
`r4` and `r14` coming back empty in a survey where the respondent answered
nearly everything.

Add: **zero marked answers across an entire section → `needs_review`**, and
cross-check against Paddle glyph observations, which are weak at *which* option
is marked but decent at *whether there is a mark somewhere in this region*.

---

## 4. Answers to §15

**Q1 — is `questions → shared choices → rows → marked_answers` universal?**
Nearly. Real patterns it cannot hold, which should be named as out-of-scope
rather than discovered later:

- **Ranking questions** (respondent writes 1/2/3 into boxes) — the response is
  an ordinal, not a mark. Not representable. Declare out of scope.
- **Continuous / mark-anywhere-on-a-line scales** — same. Declare out of scope.
- **A matrix with one extra column outside the shared set** (a per-row `N/A` or
  `weiss nicht`) — §4.2's rule forces a split into separate questions, which
  fragments a visually single matrix and wrecks the CSV shape. Common in real
  surveys. Allow an optional per-row `extra_choices` rather than splitting.
- **Conditional follow-ups** — see Q3 below.

**Q2 — return every printed choice, or only marked ones?**
**Every choice.** It is the anchor `choice_position` indexes into and the only
thing count-validation and template alignment can compare against. Returning
only marked labels makes positions meaningless.

But state the cost honestly: this **undoes the measured win** from the ID-only
contract (43/43 parseable, −22% runtime, no variable labels — per
`gemma-survey-audit-comparison.md`). You are re-accepting long-label
transcription variance. Mitigate with fuzzy matching, not exact equality — which
means §5's normalization list (Unicode, whitespace, punctuation) is **too strict
for template alignment**. Add normalized edit-distance matching there, and never
gate on exact label equality.

**Q3 — split Q3-style subquestions, or rows with different choice sets?**
**Split — but record the relationship.** Add optional `parent_question_index` and
`condition_text`. Pure flattening loses the fact that Q3b is only meaningful when
Q3a = `ja`, and a downstream analyst then cannot distinguish an unanswered
conditional from a missing answer. That distinction is the whole point of a
survey export.

**Q4 — is `associated_text` appropriate here?**
Yes, keep it, scoped exactly as §9 has it. A selected `Andere: ______` is one
response; splitting it across two enrichments would force a join on data that has
no join key. Add one rule: `associated_text` without a marked choice in the same
row → `needs_review` (§7.2 has this — good, keep it).

**Q5 — is position-first same-layout alignment safe?**
Position-first is right, but make these drift checks mandatory and *blocking*,
not warnings: question count, per-question choice count, per-question row count,
and normalized-label similarity below threshold on any of them. Any failure →
that document is extracted standalone, not aligned to the frozen template. The
silent-corruption risk here is a template frozen from a document whose first-pass
structure was itself wrong.

**Q6 — should the first document define the template?**
First document, **but only after its structure passes validation**, and record in
provenance which document defined it. Requiring a separate template-building pass
adds a manual step the support use case will not reliably perform. Add a UI
affordance to re-freeze from a different document when the first one turns out to
be a bad exemplar.

**Q7 — two-stage structure-then-marks?**
**Not as two calls per section** — you double latency and the second call still
needs the structure in context, so the token saving is illusory.

**But the underlying idea is the strongest thing available to you, at the batch
level.** Combine it with §8:

1. First document: schema-free universal contract → structure + marks.
2. Validate and freeze that structure (with review).
3. Every subsequent respondent: **ID-based mark-only contract against the frozen
   structure.**

That gets the reliability of a validated allow-list *and* the measured
efficiency of the ID-only contract, with the schema now coming from a VLM that
saw the pixels and was checked once — instead of from Paddle, unreviewed, as
today. Cost amortizes across the batch, which is exactly the shape of the
support request.

§8 currently says *"run the same universal VLM prompt on each respondent's
complete section,"* which throws the frozen structure's main value away. **Change
§8 to switch contracts once the template is frozen.** This is my most important
non-blocking recommendation.

**Q8 — bounding boxes in the response?**
See §3.1 — yes, benchmark them, but the echo check in 3.1(a) is the part you
should ship regardless, because it works even if the boxes are garbage.

**Q9 — release thresholds?**
Don't set numbers before seeing label prevalence, but fix the *shape* now:

- **False selections on the held-out set: zero tolerated silently.** Every one
  must be caught by a review trigger. This is the asymmetric-cost error — a wrong
  answer that looks confident is worse than ten flagged uncertains.
- **Review recall ≥ 95%** of wrong automatic answers flagged, measured on
  held-out data. This is the release gate; auto-accept rate is a *reported
  metric*, not a gate.
- **Selected-mark recall must materially beat the Paddle-glyph baseline** —
  otherwise the whole enrichment is unjustified.
- **Exact-section accuracy** reported but not gated initially.

**Q10 — is Qwen the benchmark favourite?**
No. Nothing in the current evidence supports a favourite: as §2.3 correctly
concludes, the observed failures are input-contract bugs, not model-quality
signal. Naming a favourite pre-benchmark invites reading the results toward it.
Qwen-30B-A3B is worth testing, and its weights genuinely do not leave safe margin
on the 4090 (`markup-detection-proposed-plan.md` §6) — so note that a Qwen win
carries a **deployment cost** Gemma does not: a larger node, or GPUStack and the
offline-boundary change that comes with it. That cost belongs in the decision,
not just accuracy.

---

## 5. Non-blocking, worth fixing

**5.1 — The §11.1 A/B config 1 is not decision-grade.** Comparing against the
current Paddle-derived prompt measures "is the new thing better than a
known-buggy thing." Two of the bugs are ~10 lines: the Q7 leakage (strip mark
glyphs from `question_text` before prompting) and the `_table_schema` header
search accepting a `○`-bearing response row as the header. **Fix both, then
A/B.** A repaired Paddle schema is also still the better contract for the frozen
batch path in Q7 above, so the fix is not throwaway work.

**5.2 — The IR migration is one bullet and shouldn't be.** §12 Phase 1 says
"adapt the current IR/export mapping to shared choices plus rows."
[`FormOption`](src/core/doc_ir.py#L150) currently nests inside `FormRow` with a
`state` field, and [`form_responses_to_dataframe`](src/core/doc_ir.py#L587)
filters `row.options` by state. Changing the IR touches `doc_ir`, the
`form_extract` merge, the Responses tab in `OCR.py`, the CSV, the JSON bundle,
and the tests.

**Recommendation: don't change the IR.** Materialize the cross-product back into
`FormRow.options` at parse time. §4's own justification for shared choices is
purely output size ("keeps large matrices compact") — that is a prompt-economics
argument, not an IR-design argument. Keep the wire format compact and the IR as
it is. Say so explicitly in the plan, or this quietly becomes a multi-file
refactor.

**5.3 — `unmapped_marks` loses its meaning.** Under the ID-only contract it had a
clear job: ink matching no supplied ID. Now the model owns the structure, so it
can always invent a question to hold any mark — a well-behaved model will
essentially never populate it. Either define it operationally (marks outside any
question region: margin notes, stray ink) or drop it. Critically: **an empty
`unmapped_marks` is not evidence of completeness** and must not be treated as
such.

**5.4 — Pin generation settings; keep the cache.** The previous plan required
`think:false`, `temperature 0`, fixed seed, bounded output budget, and a cache
keyed on crop bytes + schema + provider + model + prompt version
(`markup-detection-proposed-plan.md` §5.5, §11). §10 only says "same generation
settings" without pinning them. Since long-label transcription *raises*
run-to-run variance, and §11.3 wants to measure consistency, pin them explicitly
and add `contract_version` to the cache key.

**5.5 — Keep the good parts of the current prompt.** §6's draft drops the
strongest anti-false-positive line, and Q8/Q13 over-selection is an *observed*
failure. Carry over verbatim: *"A printed circle with a white/empty centre is NOT
a response. Before returning any mark, verify that separate handwritten ink
visibly crosses or fills that option's centre; never list all printed options
merely because their outlines are visible."* Also keep the framing that this is a
high-resolution scan of a paper form and that OCR glyphs must be ignored — §6
drops both.

**5.6 — Don't validate the fix on the document that motivated it.** Phase 0
commits to labelled JSON for "the existing 13 complete sections" — one survey.
`markup-detection-proposed-plan.md` §8 required 10–20 real pages across several
form designs and explicitly warned against deciding on `split_1_2.pdf` alone.
That warning applies with more force now, because a contract tuned on one German
survey's layout is exactly how the current contract got here. Keep §11.2's
example list as the **Phase 4 gate**, and label Phase 1's single-document A/B a
smoke test, not evidence.

---

## 6. Suggested revised phasing

| Phase | Change from the plan |
|---|---|
| **0** | As written, **plus**: fix the two Paddle-schema bugs (5.1) so the baseline is honest; state that all current output is `needs_review` behind the approval gate. |
| **1** | Universal contract **with** `choice_text` echo (3.1a), TextLab-derived `selection_rule` (3.2), under-selection check (3.4). IR unchanged — materialize choices into `FormRow.options` (5.2). Single-document A/B = smoke test only. |
| **1b** *(new)* | Geometry-as-position-verifier: count/order/ink-at-position agreement (3.1b). Benchmark model bboxes here; drop them if unusable. |
| **2** | As written, plus boundary + choice-clipping metrics (3.3). |
| **3** | Same-layout templates, **switching to the ID-based mark-only contract once the structure is frozen** (Q7). Mandatory blocking drift checks (Q5). |
| **4** | Provider benchmark on the **multi-document** held-out set (5.6). Gate on false-selection and review recall (Q9). Include Qwen's deployment cost in the decision (Q10). |
| **5** | As written. |

---

## 7. Bottom line

The plan correctly identifies that the input contract, not the model, is what
broke — and the evidence for that is solid and independently verifiable. The
schema-free direction is right and I'd approve it.

The gap is that it hands the model both the structure and the answer and then
validates the answer *against the structure the model supplied*. Four fixes close
that loop: the `choice_text` echo, TextLab-derived selection rules, geometry as a
position verifier, and an under-selection trigger. All four are cheap relative to
the benchmark work already planned.

And the batch insight in Q7 — schema-free once, validated, frozen, then ID-based
mark-only for the remaining respondents — is worth more than the contract choice
itself for the actual support use case, which is a stack of identical
questionnaires.
