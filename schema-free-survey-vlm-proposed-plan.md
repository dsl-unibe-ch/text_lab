# Schema-free survey VLM extraction — proposed plan

Date: 2026-07-21  
Status: proposal for review; not yet implemented  
Scope: the optional **Extract survey/form responses** enrichment in TextLab

## 1. Decision summary

Keep PaddleOCR-VL as the document-layout and candidate-section detector, but do
not pass Paddle's inferred question/row/option schema to the response VLM.

For every candidate markup section:

1. Paddle identifies the complete question-section boundary.
2. Conservative OCR/visual evidence decides whether the section contains form
   controls.
3. The complete, unmodified 300-DPI section image is sent to the selected VLM.
4. Gemma or Qwen reconstructs the visible form structure and respondent marks
   using one universal structured-output contract.
5. TextLab assigns deterministic IDs from array positions after parsing.
6. Paddle glyphs, geometry and selection rules are used only as independent
   validation/review evidence after the VLM responds.

The proposed flow is:

```text
Paddle layout
  -> complete question sections
  -> conservative markup-candidate filter
  -> one complete section image per VLM call
  -> universal schema-free response
  -> deterministic TextLab IDs
  -> validation against geometry/Paddle/template
  -> review UI and exports
```

This proposal replaces only the semantic prompt/schema construction. It does
not change ordinary OCR, figure descriptions, the opt-in survey control, or the
current release gate that marks all unapproved-model results for review.

## 2. Why change the current approach

The original goal of the Paddle-derived schema was sound: give the VLM stable
IDs and prevent it from rewriting the form. The complete-section audit showed
that this helps only when Paddle reconstructs the form correctly. When the
schema is incomplete, it actively prevents a correct answer.

### 2.1 Audit progression

Input: `split_1_2.pdf`  
Model: local Ollama `gemma4:26b`

| Configuration | Calls | Parseable | Errors | Relevant timing |
|---|---:|---:|---:|---:|
| Rich nested VLM response on snippets | 43 | 42 | 1 | 223.99 s wall time |
| Canonical ID-only response on snippets | 43 | 43 | 0 | 175.41 s wall time |
| Canonical ID-only response on complete sections | 13 | 13 | 0 | 50.4 s summed model time |

Moving to one complete question section per call fixed the context fragmentation
problem and reduced the call count. It did not fix incorrect Paddle-derived
semantic schemas.

Review material:

- [Complete-section contact sheet](gemma_survey_audit_complete_sections_v2/REVIEW.md)
- [Complete-section response CSV](gemma_survey_audit_complete_sections_v2/form_responses.csv)
- [Audit comparison](gemma-survey-audit-comparison.md)

### 2.2 Per-question schema findings

| Question | Current prompt quality | Finding |
|---|---|---|
| Q1 | Good | Ten rating choices have usable labels and stable IDs. |
| Q2 | Mostly good | Choice mapping is usable, but the question text is truncated to its first line. |
| Q3 | Invalid | The primary `Ja/Nein` row is absent. Two conditional subquestions with different choice sets are merged into one 13-choice row. Gemma invents an ID for `Nein`. |
| Q4 | Invalid | `rows` is empty, so `Mittelmässig` cannot be mapped to a supplied ID. |
| Q5 | Good | Five choices are represented correctly. |
| Q7 | Contaminated | The Paddle-selected `✗` remains in `question_text`, leaking the observed answer into the prompt. |
| Q8 | Invalid | The image has five choices; the prompt has two, with the last four labels collapsed into choice 2. The visually marked third choice is therefore returned as choice 2. |
| Q9 | Partial | Only the first matrix row gets real column labels; later rows use `option 1` through `option 5`. |
| Q10 | Partial | The same header-propagation problem is repeated across 16 rows, producing a 7.3 KB prompt. |
| Q11 | Invalid | Four visible gender choices are present, but `rows` is empty. Gemma uses `Frau` as an invented ID. |
| Q12 | Invalid | The image contains seven age ranges; the prompt creates six generic choices and places all age labels in the row label. |
| Q13 | Good | Eight locations have usable labels and IDs. |
| Q15 | Invalid | The visible `Ja/Nein` controls are absent from the schema, leaving `rows: []`. |

### 2.3 General prompt/schema problems

1. All calls use the same generic JSON schema in which `option_id` and
   `visible_row_ids` accept arbitrary strings. Structured output therefore
   guarantees JSON shape, but cannot prevent invented IDs.
2. Removing `paddle_state` did not remove answer leakage. Checked/unchecked OCR
   glyphs can still occur inside `question_text`.
3. `_text_schema` assumes every mark glyph belongs to one logical row. It
   cannot represent sections such as Q3 with multiple subquestions.
4. `_table_schema` can mistake the preceding response row for the matrix header,
   which turns meaningful column labels into `option N`.
5. A correct VLM observation cannot be reconciled with a choice that the input
   schema omitted. Q8 is the clearest example.

These are input-contract failures rather than useful evidence about Gemma
versus Qwen accuracy. A provider comparison should not be based on them.

## 3. Responsibilities after the change

### PaddleOCR-VL remains responsible for

- page and region layout;
- column/spread-aware question-section boundaries;
- printed OCR shown in the Document and Layout preview tabs;
- candidate evidence such as recognised circle/square glyphs, checkbox layout
  regions and compact `Ja`/`Nein` rows;
- supporting observations used after VLM extraction;
- same-layout section coordinates learned from the first batch document.

### The response VLM becomes responsible for

- identifying every logical question/subquestion inside the supplied section;
- identifying shared choices and matrix rows from the pixels;
- transcribing the visible question, row and choice labels;
- locating respondent-added marks;
- classifying selected, cancelled and ambiguous marks;
- returning the choice position associated with each mark.

### TextLab remains responsible for

- deterministic IDs;
- schema validation and normalization;
- selection-rule validation;
- disagreement and uncertainty handling;
- template alignment in same-layout batches;
- evidence preservation, review UI and exports;
- deciding which model/provider has passed the release benchmark.

The model must never be asked to construct IDs.

## 4. Universal response contract

All markup sections use the same top-level and nested structure. Simple
questions, checkbox lists, ratings and matrices differ only in the number of
questions, rows and shared choices they contain.

```json
{
  "questions": [
    {
      "question_text": "How satisfied are you?",
      "response_type": "single",
      "selection_rule": "zero_or_one",
      "choices": [
        {"choice_text": "Dissatisfied"},
        {"choice_text": "Neutral"},
        {"choice_text": "Satisfied"}
      ],
      "rows": [
        {
          "row_text": "",
          "marked_answers": [
            {
              "choice_position": 3,
              "state": "selected",
              "visual_mark": "x",
              "associated_text": ""
            }
          ]
        }
      ]
    }
  ],
  "unmapped_marks": []
}
```

### 4.1 Fixed enums

`response_type`:

- `single`
- `multiple`
- `rating`
- `matrix`
- `unknown`

`selection_rule`:

- `zero_or_one`
- `zero_or_more`
- `one_per_row`
- `unknown`

`state`:

- `selected`
- `cancelled`
- `ambiguous`

`visual_mark`:

- `x`
- `tick`
- `filled`
- `scribbled`
- `other`
- `uncertain`

### 4.2 Structural conventions

- Questions are returned in visual reading order.
- Every question/subquestion with a different choice set is a separate item in
  `questions`.
- `choices` contains every printed choice once, including unselected choices.
- A simple question, rating or checkbox list has one row with `row_text: ""`.
- A matrix has shared `choices` and one row per visible statement.
- Every visible matrix row is returned, even when `marked_answers` is empty.
- `choice_position` is one-based and refers to the corresponding entry in
  `choices`.
- `marked_answers` contains only respondent-marked choices. Clean printed
  outlines are omitted.
- `associated_text` contains respondent text visibly linked to that marked
  choice, such as a selected `Other` field; otherwise it is an empty string.
- If different rows genuinely have different choice sets, they are represented
  as separate question/subquestion objects rather than forcing incompatible
  choices into one matrix.
- `unmapped_marks` is reserved for visible response ink that cannot be attached
  to a logical question/row/choice after inspecting the whole section.

This structure keeps large matrices compact. Q10 needs six shared choices and
16 rows, rather than repeating 96 labelled choice objects.

## 5. Deterministic IDs and normalization

TextLab assigns identifiers only after successful schema validation:

```text
p{page}_s{section}_q{question_index}
p{page}_s{section}_q{question_index}_r{row_index}
p{page}_s{section}_q{question_index}_c{choice_position}
```

Indices follow the returned visual order and are one-based. The model never
sees or returns these IDs.

Normalization is deliberately limited to:

- Unicode normalization;
- whitespace collapsing;
- removal of duplicated surrounding punctuation;
- consistent empty strings for absent row/associated text.

TextLab must not paraphrase labels or silently merge/split model-returned
questions. Structural changes require review or template evidence.

## 6. Proposed universal prompt

The production wording should remain short enough to leave visual/context
budget for large matrices. A starting point is:

```text
Read only the visible form structure and respondent-added response marks in
this complete question section.

Reconstruct every logical question or subquestion, its printed choices, and
its rows. Questions with different choice sets must be separate question
objects. A matrix has one shared choice list and one row per statement.

Return every printed choice and every visible matrix row, including unanswered
rows. In marked_answers return only choices with visible respondent-added ink.
A clean printed circle or square is unselected and must not be returned as a
mark. Never infer an answer from wording, conditional logic, or expected form
rules. Use cancelled or ambiguous when the visible intent is not clear.

choice_position is the one-based visual position in the question's choices
array. Do not create identifiers. associated_text is only respondent text
visibly linked to a marked choice; otherwise return an empty string.

Return only JSON matching the supplied response schema.
```

The prompt receives no Paddle question text, labels, option IDs, glyph states or
geometric classifications.

## 7. Validation and failure policy

### 7.1 Structural validation

Reject or flag the response when:

- the top-level JSON schema is invalid;
- a question has no choices or rows;
- `choice_position` is outside the corresponding `choices` array;
- a row returns the same choice position more than once;
- a matrix has no row labels where visible row text is expected;
- question/row/choice counts disagree with a frozen same-layout template;
- a large visible section unexpectedly returns no questions or rows.

### 7.2 Semantic validation

Mark `needs_review` when:

- a `zero_or_one` row has multiple selected answers;
- a `one_per_row` matrix row has multiple selected answers;
- any state is `cancelled` or `ambiguous`;
- VLM marks disagree with Paddle checked-glyph observations;
- geometric evidence strongly disagrees with the selected position;
- associated text appears without a marked choice;
- template positions or labels drift beyond the accepted tolerance.

Paddle/geometry disagreement must not overwrite the VLM response. It remains a
separate observation and review reason.

### 7.3 Failure behavior

- Preserve the full crop, prompt, raw envelope and raw content.
- Emit a failed/review group rather than a guessed response.
- Do not fall back to Paddle's mark state as the accepted answer.
- Do not retry the same model repeatedly and treat agreement as independent
  evidence.
- Keep every unapproved provider behind the existing release gate.

## 8. Same-layout batch processing

The schema-free contract is compatible with the existing **All files use the
same questionnaire layout** option.

### First document

- Learn normalized section coordinates from Paddle.
- Store the normalized VLM structure: question order, question labels, choice
  labels, row order and row labels.
- Assign and freeze TextLab IDs from those positions.

### Later documents

- Reuse the normalized section coordinates.
- Run the same universal VLM prompt on each respondent's complete section.
- Align results to the frozen template primarily by question/row/choice
  position, with normalized text as secondary evidence.
- Keep labels and IDs from the frozen template; never copy response states.
- Flag changed counts, reordered labels or material text drift for review.

This avoids sending a potentially incorrect Paddle schema while retaining
stable CSV columns across respondents.

A future registered-template differencing implementation remains useful as an
independent provider/baseline, especially for large batches.

## 9. Mark-only scope

This proposal extracts markup responses and text directly associated with a
marked choice. It does not automatically expand into general handwriting
recognition.

- Q6-style standalone numeric/free-text answers remain outside this extractor.
- Q14-style open comments remain outside this extractor.
- A selected `Other` choice may carry `associated_text` because the text is
  part of that markup response.

Standalone handwritten-answer extraction can be added later as an independent
opt-in enrichment with its own schema and benchmark.

## 10. Provider strategy

The universal prompt/schema must be provider-neutral. The same images, prompt,
JSON schema and generation settings should be used for:

- local `gemma4:26b`;
- local Gemma 31B if it is staged and fits the selected node;
- university GPUStack `qwen3-vl-30b-a3b-instruct`;
- any later local Qwen deployment.

GPUStack remains private university infrastructure, but it still changes the
offline/local processing boundary. Remote use must remain explicit and must
send only the selected question sections, never the entire document.

## 11. Benchmark plan

### 11.1 A/B configurations

Run the same labelled complete-section crops through:

1. current Paddle-derived ID prompt;
2. schema-free universal prompt with Gemma;
3. schema-free universal prompt with Qwen;
4. geometry/Paddle-only frozen baselines;
5. same-layout template differencing where applicable.

Do not change crops or generation settings between Gemma and Qwen.

### 11.2 Required examples

The development/held-out sets must include:

- simple `Ja/Nein`;
- horizontal and vertical single-choice lists;
- multiple-choice checkbox lists;
- ratings;
- matrices with shared column labels;
- a section containing multiple conditional subquestions like Q3;
- omitted OCR control glyphs like Q4/Q11;
- labels-before-controls layouts like Q12;
- selected `Other` plus associated handwriting;
- cancelled and ambiguous marks;
- unanswered questions and matrix rows;
- ordinary tables/circles that are not form fields;
- same-layout respondent batches;
- layout/template drift.

### 11.3 Metrics

Measure independently:

- JSON parse rate;
- question/subquestion count accuracy;
- choice count and normalized label accuracy;
- matrix row count and normalized label accuracy;
- selected choice-position precision/recall/F1;
- false-selection rate on clean outlines;
- cancellation/ambiguity recall;
- associated-text accuracy;
- exact fully-correct section rate;
- seconds per section/document and first-load time;
- peak VRAM and provider failures;
- review rate and reasons.

Provider selection must be based on held-out exact response accuracy and
false-selection risk, not visual plausibility on the supplied survey alone.

## 12. Implementation phases

### Phase 0 — freeze the current comparison baseline

- Keep the current Paddle-derived path available only for A/B evaluation and
  rollback during development.
- Version audit outputs, prompts and normalized response schemas.
- Create labelled expected JSON for the existing 13 complete sections.

### Phase 1 — universal extraction contract

- Add the universal JSON schema and prompt.
- Remove Paddle schema content from VLM requests.
- Implement strict parser and structural validation.
- Add deterministic position-derived IDs.
- Adapt the current IR/export mapping to shared choices plus rows.

### Phase 2 — validation and review integration

- Reconcile VLM results with Paddle/geometry only after extraction.
- Preserve every observation and disagreement.
- Update response cards/CSV to display VLM-transcribed labels and deterministic
  IDs.
- Retain the full section as evidence.

### Phase 3 — same-layout templates

- Freeze the first valid section structure per batch.
- Align subsequent outputs by position and normalized text.
- Flag count/order/text drift without copying answers.
- Benchmark against registered image differencing.

### Phase 4 — provider benchmark

- Run Gemma and Qwen on identical development and held-out sets.
- Produce a per-question error report and runtime/VRAM comparison.
- Select the internal default only after the agreed release thresholds pass.

### Phase 5 — cleanup

- Remove the Paddle-derived semantic prompt if the universal approach wins.
- Keep Paddle candidate detection and post-response observations.
- Document the selected provider, privacy boundary and review limitations.

## 13. Tests to add

### Deterministic unit tests

- universal JSON-schema acceptance/rejection;
- deterministic ID generation;
- out-of-range and duplicate choice positions;
- simple, multiple, rating and matrix normalization;
- multiple subquestions with distinct choice sets;
- empty/unanswered rows;
- associated text rules;
- same-layout alignment and drift;
- Paddle/geometry disagreement without answer mutation;
- audit persistence of exact image/prompt/raw/normalized response.

### Real visual regression tests

- Q3 yields three logical questions and selects `Nein` in the primary question;
- Q8 yields five choices and selects visual choice 3;
- Q10 yields six shared choices and all 16 matrix rows;
- Q11 yields four gender choices despite missing Paddle control glyphs;
- Q12 yields seven age choices and selects position 2;
- Q15 yields `Ja/Nein` and selects `Ja`;
- Q6/Q14 do not enter the mark-only extractor.

Synthetic tests validate mechanics but do not replace labelled scans.

## 14. Audit and privacy requirements

Development audit mode should continue saving, for every call:

- exact complete-section `input.png`;
- exact `prompt.txt`;
- exact `response_schema.json`;
- request/provider metadata;
- complete raw response envelope;
- raw content before parsing;
- parsed response or error;
- normalized TextLab response with generated IDs;
- Paddle/geometry/template validation warnings.

Audit mode remains disabled by default because these files contain persistent,
potentially sensitive document images and respondent data.

## 15. Questions for Claude's review

1. Is the proposed `questions -> shared choices -> rows -> marked_answers`
   structure sufficiently universal, or is there a real form pattern it cannot
   represent without special cases?
2. Should every printed choice be returned once as proposed, or should the VLM
   return only marked labels/positions to reduce output further?
3. Is splitting Q3-style content into separate question objects preferable to
   representing its subquestions as rows with different choice sets?
4. Is `associated_text` appropriate in this mark-only schema, or should it be a
   separate enrichment even for a selected `Other` field?
5. Is position-first same-layout alignment safe enough, and what drift checks
   should be mandatory before reusing frozen labels/IDs?
6. Should the first same-layout document define the template, or should a
   separate template-building pass/document be required?
7. Would a two-stage VLM approach—structure first, marks second—materially
   improve reliability enough to justify doubling calls?
8. Should the response schema include bounding boxes for questions/rows/choices,
   or are ordered positions plus the retained full crop sufficient for v1?
9. What held-out false-selection and exact-section thresholds should block
   release?
10. Does Qwen's observed strength justify making it the initial benchmark
    favourite, while still keeping provider selection evidence-based?

## 16. Recommended approval

Approve Phases 0 and 1 as an experimental A/B implementation. Do not delete the
current path or select Gemma/Qwen as production authority until the universal
contract has been run on the same labelled complete-section set.

The recommended direction is nevertheless clear: Paddle should find and bound
the relevant form section, while the response VLM should read the complete
visible structure without being constrained by a frequently incomplete Paddle
semantic schema.
