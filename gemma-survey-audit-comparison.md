# Gemma survey audit comparison

Date: 2026-07-21  
Input: `split_1_2.pdf`  
Model: local Ollama `gemma4:26b`

## What reaches Gemma

Gemma receives one unmodified, complete Paddle-bounded question section cut
from the original 300-DPI page plus a text schema containing Paddle-derived
question, row, option labels, and stable IDs. The geometric/stroke detector
does not alter the pixels and its state is not included in the prompt. Geometry
can only:

1. help propose a candidate section when Paddle omitted mark glyphs and at
   least two aligned circle/box shapes remain; and
2. force the merged result to `needs_review` after Gemma responds.

## Audits

| Contract | Calls | Parseable | Errors | Wall time |
|---|---:|---:|---:|---:|
| Original rich group/row response | 43 | 42 | 1 | 223.99 s |
| Transitional ID-only response | 43 | 43 | 0 | 176.69 s |
| Canonical ID-only response for every call | 43 | 43 | 0 | 175.41 s |
| Canonical ID-only, complete question sections | 13 | 13 | 0 | not recorded |

The original contract asked Gemma to repeat question text, row labels, question
type, selection rule, and nested marks. Those fields varied between tiles even
when the detected option ID was consistent.

The revised contract allows only:

```json
{
  "visible_row_ids": ["stable_row_id"],
  "marks": [
    {
      "option_id": "stable_option_id",
      "state": "selected",
      "visual_mark": "x"
    }
  ],
  "unmapped_marks": []
}
```

TextLab now owns all question/row structure and labels. The canonical tiled
audit used this one top-level contract for all 43 calls, including candidates
for which
Paddle supplied no usable option schema. None of the raw answers contains the
old `groups` structure. Gemma still varies insignificant JSON whitespace and
can put an unknown human label in `marks.option_id`; the normalizer treats that
as an unmapped result and forces review rather than accepting invented form
structure.

This eliminated the malformed response, reduced runtime by about 22%, and
removed obvious over-selection on Q5 and Q7 in this run. It did not eliminate
visual errors: Q8 still treated several printed/unclear marks as ambiguous, and
Q13 returned two selected locations. These remain review cases. The false Q3
prize-list candidate also shows that candidate proposal and visual accuracy
need to be evaluated separately from output standardization.

The later complete-section audit removed independent visual snippets and
tightened candidate proposal. It reduced the run from 43 calls to 13; summed
Ollama response time fell from 95.2 to 50.4 seconds. Q2 no longer produced an
unmapped answer from its isolated right-hand handwriting, Q8 returned one
selection, and Q13 returned one location. The numbered prize list is no longer
misidentified as Q3; its actual `Ja`/`Nein` response is correctly bounded as
Q15. Q6 and Q14 are open/numeric or free-text questions and are not submitted
to this mark-only extractor.

## Review files

- [Original rich-contract contact sheet](gemma_survey_audit/REVIEW.md)
- [Transitional ID-only contact sheet](gemma_survey_audit_id_only/REVIEW.md)
- [Final canonical contact sheet](gemma_survey_audit_canonical/REVIEW.md)
- [Complete-section contact sheet](gemma_survey_audit_complete_sections_v2/REVIEW.md)
- [Original final response CSV](gemma_survey_audit/form_responses.csv)
- [Transitional ID-only final response CSV](gemma_survey_audit_id_only/form_responses.csv)
- [Final canonical response CSV](gemma_survey_audit_canonical/form_responses.csv)
- [Complete-section response CSV](gemma_survey_audit_complete_sections_v2/form_responses.csv)

Every call directory contains the exact `input.png`, `prompt.txt`,
`response_schema.json`, `raw_response.json`, `raw_content.txt`, and either a
`parsed_response.json` or `error.txt`.
