# Topic Modeling

Topic extraction from text collections with BERTopic, LDA and Top2Vec,
including evaluation.

**Status:** moved, not refactored yet. The modules were moved here from
`src/core` with only import and path updates; the package docstring in
`__init__.py` says what each one does.

- Modules: `bertopic_engine.py`, `evaluation.py`, `lda_engine.py`, `small_corpus.py`, `top2vec_engine.py`, `topic_config.py`, `topic_pipeline.py`, `topic_utils.py`
- UI: `Topic_Modeling.py` in `src/textlab/ui/streamlit/pages/`
- Tests: `tests/test_topic_modeling.py`

Once the feature is refactored, this file documents its pipeline, public API,
files written to disk, configuration and tests, as described in the developer
guide ([architecture](../../../../docs/dev/architecture.md#feature-packages)).
