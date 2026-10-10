"""Backend for the Topic Modeling feature.

Topic extraction from text collections with BERTopic, LDA and Top2Vec,
including evaluation.

Moved from ``src/core`` with only import and path updates. The modules are
reorganized when the feature is refactored (see ``docs/dev/architecture.md``):

- ``topic_pipeline``: runs the selected algorithm and collects its outputs.
- ``bertopic_engine``: BERTopic training, topics and visualizations.
- ``lda_engine``: LDA with gensim and pyLDAvis.
- ``top2vec_engine``: Top2Vec training and topics.
- ``evaluation``: diversity, coherence, perplexity and stability metrics.
- ``small_corpus``: recognizing errors caused by a corpus that is too small.
- ``topic_config``: configuration and result types.
- ``topic_utils``: data loading, preprocessing and reports.
"""
