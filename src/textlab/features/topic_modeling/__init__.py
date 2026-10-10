"""Backend for the Topic Modeling feature: topics of a text collection.

Interfaces use :mod:`.service`, which runs the analysis in a worker process
(:mod:`.worker`). The modules behind it:

- ``models``: settings, results and choices.
- ``data``: reading uploads and timestamps.
- ``embeddings``: sentence-transformer embeddings for BERTopic.
- ``text``: stopwords and tokenizing.
- ``pipeline``: runs the chosen algorithm (``bertopic_engine``,
  ``top2vec_engine``, ``lda_engine``) and the stability runs.
- ``evaluation``: diversity, coherence, perplexity and stability.
- ``small_corpus``: recognizing a collection too small to cluster.
- ``reports``: the topic table, the run report and the result ZIP.

See ``README.md`` in this folder for the pipeline and the files written.
"""
