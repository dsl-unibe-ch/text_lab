# Topic Modeling

Finds the topics of a collection of texts (a table with a text column, or a
ZIP of text files) with BERTopic, Top2Vec or LDA, evaluates them, and
returns a topic table, interactive charts and a download with every
document's topic. Used by the Topic Modeling page (Streamlit's
`Topic_Modeling.py`, and `ui/web/pages/topic_modeling.py` in the new
interface, which draws the run's overview instead of the charts).

## Pipeline

`service.analyze_table` runs, for one collection:

1. **Prepare** (`data`): drop rows without text; for BERTopic with topics
   over time, parse the timestamp column (dates, date-times or whole years)
   and drop rows whose timestamp cannot be read.
2. **Embed** (BERTopic only, `embeddings`): load the sentence-transformer
   (the shared MiniLM models, or a model the user names, downloaded to their
   own Hugging Face cache) and embed every document once. Documents longer
   than the model's context window are truncated, or embedded in chunks
   whose embeddings are averaged. A notice says how many were affected.
3. **Train** (`pipeline` and the engine modules): BERTopic
   (`bertopic_engine`: UMAP/PCA/SVD, HDBSCAN/KMeans, c-TF-IDF keywords),
   Top2Vec (`top2vec_engine`) or LDA (`lda_engine`, on spaCy lemmas from
   `text`). Each returns the topic table, every document's topic, keyword
   lists and HTML charts.
4. **Evaluate** (`evaluation`): topic diversity, coherence (C_v, C_npmi,
   U_mass), LDA perplexity and, if asked, stability: the model is trained
   twice more with unlocked seeds (reusing the embeddings) and the topics
   compared.
5. **Package** (`reports`): a ZIP with the run's settings and metrics,
   `document_topics.csv`, `topic_keywords.csv` and the charts.
6. **Overview** (`overview`): what an interface draws. Every run gets the
   parts all algorithms share, built the same way from what each engine
   gives (the topics with their keyword weights, every document's main
   topic, one vector per topic): each topic's size and most typical
   example documents, the cosine similarity of every two topics, a map
   where alike topics lie close (classical multidimensional scaling), a
   dendrogram of how they merge (SciPy, average linkage) and, with topics
   over time, each topic's documents per interval. Each algorithm adds
   what its own library draws:
    - **LDA**: pyLDAvis's view, from the data `lda_engine.prepare_ldavis`
      prepares for the dashboard anyway (topics in model order): each
      topic's place and share of the words, the terms it may rank by
      relevance (term, frequency in the topic, frequency, log probability,
      log lift), the most salient terms, and each term's spread over the
      topics (`overview.ldavis_parts`); and each example's mixture of
      topics (`lda_engine.lda_document_topics`).
    - **BERTopic**: the documents placed by meaning (UMAP of their
      embeddings, BERTopic's settings for `visualize_documents`, a random
      sample of 1,500 for large collections), and its topics over time,
      computed once (`bertopic_engine.topics_over_time_frame`) for both
      the Plotly chart and each interval's words.
    - **Top2Vec**: the documents placed by meaning (its document vectors),
      and its word clouds (the 30 words closest to each topic).

   If the overview cannot be built, the run keeps its other results and
   says so in a notice.

A collection too small to cluster fails deep inside UMAP or HDBSCAN;
`small_corpus` recognizes those failures and replaces them with a message
that suggests LDA.

The page runs this in a worker process (`service.run_topic_modeling`,
`worker.py`), so the modeling libraries (about half a minute to import) are
never loaded by the app and all GPU memory is released after every run.

## Public API

Interfaces use `service.py`:

| Name | Purpose |
|---|---|
| `load_table(filename, data)`, `load_archive(data)` | Read an upload into a table; `ValueError` if it is empty or unsupported |
| `summarize_table(table)` | A `TableSummary`: size, columns, the likeliest text column, the first rows, for choosing columns |
| `TopicModelingConfig` | The run's settings: algorithm, language, text column, algorithm options, `evaluate_stability` |
| `run_topic_modeling(table, config, source_name=, on_progress=, cancel=)` | The analysis in a worker; returns a `TopicModelingResult` |
| `analyze_table(table, config, source_name=, on_progress=)` | The same analysis in this process, for the worker and batch jobs |
| `TopicModelingResult` | Topic table, charts, metrics, the ZIP, `notices` to show users, and the `overview` (`TopicOverview`) to draw |
| `save_result(result, source_name, folder)` | Writes the ZIP and, from it, the document table, the topic table and the run report into a folder; returns a `SavedTopics` (what to show, and `SavedFile`s) |
| `Algorithm`, `SUPPORTED_LANGUAGES`, `TABLE_EXTENSIONS`, ... | Choices for the page, re-exported |

A failure users can act on (too few documents, an unreadable timestamp
column, an embedding model that cannot be loaded) is raised as `ValueError`
with a message for users, also from the worker; any other failure is a
`WorkerError` whose `details` hold the worker's traceback.

```python
from textlab.features.topic_modeling import service

table = service.load_table("responses.csv", data)
config = service.TopicModelingConfig(
    algorithm=service.Algorithm.LDA,
    language="English",
    text_column="Answer",
    num_topics=8,
)
result = service.run_topic_modeling(table, config, source_name="responses.csv")
print(result.topic_df)
```

## Layout

```
topic_modeling/
├── service.py          # the API above
├── worker.py           # python -m textlab.features.topic_modeling.worker
├── models.py           # settings, results, choices
├── data.py             # reading uploads, timestamps
├── embeddings.py       # sentence-transformers, chunked embedding
├── text.py             # stopwords and tokenizing (spaCy, NLTK)
├── pipeline.py         # runs the chosen algorithm; stability runs
├── bertopic_engine.py  # BERTopic training, topics, charts
├── top2vec_engine.py   # Top2Vec training, topics, chart
├── lda_engine.py       # LDA with gensim, pyLDAvis chart
├── evaluation.py       # diversity, coherence, perplexity, stability
├── overview.py         # what each algorithm's view draws
├── small_corpus.py     # recognizing a collection too small to cluster
├── reports.py          # topic table, run report, result ZIP
├── cli.py              # batch command (placeholder)
└── tests/
```

The page imports only `service.py`, which imports only pandas; a test
checks that importing it loads none of the modeling libraries.

## Files written

| What | Where | Removed |
|---|---|---|
| The collection, pickled for the worker, and the result ZIP | `topic_modeling/run-*` in the job workspace | When the run ends |
| The worker's request, progress and result | `topic_modeling/job-*` in the job workspace | When the run ends |
| Custom embedding models the user names | `~/.cache/huggingface/hub` | By the user; model files, not user data |

## Configuration

No site settings. The shared embedding models are read from `HF_HOME`, the
model store the launch script mounts; spaCy models are in the image, and
NLTK stopwords are read from `NLTK_DATA` when it is set.

## Tests

In `tests/`, without GPU or models:

- `test_topic_modeling.py`: small-corpus errors, timestamps, CSV and ZIP
  reading, text splitting, the topic table, stability and metrics, chunked
  embeddings with a fake model.
- `test_service.py`: configurations and results (with the overview)
  across the worker boundary, loading and describing tables, the analysis
  with fake modeling modules (notices, progress, embeddings reused for
  stability, the ZIP, the overview, a broken overview), the worker
  hand-over and error messages, `save_result`.
- `test_overview.py`: sizes, outliers and examples, scaled weights,
  similarity and the map (alike topics close), sparse vectors, the
  timeline per year and per interval; pyLDAvis's data and the mixtures for
  LDA, the words per interval for BERTopic, the word clouds for Top2Vec,
  the document map and its sample, the hierarchy.
- `test_topic_modeling_page.py`: the page imports only the service, and the
  service loads no modeling library.

## Batch use (planned)

`cli.py` is a placeholder for a `textlab topics` command for Slurm batch
jobs. It will call `service.analyze_table` with a progress callback that
writes log lines.
