# Knowledge Graph

Interactive graphs of a collection of scientific papers: Grobid parses the
PDFs into metadata, references and text, an LLM extracts research topics
from each abstract, and the papers are drawn with their authors, topics and
citations. Used by the Knowledge Graph page.

## Pipeline

The user works through three steps; each reads and writes a corpus folder
in a location the user chooses, so a later session can resume at any step.

1. **Corpus** (`corpus.build_corpus`): every PDF in the papers folder that
   is not in the corpus yet (matched by file name) is sent to the Grobid
   server (`grobid`), and its TEI XML is read (`tei`) into
   `metadata.json` and plain text. A paper Grobid cannot parse gets
   `error.txt` and is tried again next time. Then the corpus tables are
   rebuilt (`corpus.build_corpus_table`, also offered alone as step 1b).
2. **Topics** (`topics.extract_topics`): the title and abstract of each
   paper go to a model, through an OpenAI-compatible API: the session's
   Ollama server, or GPUStack with the user's API key. The answer is
   cleaned into 3 to 8 topics, each a specific `label` in a broad
   `category`, with a confidence. Papers without an abstract get none.
3. **Graphs** (`graphs`): the ego graph of one paper (its authors, topics,
   cited papers and cited authors) or the graph of all papers, where shared
   authors, topics and categories connect papers. Graphs are NetworkX
   graphs drawn by Pyvis as a self-contained HTML page (vis.js embedded).

Grobid runs as its own Apptainer container, started once per session
(detached, so page reloads find it running); the models run in Ollama or on
GPUStack. The steps therefore run in the app process, without a worker.
NetworkX, Pyvis and the OpenAI client are imported only when used.

## Public API

Interfaces use `service.py`:

| Name | Purpose |
|---|---|
| `ensure_grobid_server()`, `GrobidError` | Start Grobid, or say why it cannot run |
| `list_pdfs(folder)`, `corpus_folder_name(folder)`, `find_corpora(location, required=)` | Papers and corpus folders |
| `build_corpus(pdfs, corpus, on_progress=)` | Step 1; returns a `CorpusSummary` |
| `build_corpus_table(corpus)` | Step 1b; the corpus table as a DataFrame |
| `ollama_models()`, `ollama_client()`, `gpustack_client(api_key)`, `GPUSTACK_MODEL` | Model choice |
| `extract_topics(corpus, client, model, on_progress=)` | Step 2; returns a `TopicSummary` |
| `read_records(path)`, `topics_json(corpus)` | The table with topics, as records or a JSON download |
| `build_paper_ego_graph(record, all_records=, ...)`, `build_full_corpus_graph(records, ...)` | Step 3; `(graph, network)` |
| `save_graph_html(network, path)`, `count_nodes(graph, node_type)` | Graph page and statistics |

```python
from pathlib import Path

from textlab.features.knowledge_graph import service

papers = Path("~/papers").expanduser()
corpus = papers.parent / service.corpus_folder_name(papers)
summary = service.build_corpus(service.list_pdfs(papers), corpus)
service.extract_topics(corpus, service.ollama_client(), "qwen3:8b")
records = service.read_records(corpus / service.TOPICS_JSONL)
graph, network = service.build_full_corpus_graph(records)
service.save_graph_html(network, corpus / "full_corpus_graph.html")
```

## Layout

```
knowledge_graph/
├── service.py  # the API above
├── models.py   # CorpusSummary, TopicSummary
├── grobid.py   # the Grobid server
├── tei.py      # metadata, references and text from TEI XML
├── corpus.py   # the corpus folder and its tables
├── topics.py   # topics from an LLM
├── graphs.py   # ego and full corpus graphs
├── cli.py      # batch command (planned)
└── tests/
```

The topic prompt in `topics.py` is model input and is kept exactly as
written.

## Files written

| What | Where | Removed |
|---|---|---|
| The corpus: per paper `P0001/` etc. with `fulltext.tei.xml`, `fulltext.txt`, `metadata.json` or `error.txt`; `corpus_metadata.json`, `corpus_table.csv`, `corpus_table.jsonl`, `corpus_table.with_topics.jsonl` | `<papers folder>_project_corpus` in the location the user chooses | When the user deletes it |
| Graph pages (`<paper>_ego_graph.html`, `full_corpus_graph.html`) | The corpus folder | When the user deletes them |
| PDFs Grobid is processing | `grobid` area of the job workspace (Grobid's tmp folder) | When the session ends |

The PDFs are read where they are and not copied.

## Configuration

| Setting | Used for |
|---|---|
| `TEXT_LAB_GROBID_CONTAINER` | The Grobid Apptainer image; without it the page explains that the feature is unavailable |
| `TEXT_LAB_GPUSTACK_URL` | GPUStack's OpenAI-compatible endpoint |
| `GROBID_PORT` | Grobid's port (8070; the admin port is the next one) |
| `OLLAMA_HOST` | The session's Ollama server |

## Tests

In `tests/`, without Grobid, models or GPU:

- `test_tei.py`: metadata, references and plain text from sample TEI.
- `test_corpus.py`: building a corpus with a fake Grobid: skipping known
  papers, numbering new ones, retrying failures, the tables.
- `test_topics.py`: cleaning answers, retries, topic extraction with a fake
  client, the clients' endpoints.
- `test_graphs.py`: ego and full graphs of sample records, the HTML page.
- `test_grobid.py`: starting the server and sending PDFs, with fakes.
- `test_knowledge_graph_page.py`: the page imports only the service, and
  the service loads neither NetworkX, Pyvis nor the OpenAI client.

## Batch use (planned)

`cli.py` sketches a `textlab knowledge-graph` command that builds a corpus
(and optionally its topics) in a batch job; the page then draws its graphs.
