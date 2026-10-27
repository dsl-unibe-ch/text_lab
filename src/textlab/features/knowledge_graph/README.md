# Knowledge Graph

Interactive graphs of a collection of scientific papers: Grobid parses the
PDFs into metadata, references and text, an LLM extracts research topics
from each abstract, and the papers are drawn with their authors, topics and
citations. Used by the Knowledge Graph page.

## Pipeline

The backend has three stages; each reads and writes a corpus folder in a
location the user chooses. The page presents one **Build knowledge graph**
action and opens saved projects directly into an explorer. PDF extractions
are reused by filename; topic extraction processes the collection again
with the chosen model. Optional updates and table repair live in Project
tools. Opening saved results requires neither the source PDFs nor running
Grobid or model services. New projects accept either a PDF/ZIP upload or a
cluster folder. Uploads require a project name and save location; their PDFs
are staged privately, while the corpus remains in the chosen save location.

1. **Corpus** (`corpus.build_corpus`): every PDF in the papers folder that
   is not in the corpus yet (matched by file name) is sent to the Grobid
   server (`grobid`), and its TEI XML is read (`tei`) into
   `metadata.json` and plain text. A paper Grobid cannot parse gets
   `error.txt` and is tried again next time. Then the corpus tables are
   rebuilt (`corpus.build_corpus_table`, also available in Project tools).
2. **Topics** (`topics.extract_topics`): the title and abstract of each
   paper go to a model, through an OpenAI-compatible API: the session's
   Ollama server, or GPUStack with the user's API key. The answer is
   cleaned into 3 to 8 topics, each a specific `label` in a broad
   `category`, with a confidence. Papers without an abstract get none.
3. **Graphs** (`graphs`): the ego graph of one paper (its authors, topics,
   cited papers and cited authors) or the graph of all papers, where shared
   authors, topics and categories connect papers. Graphs are NetworkX
   graphs drawn by Pyvis as a self-contained HTML page (vis.js embedded).

Grobid runs as its own Apptainer container, started on a build or update
(detached, so page reloads find it running); the models run in Ollama or on
GPUStack. The steps therefore run in the app process, without a worker.
NetworkX, Pyvis and the OpenAI client are imported only when used. The page
keeps the selected project and displayed graph in session state. Switching
projects resets the graph; project-specific form keys isolate paper filters.
Session state is temporary; saved corpus files remain the source of truth.

Session model choices use `common.model_config.get_available_models` and
`common.gpu_manager.get_gpu_name`, matching Chat, Visualize Data and Meeting
Notes. `common/models.json` defines the lists for standard and high-memory
GPUs; `models.local.json` can override them. The first configured model is
the initial choice, and an old selection outside the current list is reset.
Installed Ollama models are not used as the source of selectable models.
GPUStack continues to use its separately configured fixed model.

## Public API

Interfaces use `service.py`:

| Name | Purpose |
|---|---|
| `ensure_grobid_server()`, `GrobidError` | Start Grobid, or say why it cannot run |
| `staged_papers(name, data)`, `project_folder_name(name)` | Stage a PDF/ZIP with cleanup; validate an uploaded project name |
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
├── service.py  # the API above, including temporary PDF/ZIP staging
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

PDFs from cluster folders are read where they are and not copied. Uploaded
PDFs and ZIP contents use `knowledge_graph/upload-*` in the private job
workspace, removed after the build attempt, even on errors. ZIP extraction
uses shared path and size safeguards, ignores non-PDF and macOS metadata
files, and rejects duplicate PDF basenames (including case variants), since
corpus reuse is based on filename. PDF bytes must contain a PDF header.
Nested archives are not expanded. Opening the saved corpus never requires
these temporary PDFs.

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
- `test_uploads.py`: PDF/ZIP staging, nested and uppercase filenames, archive
  validation, duplicate basenames, cleanup on success and failure, reuse of
  saved extractions, and project name validation.
- `test_model_selection.py`: shared GPU-based model lists, local overrides,
  and independence from the installed Ollama model list.
- `test_knowledge_graph_page.py`: the page imports only the service, and
  the service loads neither NetworkX, Pyvis nor the OpenAI client.

UI interaction tests live in `tests/test_knowledge_graph_interface.py`,
outside the backend package to preserve its import boundary. They cover
opening saved and partial projects without processing services, persistent
graph views, project switching, and reusing saved papers after a topic model
failure, plus building and reopening uploaded PDF and ZIP projects.

## Batch use (planned)

`cli.py` sketches a `textlab knowledge-graph` command that builds a corpus
(and optionally its topics) in a batch job; the page then draws its graphs.
