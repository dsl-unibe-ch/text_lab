# Knowledge Graph

Explore connections between scientific papers, authors, research topics and
citations. Create a project with one build action, or open saved results to
continue exploring in a later session.

## Create a project

1. Choose **Create a project**.
2. Under **Add papers**, choose **Upload PDF or ZIP** (the default) or
   **Use a folder on this cluster**.
3. For uploads, select one PDF or a ZIP containing PDF papers and enter a
   **Project name**. For a cluster folder, enter the folder containing PDFs.
4. Choose **Save project in**: persistent storage on the cluster where you
   want the extracted results to remain. This path refers to the cluster,
   not your computer. The full project folder path is displayed below it.
5. To change how topics are extracted, open **Advanced settings**. Under
   **Topic extraction mode**, choose to run the language model in your Text Lab
   session (Ollama, no API key needed) or use the university's GPUStack service
   (API key required). The model reads paper titles and abstracts to identify
   research topics. With GPUStack, these are sent to the university service;
   with Ollama, they stay in your session. Session model choices follow the
   same GPU-based configuration as the other Text Lab features; the first
   configured model is selected by default.
6. Click **Build knowledge graph**.

Text Lab extracts text, metadata and citations using Grobid, extracts research
topics from titles and abstracts using the selected model, and opens the graph.
Progress is shown while processing runs. The saved folder is
`<project name>_project_corpus` for uploads or `<PDF folder name>_project_corpus`
for cluster folders, under your chosen save location.

ZIPs may contain subfolders; only PDF papers are processed. Filenames are
preserved, including in selectors such as `P0001 - Paper.pdf`. Each paper in a
ZIP must have a unique filename, even across subfolders. Rename duplicates
before uploading. Invalid archives and files without a PDF header show an error.

Uploaded PDFs are staged in the private job workspace and removed after the
build attempt, including on failures. Extracted text, metadata, topics and
graphs remain in your chosen project folder. To retry failed PDFs later, upload
the original PDF or ZIP again using the same project name and save location.
You do not need the original upload to open saved extractions or graphs.

PDF extractions are saved as processing progresses. Papers already extracted
are recognised by PDF filename and reused; failed papers are retried. Changing
a PDF without changing its filename does not trigger extraction again. Topic
extraction processes all papers and replaces the saved topic table only when
it completes. Papers without abstracts are included without research topics.

If processing fails, the saved extractions remain available for another attempt.

## Open an existing project

Choose **Open an existing project** and enter either the project folder or a
folder containing your saved projects. Select a project and click **Open
project**. The source PDFs, Grobid server and topic model service are not needed
to explore completed results.

Once a project is open, its save location is shown above the results. Use
**Choose another project** to switch collections.

## Explore and download

The project has three views:

- **Explore:** the whole collection graph, or the connections around one paper.
  The graph updates automatically when you change the view, selected paper,
  or graph options. Under **Advanced graph options**,
  choose node types, filter papers, or set a minimum topic confidence for the
  whole collection. Use **Recenter** inside the graph to fit all nodes back
  into view without rebuilding it.
- **Papers:** inspect extracted metadata and download the saved paper table as
  CSV when available.
- **Topics:** inspect saved research topics and download papers with topics as
  JSON.

The graph legend explains node colours and shapes. Hover over nodes for more
information and use the graph's navigation controls to explore connections.
Download the displayed graph as a standalone HTML file to open offline or share.
The graph remains visible when other widgets cause the page to refresh.

## Update or repair a project

Open **Project tools** and enable **Show processing controls** when needed:

- **Update graph:** supply the PDF folder and select a topic model. New and
  previously failed PDFs are processed, successful extractions are reused,
  and topics are generated for the collection with the selected model.
- **Re-extract topics:** choose a different model without processing PDFs again.
  This also completes a project whose paper extraction succeeded but whose
  topic extraction failed.
- **Rebuild paper table:** regenerate metadata tables from saved paper
  extractions. This does not change the saved topics. If the set of papers has
  changed, re-extract topics to bring the graph up to date.

If topic extraction fails during an update, the last completed topics remain
available. The page warns when those topics cover a different set of papers
from the current paper table.

## Data privacy and storage

- For cluster folder input, original PDFs stay where they are. Uploaded PDFs
  and ZIP contents are staged temporarily and are not copied permanently.
- Extracted metadata, text, topics and generated graph HTML files stay in the
  chosen project folder until you delete them. Choose persistent cluster
  storage if you want to return in another session.
- Grobid's temporary working copies use the session's private workspace and
  are deleted when the session ends.
- With the session model, titles and abstracts are processed in your own job.
  With GPUStack, they are sent to the University of Bern's service, staying on
  university infrastructure.
