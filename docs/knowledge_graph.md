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
   not your computer. By default, projects are saved beside the selected
   papers folder, or in your home directory for uploads. You can change
   either location. The full project folder path is displayed below it.
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
a PDF without changing its filename does not trigger PDF extraction again.
When you build or update a graph, topic extraction runs again for every paper
with an abstract, including reused papers. The saved topic table is replaced
only when extraction completes. Papers without abstracts are included without
research topics.

If processing fails, the saved extractions remain available for another attempt.

## Open an existing project

Choose **Open an existing project** and enter either the project folder or a
folder containing your saved projects. Select a project and click **Open
project**. The source PDFs, Grobid server and topic model service are not needed
to explore completed results.

Once a project is open, its save location is shown above the results. Use
**Choose another project** to switch collections.

## How the graph is built

| Stage | Method | What it gives you |
|---|---|---|
| Read the PDFs | Grobid extracts metadata, text and references from each PDF. | Paper titles, authors, abstracts and citations, when available. |
| Extract topics | The selected language model reads each paper's title and abstract. It is asked for 3–8 specific topics, each with a broader category and a confidence score. | Research themes for individual papers. Papers without an abstract skip this stage. |
| Build and display the network | NetworkX builds a directed graph from the extracted data. Pyvis uses a Barnes–Hut force layout to spread out nodes and draw connected nodes together. | An interactive view of authorship, topics and citations. |

Topic extraction works **paper by paper**. To discover groups of documents
across a corpus using BERTopic, Top2Vec or LDA, use
[Topic Modeling](topic_modeling.md). The distances between nodes in the
Knowledge Graph are a result of the layout; they do not measure similarity.

## Understanding the results

The saved project has three tabs: **Explore**, **Papers**, and **Topics**.
Use them together: find a connection in the graph, identify the papers behind
it, and check the topics the model assigned.

### Explore: the interactive graph

Use **Whole collection** to see links across your papers, or **Single paper**
to focus on one paper and its connections. In **Advanced graph options**, you
can choose node types to show, select papers for the whole collection view,
and set a minimum topic confidence. The graph updates as you change these
controls.

Hover over nodes for details. Drag the background to pan, zoom in or out
within the graph, or use **Recenter** to fit all nodes back into view. The
graph stays visible when unrelated controls cause the page to refresh.

In **Whole collection**, click any node to highlight the nodes one connection
away and the links to them. For example, click a topic to see which papers
are assigned to it. Other nodes fade. Click empty space to clear the highlight.

Below the graph, **Nodes** counts all displayed entities, **Connections**
counts the links between them, and **Papers** counts the blue paper nodes.
Separate citation nodes do not count towards **Papers**. These numbers
describe the current view, so they can change when you apply filters.

#### Colours, shapes, and connections

Each shape is a **node**. The same colour can mean different things when paired
with different shapes, so look at both.

| Node | Colour and shape | What it means |
|---|---|---|
| Paper in your collection | Blue circle | A paper you added. The label is its project ID, such as `P0001`. Hover to see its title and DOI, if available. |
| Author | Green triangle | An author of a paper in your collection. A shared author can connect several papers. |
| Category | Purple rectangle | A broad research area, such as *Computer Vision*. |
| Topic | Red ellipse | A specific theme within a category, such as *Object Detection Networks*. Topics are the more specific level; categories are the broader level. |
| Cited paper in your collection | Purple circle in **Single paper** | A reference matched to a paper in your collection by title. In **Whole collection**, the citation connects to the paper's existing blue node. |
| External cited paper | Orange circle | A reference that was not matched to a paper in the collection shown. |
| Cited author | Gold triangle | An author whose work is cited by a paper in the graph. A citation does not imply co-authorship. |

Read the labels on the connections to see why two nodes are linked:

- `authored` or `authored_by`: an author wrote the paper.
- `has_topic`: the model assigned this topic to the paper.
- `belongs_to`: the specific topic belongs to the broader category.
- `in_category`: the paper has a topic in that category.
- `cites`: the paper references another paper.
- `cites_work_by`: the paper references work by that author.

For example, if `P0001` and `P0002` both connect to the red topic *Object
Detection Networks*, the model assigned the same topic label to both papers.
If they only share the purple category *Computer Vision*, they share a broader
research area and may have different specific topics. Neither connection means
that one paper cites the other; look for a `cites` connection for that.

In **Single paper**, a larger topic node indicates higher model confidence.
In **Whole collection**, topics grow with the number of times they appear;
hover over a topic to see its average confidence.

### Papers: extracted metadata

The **Papers** tab lists each paper's project ID, title, authors, publication
date, and PDF filename. Use the ID to match a graph node such as `P0001` to
its source file. If you need the extracted abstract, DOI, or references,
download the paper table as CSV.

### Topics: labels assigned to each paper

The **Topics** tab lists one row per topic assigned to a paper. The same
topic can appear on multiple rows if it was assigned to multiple papers.

| Field | How to read it |
|---|---|
| **Paper** | The title of the paper the model read. |
| **Label** | The specific topic assigned to the paper, shown as a red ellipse in the graph. |
| **Category** | The broader research area containing the topic, shown as a purple rectangle. |
| **Confidence** | The model's estimate of the match on a scale from 0 to 1. Use it to filter suggestions, and check the abstract before relying on a label. It is not a validated accuracy score. |
| **Rationale** | A short explanation generated by the model for why it assigned the topic. Check it against the paper. |

## Downloading the results

| Download | Where | What you get |
|---|---|---|
| **Download graph (HTML)** | **Explore** | The current graph view as an interactive HTML file you can open offline. |
| **Download papers (CSV)** | **Papers** | The extracted paper table, including metadata, abstracts, and references when available. |
| **Download papers and topics (JSON)** | **Topics** | The paper records with their saved topic labels, categories, confidence scores, and rationales. |

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

## Troubleshooting

Click a problem to see what to do.

??? question "A paper has no topics"
    Under **Papers**, use **Download papers (CSV)** and check the
    `abstract` column. Text Lab skips topic extraction if there is no
    abstract. Also check that **Topics** is enabled in **Advanced graph
    options**. In **Whole collection**, try lowering **Minimum topic
    confidence** to see whether topics were filtered out.

??? question "I expected to see a citation between two papers"
    Turn on **Cited papers** in **Advanced graph options**. If the connection
    is still missing, download the paper table from **Papers** and check
    the citing paper's `cited_titles` column. Grobid may have missed the
    reference, or its extracted title may differ from the title in your
    collection.

??? question "The graph is too crowded or I lost track of a node"
    Use **Recenter** to fit the nodes back into view. Switch to **Single paper**
    to focus on one paper, or use **Advanced graph options** to show fewer
    papers or hide node types you do not need.

??? question "Related papers have different topic labels"
    Compare their broader categories and read the titles and abstracts to
    confirm the relationship. Text Lab connects matching topic labels; the
    model can describe similar ideas with different words. Use **Topics** to
    compare the saved labels. If needed, try another model with **Re-extract
    topics**.

??? question "I changed a PDF, but the graph still shows old information"
    Check the filename. Text Lab reuses successful PDF extractions by name.
    To compare the updated paper with the original, give the new PDF a
    different filename. To build a collection with only the updated version,
    create a new project with the revised PDF.

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
