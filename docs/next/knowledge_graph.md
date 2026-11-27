# Knowledge Graph

!!! note "Draft for the new interface"
    This guide describes the Knowledge Graph page of the new Text Lab
    interface. It replaces [the current guide](../knowledge_graph.md) when
    the new interface goes live.

Knowledge Graph maps a collection of research papers: their authors, the
research topics a language model finds in them, and the papers they cite.
You build a **project** once; it is saved in a folder on the cluster, so
you can explore it again in a later session.

Everything runs on the University's computing cluster. With the session's
own model, your papers never leave your session.

## Quick start

1. Under **New project**, **drop your papers**: one PDF, or a ZIP of PDFs.
   Or choose **A folder on the cluster** and give the folder of PDFs.
2. **Name the project** and choose where to **save** it (both optional for
   a folder of papers).
3. **Choose the model** that finds the topics, or keep the first.
4. Press **Build the knowledge graph**.

The build runs in the background; you can leave the page and come back.
When it is done, press **Open the project** on its record.

To come back to a project later, choose **Open a project**, give the
folder it is saved in, press **Find projects**, then **Open**.

## A new project

### Papers

- **Upload**: one PDF, or a ZIP of PDFs. ZIPs may hold subfolders; only
  the PDFs are read. Each PDF needs a file name of its own, also across
  subfolders: rename duplicates before uploading. Uploaded files are kept
  only while the build runs.
- **A folder on the cluster**: give its full path, such as
  `/storage/research/.../papers` (`~` stands for your home folder). The
  page counts its PDFs as you type. Only the PDFs directly in the folder
  are read, not those in its subfolders. They are read where they are;
  nothing is copied.

Papers must be PDFs with their text, not scans: convert scans with
[OCR](ocr.md) first.

### Project

- **Name**: the project folder is called `<name>_project_corpus`. For a
  folder of papers you can leave it empty: the project is then named
  after the folder.
- **Save in**: a folder on the cluster that stays, such as your research
  storage. Left empty, the project is saved next to the folder of papers,
  or in your home folder for an upload. If a project of that name is there
  already, it is updated: its papers are kept, and new ones added.

### Topics

A language model reads each paper's title and abstract and names 3 to 8
research topics, each within a broader category (such as *Object Detection
Networks* within *Computer Vision*), with a confidence from 0 to 1.

- **In this session** (the default): a model running on your session's
  GPU. The models offered depend on the GPU; the first suits most
  collections. The titles and abstracts stay in your session.
- **University GPUStack**: the University's GPUStack service, with your
  API key. The titles and abstracts go to that service, and stay on
  University infrastructure. The key is used for this build only, and not
  kept. Where the site has not set up the service, this choice is
  greyed out.

### The build

1. **Grobid reads the PDFs**: their titles, authors, abstracts and
   references. Papers read before are recognised by their file name and
   reused; papers Grobid could not read are tried again.
2. **The model finds the topics** of every paper with an abstract.
   Papers without one are kept, without topics.

The record of the build shows its progress paper by paper, and when it is
done, how many papers were read, reused, or could not be read. **Cancel**
stops it after the current paper; the papers read until then are kept in
the project and reused next time.

If the topics cannot be found (the model fails, say), the papers read are
saved all the same: open the project and extract the topics again under
**Project tools**.

## A project's page

The page shows the project's name, the folder it is saved in, and how
many papers it has and how many have topics, with three tabs: **Graph**,
**Papers** and **Topics**. Use them together: find a connection in the
graph, identify the papers behind it, and check the topics the model
assigned.

### Graph

Choose what the graph shows above it; it is drawn again at each change:

- **Whole collection** or **One paper**, chosen in the list.
- **Show**: authors, topics, cited papers and cited authors.
- **Topic confidence at least** (whole collection): topics with a lower
  confidence are left out.
- **Papers shown** (whole collection): tick the papers to show; **All**
  and **None** tick or clear them all.

Point at a node for its details, such as a paper's title and DOI. Drag
the background to move the graph, scroll to zoom, and drag a node to place
it; **Fit to view** brings the whole graph back into view. In the whole
collection, click a node to bring out the nodes one connection away and
the links to them, for example a topic and its papers; the rest fades.
Click empty space to see everything again.

Above the graph, the counts say how many **nodes** and **connections** it
has, and how many of the nodes are **papers** of the project (cited papers
outside it do not count). They describe the graph shown, so they change
with the choices above.

**Graph (HTML)** downloads the graph shown as a page that works in any
browser, offline.

#### Shapes, colours and connections

The key under the graph names each kind of node. The same colour can mean
different things with different shapes, so look at both.

| Node | Shape and colour | What it means |
|---|---|---|
| Paper of the project | Blue circle | A paper you added. The label is its project ID, such as `P0001`. |
| Author | Green triangle | An author of a paper of the project. A shared author connects papers. |
| Category | Purple box | A broad research area, such as *Computer Vision*. |
| Topic | Red ellipse | A specific theme within a category, such as *Object Detection Networks*. |
| Cited paper of the project | Purple circle, in **One paper** | A reference matched to a paper of the project by its title. In the whole collection, the citation goes to the paper's own blue node. |
| Cited paper outside the project | Orange circle | A reference not matched to a paper of the project. |
| Cited author | Gold triangle | An author whose work a paper cites. A citation does not mean co-authorship. |

The labels on the connections say why two nodes are linked:

- `authored` or `authored_by`: an author wrote the paper.
- `has_topic`: the model gave the paper this topic.
- `belongs_to` (dashed): the topic belongs to the broader category.
- `in_category`: the paper has a topic in that category.
- `cites`: the paper cites another paper.
- `cites_work_by` (dashed): the paper cites work by that author.

For example, if `P0001` and `P0002` both connect to the red topic *Object
Detection Networks*, the model gave both the same topic. If they share
only the purple category *Computer Vision*, they share a broader research
area and may have different topics. Neither means that one paper cites the
other; look for a `cites` connection for that.

In **One paper**, a larger topic has a higher confidence. In the whole
collection, topics grow with the number of papers that have them; point at
one to see its average confidence. The distances between nodes come from
the layout; they do not measure similarity.

### Papers

One row per paper: its project ID, title, authors, publication date and
PDF. Use the ID to match a node such as `P0001` to its file. **Paper table
(CSV)** downloads the table with the abstracts, DOIs and references.

### Topics

One row per topic of a paper: the paper, the **topic**, its **category**,
the model's **confidence** (from 0 to 1; not a validated accuracy score)
and **why** the model chose it. Check important topics against the
abstract. **Papers with topics (JSON)** downloads the papers with their
topics.

## Project tools

At the foot of the project's page:

- **Add papers**: upload papers or give a folder, and choose the model.
  New PDFs are read, those read before are reused, and those that failed
  are tried again; then every paper's topics are found again.
- **Extract the topics again**: with another model, say, or after topic
  extraction failed. The PDFs are not read again. The topics before stay
  until every paper has new ones.
- **Rebuild the paper table**: from the papers read, as saved in their
  folders, if the table is missing or was changed. The topics stay as they
  are.

These run as jobs of the project, listed under **Updates**; **Show the new
graph** on a finished one shows the project with its new papers or topics.
If papers were added but the topics not extracted again, the page says so:
the graph then shows the topics extracted last.

## Where things are kept

- **The project folder** (`<name>_project_corpus`, where you chose) holds
  what Grobid read from each paper (a folder per paper, `P0001` and so
  on), the paper tables, and the topics. It stays until you delete it.
- **Uploaded PDFs** are kept in your session's private workspace while the
  build runs, and deleted when it ends, also when it fails. PDFs in a
  folder on the cluster stay where they are.
- **Grobid's working copies** are kept in your session's private
  workspace, and deleted when the session ends.
- **Your session's list of projects** (under *Your projects*) and the
  records of builds and updates are kept until the session ends; the
  projects themselves stay in their folders.

## How it works

| Step | Method | What it gives you |
|---|---|---|
| Read the PDFs | Grobid extracts metadata, text and references from each PDF. | Titles, authors, abstracts and citations, when available. |
| Find the topics | The chosen language model reads each paper's title and abstract, and is asked for 3 to 8 specific topics, each in a broader category, with a confidence. | Research themes of each paper. Papers without an abstract skip this step. |
| Draw the graph | NetworkX builds a directed graph of what was found; vis-network draws it, spreading the nodes with a force layout that draws connected nodes together. | An interactive view of authors, topics and citations. |

Topics are found **paper by paper**. To discover groups of documents
across a collection with BERTopic, Top2Vec or LDA, use
[Topic Modeling](topic_modeling.md).

## Troubleshooting

??? question "'New papers cannot be read in this session'"
    Grobid, which reads the PDFs, cannot be started in this session. You
    can still open saved projects and extract their topics again. Send the
    details shown to support.

??? question "A paper has no topics"
    Download the paper table under **Papers** and check its `abstract`
    column: papers without an abstract get no topics. Check also that
    **Topics** is ticked above the graph, and in the whole collection,
    lower **Topic confidence at least**.

??? question "I expected a citation between two papers"
    Tick **Cited papers** above the graph. If the connection is still
    missing, download the paper table and check the citing paper's
    `cited_titles` column: Grobid may have missed the reference, or read
    its title differently from the title of the paper in the project.

??? question "The graph is too crowded, or I lost a node"
    Press **Fit to view**. Choose **One paper** to see one paper and its
    connections, or show fewer papers or kinds of node.

??? question "Related papers have different topics"
    The model can describe similar ideas in different words, and only
    identical topics connect papers. Compare their categories and
    abstracts, and look at the topics under **Topics**. Extracting the
    topics again with another model may help.

??? question "I changed a PDF, but the project still shows the old one"
    Papers read once are recognised by their file name and reused. Give
    the changed PDF another name, or build a new project with it.

??? question "'This project is not open in this session'"
    A link to a project works only in the session that opened it. Open the
    project again under **Open a project**.

For help with your project, contact the Data Science Lab (DSL).
