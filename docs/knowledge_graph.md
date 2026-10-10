# Knowledge Graph

The Knowledge Graph feature uses large language models (LLMs) to automatically extract key topics from the abstracts of your uploaded scientific papers. It then builds an interactive graph that reveals hidden connections across your literature collection — making it easy to explore themes, spot patterns, and navigate complex research landscapes.

---

## Step 1 — Build Your Corpus

Start by pointing the app to a folder containing your PDF collection. The app will scan the folder and list all detected PDFs so you can confirm you're working with the right files. Once ready, click **"Generate Corpus"** to kick off the extraction process.

Under the hood, this step uses [GROBID](https://grobid.readthedocs.io/en/latest/Introduction/) to parse each PDF and convert it into a structured XML format optimised for downstream processing. The output is saved to a new folder named `your_collection_project_corpus`.

> **Smart caching:** This step is computationally intensive, so the app only parses papers that are not in the corpus yet (it recognises them by file name), and tries again those that could not be parsed before. You can add PDFs to your folder and run the step again: it reports how many papers were processed, skipped and could not be parsed. A paper that is changed but keeps its file name is not parsed again; to parse it again, delete its `P0001`-style folder in the corpus.

Once your corpus is built and nothing has changed, you can skip straight to Step 2 on future runs.

A **CSV export** is also generated at this stage, giving you a human-readable summary of the extraction results. Use it to spot any issues with how GROBID processed individual papers.

> **Lost your CSV?** No problem — use **Step 1b: "Rebuild Corpus Table"** to regenerate it at any time. Just point to the folder containing your corpora; they'll be detected automatically so you can select the one you need.

---

## Step 2 — Extract Topics

With your corpus ready, the app uses an LLM to read each paper's abstract and extract **3 to 8 concise topics**. Each topic belongs to a broader category, which serves as a shared node when connecting papers in the graph.

Point to the folder containing your project corpora — they'll appear automatically in the dropdown. Select the corpus you want to process and let the LLM do the work.

The resulting topics are saved to a JSON file that feeds directly into Step 3. You can also download this file at any time for offline inspection or further analysis.

---

## Step 3 — Visualise Your Data

If you've already completed topic extraction, you can jump straight here. Point to your corpora folder, choose a corpus from the dropdown, and start exploring your data through two complementary views:

### Ego Graphs

Ego graphs let you examine each paper individually. They show the topics, authors, and other metadata associated with a single paper as a local graph. Use the checkboxes to toggle which node types are displayed, then click **"Generate Ego Graph"** to refresh the view.

You can download any visualisation as an HTML file for offline use or sharing.

### Full Corpus Graph

The Full Corpus Graph brings your entire collection together in one interactive view. Click **"Generate Full Corpus Graph"** to render the complete network of interconnections across all papers.

If the graph feels overwhelming, open the **Advanced** panel to filter which papers are included — then regenerate the graph to focus on what matters most. Like the ego graph, you can export this view as HTML for offline exploration.

## Data privacy and security

* **Your papers stay where they are.** Text Lab reads the PDFs from the folder you enter and does not copy them anywhere else permanently.
* **Output goes where you choose.** The corpus folder with the extracted metadata, tables and topics is saved in the location you pick in Step 1, and the graphs you generate are saved in it as HTML pages. It stays there until you delete it, so you can come back to it in a later session.
* **Temporary files are deleted.** While Grobid parses a paper, it keeps a working copy in your session's private workspace on the compute node, which only your user account can open. The workspace is deleted when your Text Lab session ends.
* **Choice of language model for topics.** With **Ollama (Local)**, the title and abstract of each paper are processed by a model running in your own session on the compute node. With **GPUStack**, they are sent to the University of Bern's GPUStack service: they leave your session, but stay on university infrastructure and are not sent to external providers. Choose Ollama for papers that must not leave your session.
