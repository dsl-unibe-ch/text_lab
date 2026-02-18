# Knowledge Graph

The Knowledge Graph feature uses large language models (LLMs) to automatically extract key topics from the abstracts of your uploaded scientific papers. It then builds an interactive graph that reveals hidden connections across your literature collection — making it easy to explore themes, spot patterns, and navigate complex research landscapes.

---

## Step 1 — Build Your Corpus

Start by pointing the app to a folder containing your PDF collection. The app will scan the folder and list all detected PDFs so you can confirm you're working with the right files. Once ready, click **"Generate Corpus"** to kick off the extraction process.

Under the hood, this step uses [GROBID](https://grobid.readthedocs.io/en/latest/Introduction/) to parse each PDF and convert it into a structured XML format optimised for downstream processing. The output is saved to a new folder named `your_collection_project_corpus`.

> **Smart caching:** This step is computationally intensive, but the app checks for an existing corpus before reprocessing. It will only reprocess papers that are new or have changed — and reports exactly how many files were updated.

Once your corpus is built and nothing has changed, you can skip straight to Step 2 on future runs.

A **CSV export** is also generated at this stage, giving you a human-readable summary of the extraction results. Use it to spot any issues with how GROBID processed individual papers.

> **Lost your CSV?** No problem — use **Step 1b: "Rebuild Corpus Table"** to regenerate it at any time. Just point to the folder containing your corpora; they'll be detected automatically so you can select the one you need.

---

## Step 2 — Extract Topics

With your corpus ready, the app uses an LLM to read each paper's abstract and extract **5–8 concise topic phrases**. Each topic is then assigned a category, which serves as a shared node when connecting papers in the graph.

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
