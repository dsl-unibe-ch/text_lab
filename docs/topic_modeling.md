# Topic Modeling

Topic Modeling finds the main themes in a collection of texts automatically. You upload your documents, and Text Lab groups documents that talk about similar things into **topics**, describes each topic with its most characteristic keywords, and tells you which topic every document belongs to.

Typical uses are exploring survey answers, interview transcripts, news articles, social media posts, abstracts, or any other collection that is too large to read in full.

All processing runs on the university cluster. Your documents are never sent to an external service.

## Quick start

1. **Upload your data** as a table (CSV or Excel) or as a ZIP archive of text files.
2. **Pick the column** that contains the text (tables only).
3. **Choose an algorithm.** If you are unsure, start with **BERTopic** and the default settings.
4. **Select the primary language** of your texts.
5. Click **Run Topic Extraction**. A status box shows each step; to cancel, refresh the page. Notes about the run, such as skipped rows or long documents, appear when it finishes.
6. **Explore the results** on the page and click **Download Extraction Package (.zip)** to keep them.

Running again with different settings replaces the previous results on the page, so download anything you want to keep first.

## Preparing your data

Each **document** is one unit that gets assigned to a topic: one table row or one text file.

### Tables (recommended)

* Formats: **CSV** (`.csv`) or **Excel** (`.xlsx`). Older `.xls` files are not supported; open them in Excel and save them as `.xlsx` or `.csv`.
* Use **one row per document** and one column for the text. All other columns (author, date, category, ...) are kept and appear in the exported results next to the assigned topic, so you can analyse topics by group later.
* CSV files may use commas, semicolons, tabs or pipes as separators, and may be saved by Excel on Windows. Text Lab detects both automatically.
* Rows with an empty text cell are skipped.

### ZIP archives

* Put **plain text files** (`.txt`) into a ZIP archive, one document per file. Up to 1,000 files are accepted.
* The file name is kept and appears in the exported results.
* Use this when your documents are separate files; use a table if you also have metadata.

### Tips for good results

* **Choose a sensible document size.** Topic models work best when each document is about one thing: an answer, a paragraph, an abstract or an article. A whole book as one document usually mixes many themes; consider splitting it into chapters or paragraphs.
* **More is better.** At least 5 documents are required, but meaningful topics usually need a few hundred. For small collections, use LDA.
* **Remove boilerplate** such as email signatures, page headers or legal notices, or add the words as custom stopwords (see below).

## Choosing an algorithm

| | **BERTopic** | **Top2Vec** | **LDA** |
|---|---|---|---|
| **How it works** | Uses a language model to understand the meaning of each document, then groups similar documents | Learns word and document meanings together, then groups similar documents | Counts which words occur together in documents |
| **Best for** | Most use cases; short and medium texts; many languages | Medium to large collections | Small collections; when you want a classic, well-known method |
| **Number of topics** | Found automatically, or set by you | Found automatically, or set by you | Always set by you |
| **Documents that fit no topic** | Marked as *Outlier* (HDBSCAN) | Always assigned to the closest topic | Always assigned to the most likely topic |
| **Topics over time** | Yes | No | No |
| **Speed** | Fast; the language model runs on the GPU | Slower, especially with *deep-learn* | Fast for small and medium collections |

!!! tip "Not sure? Start with BERTopic"
    BERTopic with the default settings gives good topics for most collections. If it reports that it could not find any topics, your collection is probably too small: switch to LDA.

## Settings

### Settings for all algorithms

* **Primary Language** decides which built-in list of common words (stopwords such as "the", "and", "der", "und") is removed, and which language model BERTopic uses by default.
    * English, German and French: full support, including word base forms for LDA (for example "studies" and "studied" both count as "study").
    * Spanish, Italian, Dutch, Portuguese, Russian and Arabic: built-in stopword lists.
    * Chinese: use **BERTopic**, which splits Chinese text into words. LDA and Top2Vec do not handle Chinese well.
    * Other / Mixed: uses a multilingual model, without a built-in stopword list.
* **Custom Stopwords** (BERTopic and LDA): a comma-separated list of words that should never appear as topic keywords, for example words that occur everywhere in your collection (`patient, report, interview`). Not available for Top2Vec.
* **Evaluate Topic Stability / Reproducibility** (in *Academic Evaluation Metrics*): runs the model three times and measures how similar the topics are. Useful to check whether your topics are robust, but the run takes about three times as long.

### BERTopic

**Core settings**

* **Clustering Engine**
    * **HDBSCAN** (default) finds the number of topics itself and sets aside documents that do not fit any topic as *Outlier*.
    * **KMeans** puts every document into exactly the number of topics you choose. There are no outliers.
* **Auto-detect optimal number of topics** (HDBSCAN): keep this on to let BERTopic merge very similar topics automatically. Turn it off to ask for a target number of topics.
* **Minimum Documents per Topic** (HDBSCAN): the smallest group of documents that counts as a topic. Lower it to get more, smaller topics; raise it to get fewer, broader ones.

**Embedding Model**

The embedding model is the language model that reads your documents. The defaults work well and are already available on the cluster:

* English: `all-MiniLM-L6-v2`, which reads up to 256 tokens (roughly 190 words) per document.
* Other languages: `paraphrase-multilingual-MiniLM-L12-v2`, which reads up to 128 tokens (roughly 95 words) per document.

If your documents are longer than this, Text Lab shows a warning, because by default the model only reads the beginning of each document. You have two options:

* **Embed long documents in chunks**: tick this box to have each long document split into pieces that fit the model, read piece by piece, and combined. The whole document then shapes its topic. This takes longer for long documents.
* **Use a model with a larger context window**: choose *Custom HuggingFace model* and enter a model ID such as `jinaai/jina-embeddings-v2-base-en`, `BAAI/bge-m3` or `nomic-ai/nomic-embed-text-v1.5` (about 8,000 tokens each). The model is downloaded to your home directory the first time.
    * Some models (Jina, Nomic) need **Allow custom code from this model repository**. Only tick it for models from sources you trust, because their code runs under your account.

**Text Extraction & Vocabulary**

* **Extract Phrases (N-grams)**: also consider two-word phrases such as "climate change" as keywords.
* **Minimum Topic Frequency (min_df)**: a word is only used as a keyword if it appears in at least this many topics. Keep it at 1 unless the run runs out of memory on a very large collection; higher values remove the words that make topics distinctive.
* **Auto-penalize frequent words**: reduces the weight of words that appear in almost every topic. Keep it on.

**Dimensionality Reduction**

Leave **UMAP** selected unless you have a reason to change it. **Lock Seed for Reproducibility** (on by default) makes repeated runs with the same settings give the same topics.

**Clustering & Outliers (HDBSCAN)**

* **Outlier Sensitivity (min_samples)**: lower values put fewer documents into *Outlier*.
* **Force-assign outliers to nearest topics**: assigns every outlier to its closest topic, so no document is left without a topic. Topic sizes and keywords are updated accordingly.

### Top2Vec

* **Auto-detect optimal number of topics**: keep this on, or turn it off and choose a target number. If you ask for more topics than Top2Vec finds, it keeps the topics it found.
* **Embedding Backend**
    * **Transformer (Pre-trained)**: uses a ready-made language model. Faster, and works well for general language.
    * **Doc2Vec (Train from scratch)**: learns from your documents only. Can capture specialised vocabulary but needs a larger collection.
* **Training Depth**: *fast-learn*, *learn* or *deep-learn*. Deeper training can give better topics with Doc2Vec but takes longer.
* **Minimum Word Count (min_count)**: words that appear fewer times than this in the whole collection are ignored. Lower it for small collections or if Top2Vec finds no topics; raise it to ignore rare words in large collections.

Top2Vec results can differ slightly between runs, even with the same settings.

### LDA

* **Number of Topics**: LDA always produces exactly this many topics. Try a few values (for example 5, 10 and 20) and compare the results.
* **Extract Phrases (Bigrams)**: also consider frequent two-word phrases, which appear as keywords joined by an underscore (`climate_change`).
* **Training Passes**: more passes give more stable topics but take longer. The default of 10 is fine for most collections.

### Topics over time (BERTopic)

To see how topics rise and fall over time, upload a table, tick **Analyze Topics Over Time** next to the text column, then:

* **Timestamp Column**: a column with dates (`2021-03-15`), date-times, or whole years (`2021`).
* **Number of Time Bins**: how many time intervals to split the period into. If your data has fewer distinct dates than that (for example five years), each date is shown as its own point.

Rows whose date cannot be read are skipped, and Text Lab tells you how many.

## Understanding the results

### Evaluation metrics

The scores at the top help you **compare runs on the same data**, for example two numbers of topics. They are not absolute quality grades, and they never replace reading the topics yourself.

| Metric | What it tells you | Better |
|---|---|---|
| **Topic Diversity** | How different the topics' keywords are from each other | Higher |
| **Coherence (C_v)** | How often a topic's keywords appear together in your documents; the measure that best matches human judgement | Higher (0 to 1) |
| **Coherence (C_npmi)** | A related co-occurrence measure | Higher (-1 to 1) |
| **Coherence (U_mass)** | A co-occurrence measure based on document counts | Closer to 0 |
| **LDA Perplexity** (LDA only) | How well the model predicts the documents it was trained on | Lower |
| **Topic Stability** (if enabled) | How similar the topics are across three runs | Higher (100% = identical) |

A metric shows **N/A** when it cannot be computed for a run, for example when too few keywords occur in the documents. Expand **How are these scores computed?** on the page for details.

### Topic Dictionary

The table lists every topic with:

* **Topic**: the topic number. The same numbers are used in all charts and in the exported files.
* **Count**: how many documents belong to the topic (BERTopic and Top2Vec).
* **Keywords**: the ten words that best characterise the topic. Read them together to give the topic a name.

Documents marked as *Outlier* (BERTopic with HDBSCAN) are not listed as a topic, but they appear in the exported document table.

### Interactive charts

* **BERTopic**
    * *Intertopic Distance*: each circle is a topic, larger circles hold more documents, and topics close to each other are similar. Use the slider to highlight a topic.
    * *Word Scores*: the most important keywords of the largest topics.
    * *Similarity Heatmap*: how similar every pair of topics is. Dark squares away from the diagonal point to topics you might merge.
    * *Topics Over Time*: how often each topic appears in each time interval, if you enabled it.
* **Top2Vec**: the most characteristic words of the largest topics.
* **LDA**: an interactive map of topics on the left and the keywords of the selected topic on the right. Move the relevance slider towards 0 to see words that are specific to the selected topic.

## Downloading the results

**Download Extraction Package (.zip)** contains:

* `document_topics.csv`: your original data with two new columns at the front.
    * **Dominant_Topic**: the topic of each document, or *Outlier*.
    * **Topic_Confidence**: how strongly the document belongs to that topic; higher means a clearer match. It is empty when BERTopic uses KMeans or force-assigned outliers.
* `topic_keywords.csv`: the Topic Dictionary.
* `run_configuration.txt`: every setting and score of the run. Keep it with your results so that you can report and reproduce your analysis.
* The interactive charts as `.html` files, which open in any web browser. They need an internet connection to display.

The CSV files use commas as separators. If Excel shows everything in one column, import the file with *Data > From Text/CSV* instead of double-clicking it. If you used topics over time, the rows are sorted by date, and rows without a valid date are left out.

## Troubleshooting

Click a problem to see what to do.

??? question "'BERTopic/Top2Vec could not find any topics in these documents'"
    The collection is too small or too uniform to form groups. Add more documents or switch to LDA.

??? question "Many documents are marked as Outlier"
    Lower *Minimum Documents per Topic* or *Outlier Sensitivity*, tick *Force-assign outliers to nearest topics*, or switch the clustering engine to KMeans.

??? question "Too many or too few topics"
    BERTopic: change *Minimum Documents per Topic*, or turn off auto-detection and set a target number. LDA: change *Number of Topics*.

??? question "Keywords are full of generic words"
    Add them to *Custom Stopwords*, keep *Auto-penalize frequent words* on, and check that the *Primary Language* matches your texts.

??? question "'No topic keywords could be extracted with a Minimum Topic Frequency ...'"
    Set *Minimum Topic Frequency (min_df)* back to 1.

??? question "Top2Vec finds no topics on a small collection"
    Lower *Minimum Word Count (min_count)*, use the *Transformer* backend, or switch to LDA.

??? question "Warning that documents exceed the context window"
    Tick *Embed long documents in chunks*, or choose a model with a larger context window (see [Embedding Model](#bertopic)).

??? question "'The timestamp column contains numbers that are not years'"
    Number columns are only read as whole years (`2021`). Use a column with real dates, or convert numeric dates such as `20210315` to `2021-03-15`.

??? question "The embedding model cannot be loaded"
    Check the model ID on huggingface.co. If the message mentions custom code, tick *Allow custom code from this model repository* (for trusted models only).

??? question "Results change between runs"
    For BERTopic, keep *Lock Seed for Reproducibility* on. LDA always uses a fixed seed. Top2Vec cannot be fixed, so small differences are expected.

For help choosing settings for your project, contact the Data Science Lab (DSL).
