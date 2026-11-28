# Topic Modeling

!!! note "Draft for the new interface"
    This guide describes the Topic Modeling page of the new Text Lab
    interface. It replaces [the current guide](../topic_modeling.md) when
    the new interface goes live.

Topic Modeling finds the main themes in a collection of texts. Text Lab
groups documents that talk about similar things into **topics**, describes
each topic by its most typical keywords, and tells you which topic each
document belongs to.

Typical uses are survey answers, interview transcripts, news articles,
social media posts, abstracts, or any collection too large to read in full.

Everything runs on the University's computing cluster. Your documents are
never sent to outside services.

## Quick start

1. **Drop your collection** on the form: a table (CSV or Excel) or a ZIP of
   text files. It is read at once, and its columns and first rows appear.
2. **Check the text column** (tables only). Text Lab chooses the column
   with the longest texts for you.
3. **Choose an algorithm** and the language of the texts. If you are
   unsure, keep **BERTopic** and the default settings.
4. Press **Find topics**.

The run continues in the background; you can leave the page and come
back. To try other settings, change them and press **Find topics** again:
the collection stays, and every run keeps its own record, so you can
compare them.

## Preparing your data

Each **document** is one unit that gets a topic: one table row or one
text file.

### Tables (recommended)

- **CSV** (`.csv`) or **Excel** (`.xlsx`). Save older `.xls` files as
  `.xlsx` or `.csv` first.
- **One row per document**, with one column for the text. All other
  columns (author, date, group, ...) stay next to each document's topic in
  the download, so you can compare topics by group later.
- CSV files may use commas, semicolons, tabs or pipes, and may come from
  Excel on Windows; Text Lab recognizes both.
- Rows without text are left out.

### ZIP archives

- **Plain text files** (`.txt`) in a ZIP, one document each, up to 1,000
  files. Each file's name stays next to its topic in the download.

### Tips for good results

- **A document should be about one thing**: an answer, a paragraph, an
  abstract, an article. A whole book as one document mixes many themes;
  split it into chapters or paragraphs.
- **More is better.** At least 5 documents are needed, but good topics
  usually need a few hundred. For small collections, use LDA.
- **Remove boilerplate** such as signatures or page headers, or add its
  words to *Words to leave out*.

## Choosing an algorithm

| | **BERTopic** | **Top2Vec** | **LDA** |
|---|---|---|---|
| **How it works** | A language model reads each document; documents of similar meaning form a topic | Learns word and document meanings together, then finds where documents gather | Counts which words occur together |
| **Best for** | Most collections; short and medium texts; many languages | Medium and large collections | Small collections; a classic, well-known method |
| **Number of topics** | Found, or reduced to a number | Found, or reduced to a number | Always set by you |
| **Documents that fit no topic** | Left as outliers (by density) | Always given the closest topic | Always given the most likely topic |
| **Topics over time** | Yes | No | No |

Try more than one: topics that several algorithms find are the most
reliable.

## Settings

The defaults suit a first run. Change a few at a time and compare.

### For all algorithms

- **Language of the texts** picks the common words to leave out ("the",
  "and", "der", "und") and, for BERTopic, the language model: an English
  one for English, a multilingual one for the others. English, German and
  French also get word base forms for LDA. For Chinese, use BERTopic.
- **Words to leave out** (BERTopic and LDA): words of your field, separated
  by commas, that would otherwise name every topic (`patient, report,
  interview`).
- **Check stability**: runs the model three times with other random seeds
  and compares the topics. About three times as long.

### BERTopic

- **Grouping**: *By density (HDBSCAN)* finds the topics itself and leaves
  documents that fit none as outliers; *Into a set number (KMeans)* puts
  every document into one of the number of topics you give.
- **Number of topics** (by density): *Find it*, or *Reduce to* a number:
  the topics found are merged until that many are left.
- **Documents per topic, at least**: smaller finds more, smaller topics;
  larger finds fewer, broader ones.
- **Outliers**: a lower *Outlier sensitivity* leaves fewer documents as
  outliers; *Give every outlier the topic it is closest to* leaves none.
- **Language model**: after the language (the default), English
  (`all-MiniLM-L6-v2`, reads up to 256 tokens of a document),
  multilingual (`paraphrase-multilingual-MiniLM-L12-v2`, 128 tokens), or
  another sentence-transformers model from Hugging Face, downloaded into
  your own folder on first use. For long documents, choose a model with a
  longer context, such as `BAAI/bge-m3`, or tick *Read long documents in
  pieces*: each piece is read and the results are averaged, so the whole
  document shapes its topic. Some models need *Let the model run code from
  its repository*; allow it only for repositories you trust.
- **Keywords**: also find phrases of two words; weigh down words frequent
  everywhere (keep it on); *A keyword appears in at least this many
  topics* (keep 1).
- **Dimension reduction**: keep UMAP unless you have a reason. *Fix the
  random seed* (on) makes a run repeatable.

### Top2Vec

- **Number of topics**: *Find it*, or *Reduce to* a number.
- **Embeddings**: a pre-trained transformer (knows the language) or
  Doc2Vec (learns from your texts alone and needs more of them).
- **Training depth**: *fast-learn*, *learn* or *deep-learn*; deeper is
  slower and can be better with Doc2Vec.
- **Words seen at least**: rarer words are left out. Lower it for small
  collections, or if no topics are found.

Top2Vec reads the texts as they are, so there are no words to leave out.

### LDA

- **Topics**: LDA always gives this many. Try a few (5, 10, 20) and
  compare.
- **Training passes**: more are steadier and slower; 10 suits most
  collections.
- **Also find phrases of two words**: they appear joined by an underscore
  (`climate_change`).

### Topics over time (BERTopic)

For a table, tick **Topics over time** below the text column, choose the
**timestamp column** (dates such as `2021-03-15`, date-times, or years such
as `2021`) and at most how many **intervals** to split the period into.
With fewer distinct times, each is its own interval. Rows whose time
cannot be read are left out, and the result says how many.

## The result

Each algorithm's result is shown the way its own library shows it, so it
looks as you know it from pyLDAvis or BERTopic. Everywhere, pointing at a
topic marks it in red in the chart and in the list beside it.

### LDA, as pyLDAvis shows it

- **Topics**: on the left, the **intertopic distance map**: each topic is
  a bubble, alike topics lie close, and a bubble's area is the topic's
  share of all the words. On the right, the **terms**: before you choose
  a topic, the 30 most *salient* terms of the whole collection; once you
  choose one (click its bubble, or use the list or the arrows), its 30 most
  *relevant* terms. The grey bar is a term's frequency in the collection,
  the red bar its estimated frequency in the topic. The **relevance
  slider (λ)** changes the ranking: at 1, terms rank by how probable they
  are in the topic; towards 0, by how specific they are to it. Around 0.6
  often reads best. Point at a term to see how it spreads over the
  topics: the bubbles change size.
- **Documents**: under each topic, the documents most about it. As LDA
  gives every document a mix of topics, each comes with a thin bar of its
  mix, the topic in red, and the mix in words beside it.

### BERTopic, with BERTopic's charts

- **Topics**: one small chart per topic, its keywords with their c-TF-IDF
  scores, and a few example documents.
- **Topic map**: each topic a bubble sized by its documents, alike topics
  close. Click a bubble to bring its topic into view in the list.
- **Documents**: every document as a dot, placed by meaning (UMAP of its
  embedding), each topic's number at the middle of its documents. Point at
  a topic to light up its documents; dots in no topic are lighter. Large
  collections show a random sample of 1,500 documents.
- **Hierarchy**: how the topics merge into larger ones. Topics that join
  early are alike and may be worth merging.
- **Similarity**: how alike each two topics are, in the hierarchy's order,
  so groups of alike topics show as dark blocks.
- **Over time** (if you asked for it): one line per topic, all on one
  axis. Point at a point for the count and the topic's words in that
  interval: a topic's words can change over time. A table holds the same
  counts.

### Top2Vec, as Top2Vec shows its topics

- **Topics**: a **word cloud** per topic, each word as large as it is close
  to the topic, and a few example documents.
- **Sizes**: the topics by size, the largest first.
- **Topic map**, **Documents** and **Similarity**, as for BERTopic.

### For all algorithms

- **Quality**: the scores, with what each says. They help to **compare
  runs on the same collection**; they never replace reading the topics.

| Score | What it tells you | Better |
|---|---|---|
| **Topic Diversity** | How different the topics' keywords are | Higher |
| **Coherence (C_v)** | How often a topic's keywords occur together; closest to human judgement | Higher (0 to 1) |
| **Coherence (C_npmi)** | A related co-occurrence measure | Higher (-1 to 1) |
| **Coherence (U_mass)** | Co-occurrence by document counts | Closer to 0 |
| **LDA Perplexity** (LDA) | How well the model predicts its documents | Lower |
| **Topic Stability** (if checked) | How alike the topics are across three runs | Higher (100% = the same) |

A score shows *Not computed* when a run does not allow it, for example
when too few keywords occur in the texts.

**Cancel** stops a run at once; nothing is kept then.

## Downloads

- **Everything, with the interactive charts** (`_topics.zip`): all of the
  below, and the libraries' own interactive charts (pyLDAvis, BERTopic's
  Plotly charts) as `.html` files, which open in a web browser with an
  internet connection.
- **Documents with their topic** (`_document_topics.csv`): your data with
  two columns in front: **Dominant_Topic** (the topic, or *Outlier*) and
  **Topic_Confidence** (how clearly it belongs; empty for KMeans or
  reassigned outliers). With topics over time, the rows are sorted by
  time and rows without a time are left out.
- **Topics and keywords** (`_topic_keywords.csv`).
- **Settings and scores** (`_run_report.txt`): every setting and score of
  the run, to report and repeat your analysis.

If Excel shows a CSV in one column, import it with *Data > From
Text/CSV*.

## Troubleshooting

??? question "'BERTopic/Top2Vec could not find any topics in these documents'"
    The collection is too small or too uniform to form groups. Add
    documents, or use LDA.

??? question "Many documents are in no topic"
    Lower *Documents per topic* or the *Outlier sensitivity*, give every
    outlier its closest topic, or group *Into a set number*.

??? question "Too many or too few topics"
    BERTopic: change *Documents per topic*, or *Reduce to* a number. LDA:
    change *Topics*.

??? question "Keywords are full of generic words"
    Add them to *Words to leave out*, keep *Weigh down words that are
    frequent everywhere* on, and check the language.

??? question "'No topic keywords could be extracted with a Minimum Topic Frequency ...'"
    Set *A keyword appears in at least this many topics* back to 1.
