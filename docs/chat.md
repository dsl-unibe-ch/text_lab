# Chat

Chat lets you talk to large language models (LLMs) that run on the university cluster. You can ask questions, brainstorm research ideas, draft and edit text, or upload documents to summarise, translate or question them. If you upload a table, Chat can also create plots and run statistical tests on it.

Your messages and files never leave the university network. Nothing is sent to external services such as OpenAI, Google or Anthropic.

## Quick start

1. **Choose a model** in the sidebar under *Model Selection*. If you are unsure, keep the first one.
2. **Attach files** (optional) under *Upload Context*: up to 4 documents or tables.
3. **Type your message** in the box at the bottom of the page and press Enter.
4. **Wait for the first reply.** The first message loads the model into GPU memory, which can take 1–2 minutes. Later replies start within seconds.
5. **Download Conversation (.md)** in the sidebar to keep the conversation.

Click **Start New Chat** to clear the conversation and begin again. Download anything you want to keep first: a cleared conversation cannot be restored.

## Choosing a model

The models available depend on the GPU your Text Lab session runs on. The sidebar shows the mode:

* **Standard Mode**: smaller models only. They answer quickly and are fine for most everyday tasks.
* **High-Performance Mode** (large GPUs such as the A100 or H100): all models, including large ones that give better answers for complex questions, at the cost of slower replies.

You can switch models at any time; the conversation continues with the new model. Models differ in strengths, so if one gives poor answers for your task, try another.

## Working with documents

Attach files in the sidebar under *Upload Context*. Attached files are included with every message you send, until you remove them.

| File type | What Chat does with it |
|---|---|
| **PDF** (`.pdf`) | Reads the text of the document |
| **Text** (`.txt`) | Reads the text |
| **CSV, Excel** (`.csv`, `.xlsx`, `.xls`) | Reads the first 100 rows as a table, and enables [data analysis](#analysing-data-in-chat) on the whole table |
| **TSV, JSON** (`.tsv`, `.json`) | Enables [data analysis](#analysing-data-in-chat); not read as text |

* **Limits:** up to **4 files**, each up to **10 MB**. Larger files are skipped with a warning.
* **Scanned PDFs** contain images of text rather than text, so Chat cannot read them. Convert them with [OCR](ocr.md) first.
* **Long documents** are handled automatically. When a document is too long for the model to read at once (roughly 50,000 words), Chat splits it into parts, answers your question for each part, and then combines the answers. You see the progress as *Analyzing document part 2 of 5*. This takes longer, and questions about details spread across the whole document are answered less precisely than for shorter documents.

!!! tip "Ask specific questions"
    "Summarise the methods section of `study.pdf` in five bullet points" gives a better answer than "What is this about?". Name the file when you have attached several.

## Analysing data in Chat

When you attach a table, the sidebar shows **Data tools enabled for *your file***. You can then ask for plots or statistics in plain language, for example:

* "Plot the distribution of age."
* "Is income different between men and women?"
* "Which variables are most strongly correlated with the outcome?"

For each message, Chat decides whether it needs the data tools. If it does, an activity log (*Analysing your data...*) shows the progress, and the answer contains a summary, the plots, the statistical results and the Python code that reproduces them. All other messages are answered as normal chat.

The analysis uses the same engine as [Visualize Data](visualize_data.md); see there for the available plots and tests and how to read the results.

* If you attach several tables, only the **first** one is analysed.
* Analyses use the whole table, up to 300,000 rows.

!!! tip "Ask for statistics, not mental arithmetic"
    In normal chat the model only sees the first 100 rows of a table, so it cannot reliably answer questions such as "What is the average income?". Ask for a statistic or a plot instead ("Compute the mean income per region"), which uses the data tools on the complete table.

For more control, such as choosing columns, getting R code or downloading a full report, use [Visualize Data](visualize_data.md).

## Saving your conversation

The sidebar offers downloads once the conversation has started:

* **Download Conversation (.md)**: the whole conversation as a Markdown text file, including any code.
* **Download with Plots (.html)**: appears when an analysis has produced plots. Contains the conversation together with the charts and opens in any web browser.

The conversation is lost when you click *Start New Chat*, reload the page, close the browser tab, or when your Text Lab session ends.

## Data privacy and security

You may be uploading sensitive, unpublished or proprietary research data to Chat. Text Lab is designed to keep it confidential. Here is exactly what happens to your data, and the technology behind it.

* **Documents are processed in memory.** Documents you attach for the conversation (PDF, text, and tables read as text) are processed entirely in the server's temporary working memory (RAM). They are not saved, copied or written to your university home directory.
    * *Technical transparency:* the app uses Streamlit's `uploaded_file.getvalue()` to read the file into RAM. PDFs are read with PyMuPDF (`fitz.open(stream=...)`), tables with pandas from a memory buffer (`pd.read_csv(BytesIO(...))`).
* **Tables for data analysis are stored privately.** So that the analysis tools can read a table, Chat saves a copy of the first attached table, together with the plots created from it, in a folder in your own home directory: `~/.cache/text_lab/mcp_artifacts/`. Only your user account can open this folder. These files are **not deleted automatically**; delete the folder when you no longer need them.
* **Ephemeral sessions.** Your conversation is kept only in your open browser tab (Streamlit's `st.session_state`). When you click *Start New Chat*, reload the page, close the browser, or your HPC job ends, the conversation is permanently deleted.
* **No AI training.** The models run locally on the UBELIX compute nodes via Ollama. They only read your data to answer your current question (inference). **The models do not learn from your data**, and your data is never used to train or improve them.
* **Network isolation.** All data stays within the University of Bern's secure HPC network. Nothing is ever sent to external services such as OpenAI, Google or Anthropic.

!!! note "Large file uploads and temporary storage"
    Chat accepts documents of up to **10 MB**, to keep them within what the model can read. The Text Lab server itself accepts much larger files (up to 10 GB) to support the Visualize Data and Knowledge Graph tools.

    If you upload a very large file to Chat, the web framework (Streamlit) may temporarily store it in the operating system's temporary directory (`$TMPDIR`) to avoid running out of memory, *before* Chat rejects it for being too large. For the strictest data privacy, keep to the 10 MB limit.

## Good to know

* **AI answers can be wrong.** Language models can produce inaccurate, misleading or invented statements that sound convincing. Check important facts, quotes and numbers against your sources.
* **No internet access.** The models cannot browse the web, and their knowledge ends at the date they were trained.
* **No memory between chats.** Each conversation starts from scratch. The model only knows what is in the current conversation and the attached files.

## Troubleshooting

Click a problem to see what to do.

??? question "'Could not connect to Ollama server'"
    The service that runs the models is not available, usually because your Text Lab session has just started. Wait a minute and reload the page. If the message stays, end your session and start a new one.

??? question "The first reply takes a long time"
    The model is being loaded into GPU memory, which takes 1–2 minutes the first time you use it (the page shows *Loading ... into GPU memory*). Replies are fast afterwards. Switching to another model loads that model.

??? question "'File ... exceeds the 10MB limit. Skipped.'"
    Shorten the document or attach only the relevant part. To analyse a large table, use [Visualize Data](visualize_data.md), which accepts much larger files.

??? question "The answer ignores my PDF"
    The PDF is probably scanned and contains no text layer. Convert it with [OCR](ocr.md) and attach the result.

??? question "My request for a plot or statistic was answered as normal chat"
    Check that the sidebar shows *Data tools enabled* for your table. Then ask explicitly for the plot or test, for example "Make a histogram of age" or "Run a t-test of income by sex".

??? question "'The analysis could not be started. The model produced a tool call Ollama could not parse'"
    Some models occasionally produce instructions that the system cannot read, especially with certain data. Select another model in the sidebar and ask again.

??? question "'The model did not answer within 300 seconds'"
    The model server was busy or stuck. Ask again, or try another model. If nothing responds any more, end your Text Lab session and start a new one.

??? question "'Ollama ResponseError'"
    The model server reported an error. Try again or choose another model. If the error keeps appearing, contact the Data Science Lab and include the message shown.

For help with your project, contact the Data Science Lab (DSL).
