# Chat

!!! note "Draft for the new interface"
    This guide describes the Chat page of the new Text Lab interface. It
    replaces [the current guide](../chat.md) when the new interface goes
    live.

Chat lets you talk to large language models (LLMs) that run on the
University's computing cluster. You can ask questions, brainstorm research
ideas, draft and edit text, or attach documents to summarize, translate or
question them. If you attach a table, Chat can also draw charts and run
statistical tests on it.

Your messages and files never leave the University network. Nothing is
sent to external services such as OpenAI, Google or Anthropic.

## Quick start

1. **Type your message** in the box at the foot of the page and press
   **Enter** (**Shift+Enter** starts a new line).
2. **Attach files** (optional): click **Attach** beside the box, or drop
   files anywhere on the page. Up to 4 documents or tables.
3. **Choose a model** (optional) in the list beside the Send button. If you
   are unsure, keep the first one.
4. **Watch the answer being written.** The first message to a model loads
   it into the GPU's memory, which can take a minute or two; later answers
   start within seconds.

The conversation stays when you reload the page or visit another tool.
**New chat** in the panel on the right starts another one; your earlier
conversations stay listed there until your Text Lab session ends.

## The answer

The answer appears word by word as the model writes it. While it is
written:

- **Stop** (the square in place of the Send button) stops it at once.
  What was written so far stays in the conversation, marked *Stopped*.
- Some models **think before they answer**. Their thinking appears
  folded as *Thinking*, with its newest words beside it; open it to read
  along. When the answer is done it reads *Thought for 12 s* and stays
  folded.

Under each answer:

- **Copy** copies the answer as Markdown, which keeps its headings, lists
  and tables. Each block of code has its own **Copy**.
- **Again** (the last answer only) writes it again, with the model chosen
  in the list. Use it to compare models on the same question.

If another tool is using the GPU, such as a transcription, the answer
waits for it and says so. Jobs use the GPU one at a time, so they never
slow each other down; you can cancel the other job from there.

## Conversations

The panel on the right (behind **Conversations** on narrow screens) lists
the conversations of your session, the most recent first.

- **New chat** starts an empty conversation. Click a conversation to go
  back to it.
- **Delete this conversation** removes it with its files and charts.
- Conversations are kept until your Text Lab session ends; download the
  ones you want to keep.

## How much the model can read

A model reads a limited amount of text at once, its *context window*.
Your whole conversation and your attached files are sent with every
message, so a conversation fills the window as it grows. The panel shows
how full it is under **Context used**, for example *About 6,200 of 21,000
tokens, 30%*. A token is a piece of a word; 1,000 tokens are roughly 750
English words. The number is an estimate.

- **Answers get less reliable as the window fills**, well before it is
  full: the model pays less attention to details in a long conversation.
  From 75% on, the panel suggests starting a new chat.
- **When the window is full**, the model no longer sees the oldest
  messages, and the panel says so. Start a new chat; if you need earlier
  results, paste a short summary of them into your first message.
- **Long attached files** are read in parts (see below), so they do not
  fill the window on their own. A new chat does not help with those, so
  the panel does not suggest it for them.
- **A message itself** cannot be longer than the window. Save a long text
  as a `.txt` file and attach it instead: files are read in parts.

## Choosing a model

The models offered depend on the GPU your Text Lab session runs on; the
panel names it. On GPUs with much memory (such as the A100 or H100) the
large models are offered too, which give better answers to complex
questions, at the cost of slower answers. You can change the model for
any message; the conversation continues with the new model.

## Working with documents

Attached files are shown above the message box and are sent with every
message until you remove them (the cross beside the file). Each message
lists the files it was sent with.

| File type | What Chat does with it |
|---|---|
| **PDF** (`.pdf`) | Reads the text of the document |
| **Text** (`.txt`) | Reads the text (UTF-8) |
| **JSON** (`.json`) | Reads the text, and enables [data analysis](#analysing-data) if it holds a table |
| **CSV, TSV, Excel** (`.csv`, `.tsv`, `.xlsx`, `.xls`) | Reads the first 100 rows as a table, and enables [data analysis](#analysing-data) on the whole table |

- **Limits:** up to **4 files** at a time, each up to **10 MB**.
- **Files Chat cannot read** are not attached, and a note says why: for
  example damaged or password-protected PDFs, or text files that are not
  UTF-8.
- **Scanned PDFs** hold images of text rather than text, so Chat cannot
  read them and says so. Convert them with [OCR](ocr.md) first.
- **Long documents** are handled automatically. When a document is too
  long for the model to read at once (roughly 15,000 words, about 30
  pages), Chat splits it into parts, answers your question for each part,
  and then combines the answers. You see the progress as *Analyzing
  document part 2 of 5*. This takes longer, and questions about details
  spread across the whole document are answered less precisely than for
  shorter documents.

!!! tip "Ask specific questions"
    "Summarize the methods section of `study.pdf` in five bullet points"
    gives a better answer than "What is this about?". Name the file when
    you have attached several.

## Analysing data

When you attach a table, it is marked **Charts**. You can then ask for
charts or statistics in plain language, for example:

- "Plot the distribution of age."
- "Is income different between men and women?"
- "Which variables are most strongly correlated with the outcome?"

For each message, Chat decides whether it needs the data tools. If it
does, the answer shows the data agents' steps as they work, then a
summary, the charts, the statistical results, and the Python (and R, where
there is one) code that reproduces each. Interactive charts can be zoomed,
and their points read by pointing at them. All other messages are
answered as normal chat.

The analysis uses the same engine as [Visualize Data](visualize_data.md);
see there for the available charts and tests and how to read the results.

- If you attach several tables, only the **first** one (marked *Charts*)
  is analysed.
- Analyses use the whole table, up to 300,000 rows.

!!! tip "Ask for statistics, not mental arithmetic"
    In normal chat the model only sees the first 100 rows of a table, so
    it cannot reliably answer questions such as "What is the average
    income?". Ask for a statistic or a chart instead ("Compute the mean
    income per region"), which uses the data tools on the complete table.

## Downloads

The panel offers downloads once the conversation has an answer:

- **The conversation (.md)**: the whole conversation as a Markdown text
  file, with the files attached to each message and any code.
- **With its interactive charts (.html)**: appears when an analysis has
  drawn charts. Contains the conversation together with the charts and
  opens in any web browser, without an internet connection.

## Data privacy and security

You may be attaching sensitive, unpublished or proprietary research data.
Text Lab is designed to keep it confidential.

- **Attached documents are read once and kept in memory.** Their text is
  kept in the memory of your session's Text Lab server while they stay
  attached, and is not written to your home directory. Files larger than
  1 MB pass through the session's temporary directory while they are
  uploaded (inside the session's private workspace), and are deleted
  when the upload ends.
- **Tables for data analysis are stored privately and temporarily.** So
  that the analysis tools can read a table, Chat saves a copy of each
  attached table, with the charts drawn from it, in your session's private
  workspace on the compute node. Only your user account can open it.
  They are deleted when you remove the table or delete the conversation,
  and in any case when your Text Lab session ends.
- **Conversations are kept for the session.** They live in the memory of
  your session's Text Lab server, so a reload does not lose them, and
  they are deleted when you delete them or when the session ends.
- **No AI training.** The models run on the UBELIX compute nodes via
  Ollama. They only read your data to answer your current question. **The
  models do not learn from your data**, and your data is never used to
  train or improve them.
- **Network isolation.** All data stays within the University of Bern's
  secure HPC network.

## Good to know

- **AI answers can be wrong.** Language models can produce inaccurate,
  misleading or invented statements that sound convincing. Check
  important facts, quotes and numbers against your sources.
- **No internet access.** The models cannot browse the web, and their
  knowledge ends at the date they were trained.
- **No memory between chats.** Each conversation starts from scratch. The
  model only knows what is in the current conversation and the attached
  files.

## Troubleshooting

??? question "'The language model server does not answer yet'"
    The service that runs the models is still starting, as it does at the
    beginning of a session. Wait a minute and send the message again. If
    the message stays, send `ollama.log` from the session's folder to
    support, or end your session and start a new one.

??? question "The first answer takes a long time"
    The model is being loaded into the GPU's memory, which takes a minute
    or two the first time you use it (the answer says *Loading ... into
    the GPU's memory*). Answers are fast afterwards. Choosing another
    model loads that model.

??? question "The answer waits for another job"
    Another tool, such as a transcription, is using the GPU. The answer
    starts when it ends; *Cancel that job* stops the other job instead.

??? question "'File ... exceeds the 10MB limit'"
    Shorten the document or attach only the relevant part. To analyse a
    large table, use [Visualize Data](visualize_data.md), which accepts
    much larger files.

??? question "'File ... contains no text, probably because it is scanned'"
    The PDF holds images of pages rather than text. Convert it with
    [OCR](ocr.md) and attach the result.

??? question "'File ... could not be read'"
    The file is damaged, password-protected, or (for text and JSON files)
    not saved as UTF-8. Open it, save it again (as UTF-8 for text), and
    attach it again.

??? question "My request for a chart or statistic was answered as normal chat"
    Check that your table is marked *Charts* above the message box. Then
    ask explicitly for the chart or test, for example "Make a histogram of
    age" or "Run a t-test of income by sex".

??? question "'No answer' with an error"
    The model server or the data agents reported an error, shown in the
    answer. Press **Try again**, or choose another model first. If the
    error keeps appearing, open *Details for a problem report*, copy them
    and send them to the Data Science Lab.

For help with your project, contact the Data Science Lab (DSL).
