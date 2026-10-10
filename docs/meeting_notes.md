# Meeting Notes Generator

The Meeting Notes Generator turns a recording into written notes in one step. Upload the audio of a meeting, interview or lecture, and you get back a full transcript together with a structured summary: decisions and action items for a meeting, themes and quotes for an interview, or study notes for a lecture. If you already have a transcript, you can summarise it directly.

All processing runs on the university cluster. Your audio and text never leave the university network, and nothing is sent to external services such as OpenAI, Google or Anthropic.

## Quick start

1. Open the **From Audio** tab.
2. **Upload your recording** under *Audio file*.
3. **Choose the language** spoken in the recording, or keep *Auto-detect*.
4. **Choose a summary type**, for example *Meeting Notes*.
5. Click **Transcribe and Summarize**.
6. When it is done, read the summary and download the results.

A one-hour recording usually takes several minutes. The page shows each step as it happens.

## Choosing what to start from

The page has two tabs:

* **From Audio**: you have a recording. Text Lab transcribes it and then summarises the transcript.
* **From Transcript**: you already have the text, for example a transcript you made on the [Transcription](transcription.md) page or notes from another source. Only the summary is created, which is much faster.

## Summary types

The summary type decides what the notes focus on and how they are organised.

| Summary type | Best for | What you get |
|---|---|---|
| **General Summary** | Any recording | A short introduction, a bulleted list of key points and a brief conclusion |
| **Meeting Notes** | Team meetings, project meetings | Summary, key decisions, action items (with owner and deadline when they were mentioned), topics discussed and participants |
| **Interview Summary** | Interviews, focus groups, conversations | Overview, main themes, key statements and notable quotes, conclusions |
| **Academic Abstract** | Talks and presentations about research | Background and objective, methods, findings, conclusions and implications |
| **Lecture Notes** | Lectures and seminars | Lecture overview, topics covered, key concepts and definitions, important examples, summary |

You can try another type later without transcribing the recording again (see [Changing settings after a run](#changing-settings-after-a-run)).

## From Audio

### Settings

* **Audio file**: `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg` or `.webm`. One file at a time.
* **Language**: the language spoken in the recording.
    * *Auto-detect* listens to the first 30 seconds and picks the language. The detected language and how confident the detection was are shown as the first step. If the recording starts with silence, music or a different language, choose the language yourself.
    * *Swiss German* and *Swiss German (Flurin Turbo)* use models trained on Swiss German dialects. Auto-detect never picks these, so select one of them yourself for Swiss German recordings. See [Transcription](transcription.md#swiss-german-support) for details.
* **Summary type**: see [Summary types](#summary-types).
* **Summary language**:
    * *English* writes the summary in English, whatever language was spoken.
    * *Transcript language* writes the summary in the language of the recording.
* **Summarization model**: the language model that writes the summary. If you are unsure, keep the first one. Which models are offered depends on the GPU of your Text Lab session: in *Standard mode* large models are hidden, in *High-performance mode* all models are available.
* **Speaker diarization settings (optional)**: if you know how many people speak in the recording, tick *Set minimum speakers* and *Set maximum speakers* and enter the numbers. This helps Text Lab tell the voices apart. For an interview with two people, set both to 2.

!!! note "GPU memory in use"
    If you used another Text Lab feature earlier in the session, a warning may say that models are still loaded in GPU memory. Click **Clear GPU Memory** before you start, so the transcription has enough memory.

### Running

Click **Transcribe and Summarize**. The page then shows each step:

1. **Detecting language** (only with *Auto-detect*).
2. **Running transcription pipeline**: the audio is converted, transcribed, aligned word by word, and split by speaker.
3. **Generating summary**: the first run loads the language model into GPU memory, which can take 1–2 minutes.

To cancel a run, reload the page.

### Naming the speakers

The transcript labels the voices it finds as `SPEAKER_00`, `SPEAKER_01` and so on. To replace these labels with real names or roles:

1. Once the transcription is done, open **Name the speakers (optional)** below the **Transcribe and Summarize** button.
2. Type a name for each speaker, for example *Interviewer* and *Participant*, or the people's names.
3. Click **Re-summarize with current settings**.

The first summary is created right after the transcription, before you can enter names. It still uses the `SPEAKER_00` labels. Clicking **Re-summarize** creates a new summary and a new transcript download that use your names. The recording is not transcribed again.

!!! tip "Finding out who is who"
    Open the **Full Transcript** tab and read the first few lines of each speaker to work out who they are. Speaker separation is automatic and can make mistakes, especially with similar voices, people talking over each other, or poor audio.

### Changing settings after a run

After a run you can change the **summary type**, **summary language** or **model** and click **Re-summarize with current settings**. Only the summary is created again, which is much faster than the first run. A notice reminds you when the summary shown no longer matches your current settings.

If you upload a different file or change the **language**, the recording has to be transcribed again: click **Transcribe and Summarize**.

### If the summary fails

If something goes wrong while the summary is written, the page shows the error, and the transcript stays available to read and download. Click **Retry summarization** to try again without transcribing the recording again. If it keeps failing, try a different model, or click **Clear GPU Memory** first. Open **Error details** to see the full message if you need to report the problem.

## From Transcript

1. Open the **From Transcript** tab.
2. Choose an **Input method**:
    * **Upload transcript CSV (from Transcribe page)**: the CSV download from the [Transcription](transcription.md) page. Speaker labels are kept, and you can name the speakers under *Name the speakers (optional)*.
    * **Upload text file**: a plain `.txt` file.
    * **Paste text**: paste the transcript into the text box.
3. Check the word count shown below the input to make sure the whole text was read.
4. Choose the **summary type**, **model** and **summary language**.
5. Click **Generate Summary**.

The summary stays on the page while you download files. If you then change the text or the settings, a notice reminds you to click **Generate Summary** again.

!!! tip "Better summaries from pasted text"
    If your text has speaker names at the start of each line (for example `Anna: I think we should...`), the summary can say who said what and who owns each action item.

## Results and downloads

At the top, the results show the summary type, the number of words in the transcript and, for audio, the length of the recording. Two tabs show the **Summary** and the **Full Transcript**.

Downloads:

* **Summary (.md)**: the notes only.
* **Transcript (.txt)**: the transcript, one line per segment, with speaker names if you set them.
* **Full document (.md)**: the summary and the full transcript in one file, with the file name and summary type at the top.

From Audio also offers **Download raw transcription files**:

* **Transcript CSV (WhisperX)**: each segment with start and end time and speaker. You can upload this file again later on the *From Transcript* tab.
* **Plain transcript (.txt)**: the text only, without speaker labels.

`.md` files are Markdown: plain text with simple formatting such as `**bold**` and `- lists`. They open in any text editor, and Word, Obsidian, Notion and most note apps can import them.

Results are kept only while the page is open. Download everything you want to keep before you reload the page, close the browser tab or end your Text Lab session.

## Long recordings

Recordings of several hours are supported. When a transcript is too long for the model to read at once (roughly 50,000 words), it is split into parts. Notes are taken for each part, and then the notes are combined into one summary. You see the progress as *Analyzing part 2 of 5...* and then *Synthesizing final summary...*. This takes longer than a single pass, and small details are more likely to be left out.

## Checking the results

Summaries are written by a language model, and transcripts by speech recognition. Both can be wrong. Before you share or act on the notes:

* **Check names, numbers and dates** against the transcript. Speech recognition often mishears names and technical terms.
* **Check action items and decisions.** The model may miss one, or give it to the wrong person, especially when speaker labels are wrong.
* **Use the Full Transcript tab** to look up anything in the summary that seems surprising.

The model is told to use only what is in the transcript and not to add information. It can still make mistakes.

## Data privacy and security

Recordings of meetings and interviews often contain personal or confidential information. This is what happens to your data:

* **Everything runs on the university cluster.** Transcription (WhisperX) and summarisation (language models served by Ollama) run on the UBELIX compute node of your Text Lab session. No audio or text is sent to an external service.
* **Temporary files are deleted after transcription.** For transcription, your audio is written to a temporary folder in your session's private workspace on the compute node, together with the progress and result files. The folder is deleted as soon as the transcription ends, whether it succeeded or failed, and the whole workspace is deleted when your session ends.
* **Results live only in your browser session.** The transcript and summary are kept in your open page only. They are not saved to your home directory, and they are gone when you reload the page, close the tab or your Text Lab session ends.
* **No AI training.** The models only process your recording. They do not learn from it, and your data is never used to train or improve them.
