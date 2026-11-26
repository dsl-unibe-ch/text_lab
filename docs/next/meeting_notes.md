# Meeting Notes

!!! note "Draft for the new interface"
    This guide describes the Meeting Notes page of the new Text Lab
    interface. It replaces [the current guide](../meeting_notes.md) when
    the new interface goes live.

Meeting Notes turns a recording into written notes: upload the audio of a
meeting, interview or lecture, and you get its transcript and a structured
summary, such as decisions and action items for a meeting, themes and
quotes for an interview, or study notes for a lecture. Every point in the
summary shows which passages it comes from, so you can check it. If you
already have a transcript, you can summarize it directly.

Everything runs on the University's computing cluster. Your recordings and
texts are never sent to outside services such as OpenAI, Google or
Anthropic.

## Quick start

1. Keep **From a recording** and drop the recording on the form.
2. Choose the **language**, or leave **Detect from the recording**.
3. Choose a **summary type**, for example *Meeting Notes*.
4. Press **Transcribe and summarize**.

The job runs in the background; you can leave the page and come back. It
first transcribes the recording, then writes the summary, which appears on
the job's record as it is written. When it is done, read the summary and
click the times after its points to listen to the passages they come from.

## From a recording

- **Recording**: WAV, MP3, FLAC, M4A, OGG, OPUS or WEBM.
- **Language**: the language spoken. *Detect from the recording* listens to
  the first 30 seconds. It hears Swiss German as German: for Swiss German,
  choose *Swiss German* or *Swiss German (Flurin Turbo)* yourself (see
  [Transcription](transcription.md)).
- **Number of speakers**: if you know how many people speak, give the least
  and the most number. For an interview with two people, give 2 and 2.

The transcription is the same as on the Transcription page. Then the
language model is loaded, which can take a minute or two the first time.

## From a transcript

Choose **From a transcript**, then drop a file or paste the text:

- The `_transcription.csv` file of a [Transcription](transcription.md):
  the summary then knows the speakers and cites the passages by their time.
- A text file (`.txt`), or text pasted under **Paste the text instead**.
  Lines that start with `SPEAKER_00:` keep their speaker. The summary
  cites the lines of the text, so keep one statement or speaker turn per
  line; very long lines are split into sentences.

## The summary

| Type | Best for | What you get |
|---|---|---|
| **General Summary** | Any recording | A short introduction, the key points, a brief conclusion |
| **Meeting Notes** | Team and project meetings | Summary, decisions, action items with owner and deadline when mentioned, topics, participants |
| **Interview Summary** | Interviews, focus groups | Overview, main themes, key statements and quotes, conclusions |
| **Academic Abstract** | Talks about research | Background and objective, methods, findings, conclusions |
| **Lecture Notes** | Lectures and seminars | Overview, topics, key concepts and definitions, examples, summary |

**Language model**: the first one is a good start. Larger models write
better summaries and take longer. Which models are offered depends on the
session's GPU.

**Language of the summary**: *English*, or *As spoken*, the language of the
recording or transcript.

## The result

- **Summary**: the notes. After each point, the passages it comes from:
  their time in the recording, or their line number. With a recording,
  clicking a time plays the passage; without one, it shows the line in the
  transcript. A point marked *no source* cites nothing: check those first.
- **Sources**: the passages the summary cites, with their speaker.
- **Transcript**: the whole transcript. With a recording, it is shown with
  the player, as on the Transcription page: the passage and the word being
  spoken are marked.

Speakers are called *Speaker 1*, *Speaker 2* and so on, in the order they
first speak, as on the Transcription page.

### Downloads

| File | What it is |
|---|---|
| `_summary_….md` | The notes; the sources are footnotes that quote the passages with their time and speaker |
| `_notes_….md` | The notes with their footnotes and the whole transcript, in one file |
| `_transcript.txt` | The transcript, one passage per line, with its time or line number and its speaker |
| `_transcription.csv` | For a recording: the transcript table, to summarize again later from a transcript |

### Summarizing again

Under **Summarize again**, below a result, you can give the speakers their
names (*Interviewer*, *Participant*, or people's names), and choose another
summary type, model or language. A new result is made from the same
transcript: the recording is not transcribed again. The summary, its
sources and the transcript then use the names.

!!! tip "Finding out who is who"
    Read or listen to the first passages of each speaker in the
    **Transcript** tab. Telling speakers apart is automatic and can be
    wrong, especially with similar voices, people talking over each other,
    or poor audio.

### If the summary fails

The result then shows the error, and the transcript is kept: summarize it
again, perhaps with another model. If it keeps failing, send the error's
details (under *Details for a problem report*) to support.

To stop a job, press **Cancel** on its record.

## Your data

- Everything runs on UBELIX, the University's computing cluster; the
  language models run in your session. The models do not learn from your
  recordings or texts.
- Uploads and results are kept in your session's private folder, which only
  your account can open. Uploads are deleted when their job ends, results
  when you remove them or the session ends.
