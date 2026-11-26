# Transcription

!!! note "Draft for the new interface"
    This guide describes the Transcription page of the new Text Lab
    interface. It replaces [the current guide](../transcription.md) when
    the new interface goes live.

Transcription turns recordings into text with **WhisperX**: every word gets
its time in the recording, and the speakers are told apart. The page has
three uses, chosen at the top of the form:

- **One recording**: transcribe it, then listen to it beside its transcript.
- **A ZIP of recordings**: transcribe many recordings with the same
  settings, and download all transcripts at once.
- **Review a transcript**: listen to a recording beside a transcript Text
  Lab made earlier, for example to check names and difficult passages.

Transcriptions run in the background, one at a time, so they never compete
for the GPU. You can leave the page, use other tools, and come back: the
results stay until you remove them or the session ends. The status bar at
the top shows how many jobs are running.

## Formats

Recordings: WAV, MP3, FLAC, M4A, OGG, OPUS and WEBM. For a video, save its
sound as an audio file first.

A batch is a ZIP file of recordings. Folders inside it are fine; other
files in it are ignored.

## One recording

1. Choose **One recording** and drop the file on the form, or click to
   choose it.
2. Choose the **language**, or leave **Detect from the recording**. Text
   Lab then listens to the first 30 seconds and picks the language itself.
   It hears Swiss German as German: for Swiss German, choose it yourself.
3. Set the **options** if you need them (see below), and press
   **Transcribe**.

The recording is uploaded first; a bar shows how far. The job then appears
under **Your transcriptions** with its progress. When it is done, its
result opens in the same place.

### The result

- **Listen and read**: the recording with its waveform and the transcript
  beside it. The passage being spoken is marked, and so is the word.
  Click a passage's time or any word to play from there, or click the
  waveform. With the keyboard, the waveform moves 5 seconds with the arrow
  keys and 30 seconds with Page Up and Page Down; Space plays and pauses.
  **Words** shows each word with its time, to check the timing.
- **Text**: the transcript as plain text, with the speakers.
- **Models**: the models the transcript was made with, to cite them.

### Downloads

| File | What it is |
|---|---|
| `_text.txt` | The plain text, one line per passage |
| `.srt`, `.vtt` | Subtitles, for video editors, VLC or YouTube |
| `_transcription.csv` | The passages with their times and speakers |
| `_words.csv` | Every word with its time and speaker |
| `_ELAN_compatible.tsv` | The passages, for import into ELAN |
| `_outputs.zip` | All of the above, and the recording as a WAV file for ELAN |

The tables are tab-separated, also those named `.csv`.

## A ZIP of recordings

Choose **A ZIP of recordings**, drop the ZIP file, and choose the language
and options as for one recording: they apply to every recording. With
**Detect from the recording**, the language is detected for each recording
on its own.

The result lists the recordings with their language, length and number of
speakers, and recordings that could not be read. Its ZIP holds a folder per
recording with the same files as for one recording (without the WAV).

## Review a transcript

Choose **Review a transcript**, and give the recording and its transcript:
the `_words.csv` file of a Text Lab transcription marks each word as it is
spoken; `_transcription.csv` or the ELAN table mark the passages. Nothing
is transcribed again, and the review does not wait for transcriptions that
are running.

Instead of uploading the files, you can give their paths on the cluster
under **Use files on the cluster instead**, such as
`/storage/research/...`. They are read where they are, not copied.

## Options

**Model.** The model follows from the language: Whisper `large-v3-turbo`
for most languages. Under **Use another model** you can name another
Whisper model, or the folder of one.

**Swiss German.** Text Lab has two models for Swiss German:

- **Swiss German** uses `swhisper-large-1.1`, fine-tuned for Swiss German
  dialects, with a word error rate of about 18 %.
- **Swiss German (Flurin Turbo)** uses
  [Flurin17/whisper-large-v3-turbo-swiss-german](https://huggingface.co/Flurin17/whisper-large-v3-turbo-swiss-german).

**Skip silence first.** Voice activity detection finds the speech, and only
that is transcribed. This helps with long pauses and background noise.
Speech is joined across pauses up to the length you set (0.25 seconds by
default).

**Speakers.** Each passage is marked with who speaks (Speaker 1, Speaker 2,
...), with pyannote's *speaker-diarization-community-1* model. If you know
how many people speak, give the least and the most number: it makes the
result more reliable. For an interview with two people, give 2 and 2.

## Your data

- Everything runs on the University's computing cluster, UBELIX. Your
  recordings are never sent to outside services, and the models do not
  learn from them.
- Uploads and results are kept in your session's private folder on the
  compute node, which only your account can open. Uploads are deleted when
  their job ends; results when you remove them or the session ends.
- The player plays a copy of the recording that Text Lab makes in the same
  private folder; it is deleted with its job.
