# Transcription

Transcribes recordings with WhisperX: speech to text with word-level
timestamps, optional voice activity detection (VAD), and speaker
identification (diarization). Swiss German has its own models. Used by the
Transcribe page and the Meeting Notes Generator.

## Pipeline

One pipeline serves every caller (`service.transcribe_files`):

1. **Decode** each recording with ffmpeg to 16 kHz mono samples
   (`audio.load_audio`). Files that cannot be decoded are skipped and
   reported.
2. **Detect the language** with Whisper "tiny" if none was chosen
   (`audio.detect_language`); English if detection gives no answer.
3. **Transcribe** with WhisperX. With VAD, only the speech Silero finds is
   transcribed, and segment times are shifted back to the recording's
   timeline.
4. **Align** words to the audio. Swiss German is aligned as German; the
   alignment model is loaded once per language.
5. **Diarize** when `TEXT_LAB_HF_TOKEN_FILE` points at a Hugging Face token;
   otherwise this step is skipped with a note.

The Whisper and diarization models are loaded once per run, so a ZIP batch
does not reload them for every file.

Interfaces never run the pipeline in their own process. They call
`service.run_transcription`, which starts `worker.py` through
`textlab.common.jobs`; when the worker exits, all GPU memory is released for
the next feature (for example Ollama in Meeting Notes).

## Public API

| Name | Module | Purpose |
|---|---|---|
| `run_transcription(files, options, on_progress=, cancel=)` | `service` | Run the pipeline in a worker process |
| `transcribe_files(...)` | `service` | The pipeline itself, in the calling process |
| `staged_uploads(uploads)`, `staged_zip(archive)` | `service` | Write uploads to the workspace for the worker; removed afterwards |
| `TranscriptionOptions`, `AudioFile`, `Transcript`, `TranscriptionResult` | `models` | Inputs and outputs; convert to and from JSON |
| `default_model(language)`, `alignment_language(language)`, `converted_model_missing(...)` | `whisper_models` | Model choice per language |
| `transcript_files(transcript, base_name)`, `batch_zip(...)`, `load_transcript_items(...)` | `formats` | Exports (segment, word and ELAN tables, SRT, VTT, text) and readers |
| `convert_audio_to_wav_bytes(...)`, `detect_language_from_bytes(...)` | `audio` | Decoding and language detection for the pages |

A minimal use:

```python
from textlab.features.transcription.models import TranscriptionOptions
from textlab.features.transcription.service import (
    run_transcription,
    staged_uploads,
)

options = TranscriptionOptions(model="large-v3-turbo", language="de")
with staged_uploads([("talk.mp3", audio_bytes)]) as files:
    result = run_transcription(files, options, on_progress=print)
transcript = result.transcripts[0]
```

The browser player (`ui/streamlit/components/audio_player.py`) and the
Streamlit status box (`ui/streamlit/components/progress.py`) belong to the
user interface, not to this package.

## Files written

| What | Where | Removed |
|---|---|---|
| Uploaded recordings, extracted ZIP members | `transcription/upload-*`, `transcription/batch-*` in the job workspace | When the run ends |
| Worker request, progress and result | `transcription/job-*` in the job workspace | When the run ends |
| ffmpeg's temporary copy of hard-to-pipe formats | `$TMPDIR` (inside the workspace) | Immediately after decoding |

Nothing is written outside the workspace; results are returned to the page,
which offers them as downloads.

## Configuration

| Setting | Used for |
|---|---|
| `TEXT_LAB_HF_TOKEN_FILE` | Hugging Face token for diarization; optional |
| `TEXT_LAB_CUSTOM_WHISPER_DIR` | Folder with `swhisper-large-1.1` and `flurin-swiss-german-turbo-ct2`; needed only for Swiss German |

The standard Whisper, alignment and diarization models come from the model
stores the launch script mounts (`HF_HOME`, `/opt/whisper`, `/opt/torch`).

## Tests

In `tests/`:

- `test_formats.py`: exact output of every export, and the readers.
- `test_service.py`: the pipeline with WhisperX, torch and ffmpeg faked:
  language handling, model loading, diarization, VAD offsets, skipped
  files, progress and cancellation.
- `test_staging.py`: uploads and ZIP batches in the workspace.
- `test_models_and_settings.py`: JSON round trips and model choice.
- `test_integration.py` (`container`, `slow`): the real pipeline in a
  worker on generated audio; skipped when the model store is not mounted.

## Batch use (planned)

`cli.py` is a placeholder for a `textlab transcribe` command for Slurm batch
jobs. It will call `transcribe_files` directly, since a batch job already
runs in its own process.
