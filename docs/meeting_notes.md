# Meeting Notes Generator

The Meeting Notes Generator combines transcription and summarization into a single automated workflow. It allows you to quickly convert audio recordings or existing transcripts into structured meeting notes and summaries.

## Workflows

The tool offers two main workflows, accessible via separate tabs:

### 1. From Audio

This is the full pipeline. You can upload an audio file, and the tool will first transcribe it and then automatically generate a structured summary.

*   Upload your audio file.
*   The system transcribes the audio using the transcription pipeline.
*   Once transcribed, a Large Language Model (LLM) summarizes the content.
*   You receive both the full transcript and the generated meeting notes.

### 2. From Transcript

If you already have a transcript, you can upload it directly to generate a summary.

*   Upload an existing transcript file (CSV or plain text).
*   The LLM analyzes the text and produces structured meeting notes.

## Output

The final output includes a structured summary highlighting key points, decisions, and action items from the meeting, along with the full text transcript for reference.
