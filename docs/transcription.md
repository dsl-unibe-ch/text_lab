# Transcription Service

The Transcription tool uses **WhisperX** to convert spoken audio into written text with precise timestamps and speaker identification.

## Supported Formats
You can upload audio files in the following formats:
* `.wav`
* `.mp3`
* `.flac`
* `.m4a`

## How to Use

1.  **Select Workflow:** Choose "Transcribe audio" to process a new file.
2.  **Upload File:** Drag and drop your audio file.
3.  **Language Selection:** * The system will attempt to auto-detect the language.
    * **Important:** You must verify or manually select the correct language from the dropdown.

### 🇨🇭 Swiss German Support
Text Lab includes a specialized fine-tuned model for **Swiss German**. 
* Select **Swiss German** from the language list.
* This uses the `swhisper-large-1.1` model, which achieves a Word Error Rate (WER) of approximately **18%**, significantly outperforming standard models on Swiss German dialects.

### Configuration Options

* **VAD (Voice Activity Detection):** * Check this to filter out silence before processing. 
    * *Max Pause:* Determines how much silence is allowed between words before splitting a segment.
* **Speaker Diarization (Who is speaking?):**
    * **Min/Max Speakers:** If you know how many people are in the recording, set these numbers to help the AI distinguish between voices (e.g., for an interview, set Min=2, Max=2).

## Results & Export

Once the transcription is complete, you can:
* **Preview:** Listen to the audio with a synchronized interactive text player.
* **Edit View:** Toggle between "Segments" (sentences) and "Words" view.
* **Download:** * **Text (.txt):** Plain text transcript.
    * **CSV:** Contains timestamps and speaker labels.
    * **ZIP:** Download all formats and the processed WAV file in one package.
