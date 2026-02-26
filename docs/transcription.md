# Transcription Service

The Transcription tool uses **WhisperX** to convert spoken audio into written text with precise timestamps and speaker identification.

## Supported Formats
You can upload audio files in the following formats:

* `.wav`
* `.mp3`
* `.flac`
* `.m4a`

## How to Use

1. **Select Workflow:** Choose "Transcribe audio" to process a new file.
2. **Upload File:** Drag and drop your audio file or upload it.
3. **Language Selection:**
   * The system will attempt to auto-detect the language.
   * **Important:** You must verify or manually select the correct language from the dropdown.

### 🇨🇭 Swiss German Support

Text Lab includes a specialized fine-tuned model for **Swiss German**.

* Select **Swiss German** from the language list.
* This uses the `swhisper-large-1.1` model, which achieves a Word Error Rate (WER) of approximately **18**, significantly outperforming standard models on Swiss German dialects. More info about the fine-tuned whisper will be published soon.

### Configuration Options

* **VAD (Voice Activity Detection) (Optional):**
  * Check this to filter out silence before processing. 
  * *Max Pause:* Determines how much silence is allowed between words before splitting a segment.
* **Speaker Diarization (Who is speaking?):**
  * **Min/Max Speakers (Optional):** If you know how many people are in the recording, set these numbers to help the AI distinguish between voices (e.g., for an interview, set Min=2, Max=2). 

## Results & Export

Once the transcription is complete, you can:

* **Preview:** Listen to the audio with a synchronized interactive text player.
* **Edit View:** Toggle between "Segments" (sentences) and "Words" view.
* **Download:**
  * **Text (.txt):** Plain text transcript.
  * **CSV:** Contains timestamps and speaker labels.
  * **ZIP:** Download all formats and the processed WAV file in one package.

---

## 🔒 Data Privacy & Security

Audio recordings (such as interviews, meetings, or field notes) often contain highly sensitive personal data. Text Lab processes all audio using a strict **"Zero-Footprint"** architecture to ensure absolute confidentiality.

* **100% Local Processing:** Unlike commercial services (e.g., standard OpenAI Whisper APIs, Otter.ai, etc.), your audio files are **never** sent to the cloud. All transcription is performed locally on the University of Bern's secure UBELIX high-performance computing nodes.
* **Volatile Memory Storage:** When you upload an audio file, it is loaded into the server's temporary volatile memory (RAM). 
* **Secure Temporary Decoding:** If the app needs to convert complex formats (like `.m4a` or `.mp3`), it may briefly write a hidden temporary file to the node's secure storage. This file is **instantly and permanently deleted** the moment the decoding is finished—usually within milliseconds.
* **Ephemeral Sessions:** The generated transcripts and the audio waveforms loaded into the browser's playback tool are tied strictly to your active session. When you close the browser tab or your HPC job ends, **all audio and text data are instantly wiped.** Your data is never saved to your university home directory.
* **No AI Training:** The Whisper models only perform *inference* (listening and transcribing). They do not learn from your voice recordings, and your data is never used to train or improve the AI.