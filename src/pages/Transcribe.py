import os
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

import base64
import csv
import html
import io
import json
import mimetypes
import sys
import tempfile
import uuid
import zipfile
import whisperx
import torch
import subprocess
import numpy as np
import soundfile as sf

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from auth import check_token
from utils import generate_transcription_csv, generate_words_csv, get_vad_segments, read_hf_token

# Language mapping from display names to codes
LANGUAGE_MAPPING = {
    "English": "en",
    "German": "de",
    "French": "fr",
    "Spanish": "es",
    "Italian": "it",
    "Swiss German": "ch_de",
    "Japanese": "ja",
    "Chinese": "zh",
    "Dutch": "nl",
    "Ukrainian": "uk",
    "Portuguese": "pt",
    "Arabic": "ar",
    "Czech": "cs",
    "Russian": "ru",
    "Polish": "pl",
    "Hungarian": "hu",
    "Finnish": "fi",
    "Persian": "fa",
    "Greek": "el",
    "Turkish": "tr",
    "Danish": "da",
    "Hebrew": "he",
    "Vietnamese": "vi",
    "Korean": "ko",
    "Urdu": "ur",
    "Telugu": "te",
    "Hindi": "hi",
    "Catalan": "ca",
    "Malayalam": "ml",
    "Norwegian Bokmål": "no",
    "Norwegian Nynorsk": "nn",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Croatian": "hr",
    "Romanian": "ro",
    "Basque": "eu",
    "Galician": "gl",
    "Georgian": "ka",
    "Latvian": "lv",
    "Tagalog": "tl",
    "Swedish": "sv"
}
LANGUAGE_CODE_TO_NAME = {code: name for name, code in LANGUAGE_MAPPING.items()}

WAVESURFER_MAX_BYTES = 75 * 1024 * 1024
WAVESURFER_MAX_SECONDS = 30 * 60


def parse_timestamp(value):
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    parts = text.split(":")
    try:
        if len(parts) == 3:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
        elif len(parts) == 2:
            hours = 0.0
            minutes = float(parts[0])
            seconds = float(parts[1])
        else:
            return float(text)
    except ValueError:
        return 0.0
    return hours * 3600.0 + minutes * 60.0 + seconds


def load_tsv_rows(text_data):
    reader = csv.DictReader(io.StringIO(text_data), delimiter="\t")
    return list(reader)


def load_transcript_items(text_data, mode):
    rows = load_tsv_rows(text_data)
    items = []
    
    # Auto-detect if CSV has speaker information
    has_speakers = False
    if rows and ('Speaker' in rows[0] or 'speaker' in rows[0]):
        speaker_value = rows[0].get('Speaker') or rows[0].get('speaker')
        if speaker_value and speaker_value.strip():
            has_speakers = True
    
    for row in rows:
        start = parse_timestamp(row.get("Start"))
        end = parse_timestamp(row.get("End"))
        if mode == "words":
            text = row.get("Word") or row.get("word") or row.get("Text") or ""
            items.append({"start": start, "end": end, "text": text})
        else:  # segments mode
            text = row.get("Text") or row.get("text") or row.get("Word") or ""
            if has_speakers:
                speaker = row.get("Speaker") or row.get("speaker") or "UNKNOWN"
                items.append({"start": start, "end": end, "text": text, "speaker": speaker})
            else:
                items.append({"start": start, "end": end, "text": text})
    
    return items, has_speakers


def transcription_text_from_csv(transcription_csv):
    rows = load_tsv_rows(transcription_csv)
    lines = []
    for row in rows:
        text = row.get("Text") or row.get("text") or ""
        text = str(text).strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def audio_bytes_from_path(path):
    with open(path, "rb") as f:
        return f.read()

def format_duration(seconds):
    total = int(round(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def decode_audio_bytes(audio_bytes, sr=16000):
    """
    Decode audio bytes to mono float32 waveform at the target sample rate.
    Uses ffmpeg on stdin first, then falls back to a temp file when
    container demuxing from pipe returns empty audio (common with some M4A/MP4 files).
    """
    output_args = [
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sr),
        "pipe:1",
    ]

    cmd_pipe = ["ffmpeg", "-nostdin", "-threads", "0", "-i", "pipe:0", *output_args]
    pipe_proc = subprocess.run(cmd_pipe, input=audio_bytes, capture_output=True, check=False)
    audio = np.frombuffer(pipe_proc.stdout, np.int16).flatten().astype(np.float32) / 32768.0
    if pipe_proc.returncode == 0 and audio.size > 0:
        return audio

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        cmd_file = ["ffmpeg", "-nostdin", "-threads", "0", "-i", tmp_path, *output_args]
        file_proc = subprocess.run(cmd_file, capture_output=True, check=False)
        audio = np.frombuffer(file_proc.stdout, np.int16).flatten().astype(np.float32) / 32768.0
        if file_proc.returncode == 0 and audio.size > 0:
            return audio

        pipe_stderr = pipe_proc.stderr.decode(errors="replace")
        file_stderr = file_proc.stderr.decode(errors="replace")
        raise RuntimeError(
            "Failed to decode audio from both stdin and temp file.\n"
            f"stdin rc={pipe_proc.returncode}, samples={int(np.frombuffer(pipe_proc.stdout, np.int16).size)}\n"
            f"file rc={file_proc.returncode}, samples={int(np.frombuffer(file_proc.stdout, np.int16).size)}\n"
            f"stdin stderr:\n{pipe_stderr}\n\nfile stderr:\n{file_stderr}"
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def audio_debug_stats(waveform):
    if waveform is None:
        return "waveform=None"
    shape = getattr(waveform, "shape", None)
    dtype = getattr(waveform, "dtype", None)
    size = int(getattr(waveform, "size", 0))
    if size > 0:
        min_val = float(np.min(waveform))
        max_val = float(np.max(waveform))
        return f"shape={shape}, size={size}, dtype={dtype}, min={min_val:.6f}, max={max_val:.6f}"
    return f"shape={shape}, size={size}, dtype={dtype}, min=NA, max=NA"


def convert_audio_to_wav_bytes(audio_bytes, original_filename, sr=16000):
    """
    Convert uploaded audio to WAV bytes to avoid MP3 encoder delay issues.
    Uses 16kHz mono to keep size manageable for browser playback.
    Returns WAV bytes, WAV filename, and decoded waveform for WhisperX.
    """
    audio = decode_audio_bytes(audio_bytes, sr=sr)
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, audio, sr, format="WAV", subtype="PCM_16")
    wav_bytes = wav_buffer.getvalue()
    wav_filename = os.path.splitext(original_filename)[0] + ".wav"
    return wav_bytes, wav_filename, audio


@st.cache_resource(show_spinner=False)
def get_language_detector():
    from faster_whisper import WhisperModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel("tiny", device=device, compute_type=compute_type)
    return model


def detect_language_from_audio_bytes(audio_bytes, sr=16000):
    audio = decode_audio_bytes(audio_bytes, sr=sr)
    if audio is None or audio.size == 0:
        raise ValueError("Decoded waveform is empty.")

    max_seconds = 30
    snippet = audio[: int(sr * max_seconds)]
    detector = get_language_detector()
    _, info = detector.transcribe(
        snippet,
        beam_size=1,
        language=None,
        task="transcribe",
        vad_filter=False,
        without_timestamps=True,
    )
    language_code = getattr(info, "language", None)
    language_prob = float(getattr(info, "language_probability", 0.0) or 0.0)
    return language_code, language_prob


def create_wavesurfer_preview(wav_bytes, audio, sr):
    duration_seconds = len(audio) / float(sr) if audio is not None else 0.0
    needs_preview = False
    if wav_bytes and len(wav_bytes) > WAVESURFER_MAX_BYTES:
        needs_preview = True
    if duration_seconds > WAVESURFER_MAX_SECONDS:
        needs_preview = True

    if not needs_preview or audio is None:
        return wav_bytes, None

    max_samples_by_seconds = int(WAVESURFER_MAX_SECONDS * sr)
    max_samples_by_bytes = max(int((WAVESURFER_MAX_BYTES - 44) / 2), 1)
    preview_samples = min(len(audio), max_samples_by_seconds, max_samples_by_bytes)

    if preview_samples >= len(audio):
        return wav_bytes, None

    preview_audio = audio[:preview_samples]
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, preview_audio, sr, format="WAV", subtype="PCM_16")
    preview_bytes = wav_buffer.getvalue()
    return preview_bytes, preview_samples / float(sr)

def audio_data_url(audio_path_or_bytes, path_hint):
    """
    Create data URL for audio. Handles both file paths (string) and bytes.
    If a file path is provided, reads the file to avoid message size limits.
    """
    if isinstance(audio_path_or_bytes, str):
        # It's a file path - read it
        with open(audio_path_or_bytes, 'rb') as f:
            audio_bytes = f.read()
    else:
        # It's already bytes
        audio_bytes = audio_path_or_bytes
    
    mime_type, _ = mimetypes.guess_type(path_hint)
    if not mime_type:
        mime_type = "audio/wav"
    b64 = base64.b64encode(audio_bytes).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


def build_player_html(audio_url, items, mode):
    component_id = f"ws_{uuid.uuid4().hex}"
    items_json = json.dumps(items)
    safe_audio_url = html.escape(audio_url, quote=True)
    safe_mode = html.escape(mode, quote=True)

    return f"""
    <div class="ws-container" id="{component_id}_container">
      <div id="{component_id}_waveform"></div>
      <div class="ws-controls">
        <button id="{component_id}_toggle">Play / Pause</button>
        <span id="{component_id}_time" class="ws-time">00:00:000</span>
      </div>
      <div id="{component_id}_transcript" class="ws-transcript"></div>
    </div>

    <script src="https://unpkg.com/wavesurfer.js@7"></script>
    <script>
      const audioUrl = "{safe_audio_url}";
      const items = {items_json};
      const mode = "{safe_mode}";
      const waveformId = "{component_id}_waveform";
      const transcriptId = "{component_id}_transcript";
      const toggleId = "{component_id}_toggle";
      const timeId = "{component_id}_time";

      const ws = WaveSurfer.create({{
        container: "#" + waveformId,
        waveColor: "#c9c9c9",
        progressColor: "#2d2d2d",
        cursorColor: "#e76f51",
        height: 120,
        barWidth: 2,
        barGap: 2,
        normalize: true
      }});

      const transcriptEl = document.getElementById(transcriptId);
      const timeEl = document.getElementById(timeId);
      const toggleEl = document.getElementById(toggleId);
      let activeIndex = -1;

      function formatTime(seconds) {{
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        const millis = Math.floor((seconds - Math.floor(seconds)) * 1000);
        return (
          String(mins).padStart(2, "0") +
          ":" +
          String(secs).padStart(2, "0") +
          ":" +
          String(millis).padStart(3, "0")
        );
      }}

      function renderTranscript() {{
        transcriptEl.innerHTML = "";
        items.forEach((item, idx) => {{
          const el = document.createElement(mode === "word" ? "span" : "div");
          el.className = mode === "word" ? "ws-word" : "ws-seg";
          el.dataset.start = item.start;
          el.dataset.end = item.end;
          if (mode === "word") {{
            el.textContent = item.text + " ";
          }} else {{
            const header = document.createElement("div");
            header.className = "ws-meta";
            const label = item.speaker ? item.speaker + " " : "";
            header.textContent = label + formatTime(item.start) + " - " + formatTime(item.end);
            const text = document.createElement("div");
            text.className = "ws-text";
            text.textContent = item.text;
            el.appendChild(header);
            el.appendChild(text);
          }}
          el.addEventListener("click", () => {{
            const duration = ws.getDuration() || 1;
            ws.seekTo(item.start / duration);
          }});
          transcriptEl.appendChild(el);
        }});
      }}

      function updateActive(currentTime) {{
        if (!items.length) {{
          return;
        }}
        let idx = -1;
        for (let i = 0; i < items.length; i++) {{
          if (currentTime >= items[i].start && currentTime <= items[i].end) {{
            idx = i;
            break;
          }}
        }}
        if (idx === activeIndex) {{
          return;
        }}
        if (activeIndex >= 0) {{
          const prev = transcriptEl.children[activeIndex];
          if (prev) {{
            prev.classList.remove("ws-active");
          }}
        }}
        if (idx >= 0) {{
          const next = transcriptEl.children[idx];
          if (next) {{
            next.classList.add("ws-active");
            next.scrollIntoView({{block: "nearest"}});
          }}
        }}
        activeIndex = idx;
      }}

      function syncTime() {{
        const t = ws.getCurrentTime();
        timeEl.textContent = formatTime(t);
        updateActive(t);
      }}

      ws.on("ready", () => {{
        renderTranscript();
        syncTime();
      }});
      ws.on("timeupdate", syncTime);
      ws.on("audioprocess", syncTime);
      ws.on("seek", syncTime);

      toggleEl.addEventListener("click", () => {{
        ws.playPause();
      }});

      ws.load(audioUrl);
    </script>

    <style>
      .ws-container {{
        font-family: Arial, sans-serif;
      }}
      .ws-controls {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 8px 0 16px 0;
        color: #ffffff;
      }}
      .ws-time {{
        font-variant-numeric: tabular-nums;
        background: #1b1b1b;
        border: 1px solid #3a3a3a;
        padding: 4px 8px;
        border-radius: 6px;
      }}
      .ws-transcript {{
        border: 1px solid #e0e0e0;
        padding: 12px;
        max-height: 360px;
        overflow-y: auto;
        background: #fafafa;
        line-height: 1.5;
      }}
      .ws-seg {{
        padding: 6px 8px;
        margin-bottom: 6px;
        border-radius: 6px;
        cursor: pointer;
      }}
      .ws-meta {{
        font-size: 12px;
        color: #666666;
        margin-bottom: 4px;
      }}
      .ws-text {{
        font-size: 14px;
      }}
      .ws-word {{
        cursor: pointer;
        padding: 2px 2px;
        border-radius: 4px;
      }}
      .ws-active {{
        background: #ffe8d6;
      }}
      button {{
        border: 1px solid #333333;
        background: #ffffff;
        padding: 6px 10px;
        cursor: pointer;
      }}
    </style>
    """


def main():
    st.set_page_config(page_title="Transcribe", layout="wide")
    check_token()
    st.title("Transcribe")
    
    # Mode selection
    workflow_mode = st.radio(
        "Workflow",
        ["Transcribe audio", "Upload existing transcription"],
        index=0,
        horizontal=True,
        help="Choose to transcribe new audio or review existing transcription files"
    )
    
    st.divider()
    
    if workflow_mode == "Upload existing transcription":
        # Upload mode - load existing files
        st.write("Load audio and a WhisperX CSV/TSV to review alignment with playback.")
        
        col1, col2 = st.columns(2)
        with col1:
            audio_path = st.text_input("Audio file path (server-side)", value="", key="upload_audio_path")
            audio_upload = st.file_uploader("Or upload audio", type=["wav", "mp3", "flac", "m4a"], key="upload_audio")
        with col2:
            transcript_path = st.text_input("Transcript CSV/TSV path (server-side)", value="", key="upload_transcript_path")
            transcript_upload = st.file_uploader("Or upload CSV/TSV", type=["csv", "tsv"], key="upload_transcript")

        display_mode = st.radio(
            "Display transcript as",
            ["segments", "words"],
            index=0,
            horizontal=True,
            help="Segments = sentence-level view, Words = word-by-word view",
            key="upload_mode"
        )

        audio_bytes = None
        audio_label = None
        audio_wave = None
        wav_bytes = None
        wav_filename = None
        if audio_upload is not None:
            audio_bytes = audio_upload.read()
            audio_label = audio_upload.name
            # Convert to WAV to avoid MP3 encoder delay issues
            with st.spinner("Converting audio to WAV..."):
                wav_bytes, wav_filename, audio_wave = convert_audio_to_wav_bytes(audio_bytes, audio_upload.name)
        elif audio_path:
            if os.path.exists(audio_path):
                audio_bytes = audio_bytes_from_path(audio_path)
                audio_label = audio_path
                # Convert to WAV to avoid MP3 encoder delay issues
                with st.spinner("Converting audio to WAV..."):
                    wav_bytes, wav_filename, audio_wave = convert_audio_to_wav_bytes(audio_bytes, os.path.basename(audio_path))
            else:
                st.error("Audio path does not exist.")

        transcript_text = None
        transcript_label = None
        if transcript_upload is not None:
            transcript_text = transcript_upload.read().decode("utf-8", errors="replace")
            transcript_label = transcript_upload.name
        elif transcript_path:
            if os.path.exists(transcript_path):
                with open(transcript_path, "r", encoding="utf-8", errors="replace") as f:
                    transcript_text = f.read()
                transcript_label = transcript_path
            else:
                st.error("Transcript path does not exist.")

        if audio_bytes and transcript_text and wav_bytes and audio_wave is not None:
            st.success(f"Loaded audio: {audio_label}")
            st.success(f"Loaded transcript: {transcript_label}")
            items, has_speakers = load_transcript_items(transcript_text, display_mode)
            
            # Determine the player mode based on display mode and speaker detection
            if display_mode == "words":
                player_mode = "word"
            else:
                player_mode = "diarization" if has_speakers else "segments"
            
            player_wav_bytes, preview_seconds = create_wavesurfer_preview(wav_bytes, audio_wave, sr=16000)
            if preview_seconds is not None:
                st.warning(
                    f"Waveform preview limited to the first {format_duration(preview_seconds)} "
                    f"to avoid large in-browser audio. Full audio is unchanged."
                )

            # Use WAV bytes for playback to avoid encoder delay issues
            audio_url = audio_data_url(player_wav_bytes, wav_filename or audio_label)
            st.components.v1.html(
                build_player_html(audio_url, items, player_mode),
                height=520,
                scrolling=True,
            )
        # else:
        #     st.info("Provide both audio and transcript to start.")
    
    else:
        # Transcribe mode - full WhisperX pipeline
        st.write("Upload an audio file to transcribe using WhisperX, then review the results.")

        if "transcribe_language_name" not in st.session_state:
            st.session_state.transcribe_language_name = "German"

        # File upload and language selection
        col1, col2 = st.columns(2)
        with col1:
            transcribe_audio = st.file_uploader("Upload audio file", type=["wav", "mp3", "flac", "m4a"], key="transcribe_audio")

        if transcribe_audio is not None:
            upload_size = getattr(transcribe_audio, "size", "unknown")
            upload_signature = f"{transcribe_audio.name}:{upload_size}"
            if st.session_state.get("transcribe_lang_detect_sig") != upload_signature:
                with st.spinner("Detecting language..."):
                    try:
                        upload_bytes = transcribe_audio.getvalue()
                        detected_code, detected_prob = detect_language_from_audio_bytes(upload_bytes, sr=16000)
                        st.session_state.transcribe_lang_detect_sig = upload_signature
                        st.session_state.transcribe_lang_detect_code = detected_code
                        st.session_state.transcribe_lang_detect_prob = detected_prob
                        if detected_code in LANGUAGE_CODE_TO_NAME:
                            st.session_state.transcribe_language_name = LANGUAGE_CODE_TO_NAME[detected_code]
                            st.session_state.transcribe_lang_detect_msg = (
                                f"Auto-detected language: {LANGUAGE_CODE_TO_NAME[detected_code]} "
                                f"({detected_code}, confidence {detected_prob:.0%}). "
                                "You can still change it manually."
                            )
                            st.session_state.transcribe_lang_detect_level = "info"
                        else:
                            st.session_state.transcribe_lang_detect_msg = (
                                f"Detected language code '{detected_code}' is not in the selectable list. "
                                "Please select language manually."
                            )
                            st.session_state.transcribe_lang_detect_level = "warning"
                    except Exception as e:
                        st.session_state.transcribe_lang_detect_sig = upload_signature
                        st.session_state.transcribe_lang_detect_code = None
                        st.session_state.transcribe_lang_detect_prob = 0.0
                        st.session_state.transcribe_lang_detect_msg = (
                            f"Could not auto-detect language ({str(e)}). Please set it manually."
                        )
                        st.session_state.transcribe_lang_detect_level = "warning"

        with col2:
            language_name = st.selectbox(
                "Language",
                list(LANGUAGE_MAPPING.keys()),
                key="transcribe_language_name",
                help="Select language. Swiss German automatically uses the Swiss German model."
            )
            language = LANGUAGE_MAPPING[language_name]
            detect_msg = st.session_state.get("transcribe_lang_detect_msg")
            detect_level = st.session_state.get("transcribe_lang_detect_level", "info")
            if transcribe_audio is not None and detect_msg:
                if detect_level == "warning":
                    st.warning(detect_msg)
                else:
                    st.info(detect_msg)
        
        # Configuration sections
        col_config1, col_config2 = st.columns(2)
        
        with col_config1:
            st.write("**Model Configuration**")
            
            # Auto-select model based on language
            if language == "ch_de":
                default_model = "/storage/research/dsl_shared/solutions/whisperx/cache/whisper/swhisper-large-1.1"
                st.info("ℹ️ Using Swiss German Whisper model")
            else:
                default_model = "large-v3-turbo"
                st.info(f"ℹ️ Using default Large v3 Turbo whisper model")
            
            use_custom_model = st.checkbox("Use custom model path", value=False)
            if use_custom_model:
                whisper_model = st.text_input("Custom model path", value=default_model, help="Enter model name or path")
            else:
                whisper_model = default_model
                # st.text(f"Using model: {whisper_model}")
            
            # VAD options
            use_vad = st.checkbox("Use VAD pre-filtering", value=False, help="Use Silero VAD to filter speech segments before transcription")
            if use_vad:
                vad_max_pause = st.slider("VAD max pause (seconds)", min_value=0.1, max_value=1.0, value=0.25, step=0.05, help="Maximum pause duration to merge VAD segments")
            else:
                vad_max_pause = 0.5
        
        with col_config2:
            st.write("**Speaker Diarization**")
            # st.info("ℹ️ Diarization runs automatically if HuggingFace token is found")
            
            col_min, col_min_val = st.columns([1, 1])
            with col_min:
                use_min_speakers = st.checkbox("Set min speakers", value=False)
            with col_min_val:
                if use_min_speakers:
                    min_speakers = st.number_input("Min", min_value=1, value=2, step=1, label_visibility="collapsed")
                else:
                    min_speakers = None
            
            col_max, col_max_val = st.columns([1, 1])
            with col_max:
                use_max_speakers = st.checkbox("Set max speakers", value=False)
            with col_max_val:
                if use_max_speakers:
                    max_speakers = st.number_input("Max", min_value=1, value=4, step=1, label_visibility="collapsed")
                else:
                    max_speakers = None
        
        # Create a unique key for the current configuration to detect changes
        current_config = f"{transcribe_audio.name if transcribe_audio else 'none'}_{language}_{whisper_model}_{use_vad}_{vad_max_pause}"
        
        # Clear results if configuration changed
        if 'last_config' not in st.session_state:
            st.session_state.last_config = None
        if st.session_state.last_config != current_config:
            st.session_state.transcription_results = None
        
        if st.button("Start Transcription", type="primary"):
            if transcribe_audio is None:
                st.error("Please upload an audio file first.")
            else:
                # Determine device
                if torch.cuda.is_available():
                    device = "cuda"
                    compute_type = "float16"
                else:
                    device = "cpu"
                    compute_type = "float32"
                
                with st.spinner(f"Transcribing audio..."):
                    try:
                        # Convert to WAV bytes to avoid MP3 encoder delay issues
                        audio_bytes = transcribe_audio.getvalue()
                        sampling_rate = 16000
                        with st.spinner("Converting audio to WAV..."):
                            wav_bytes, wav_filename, audio = convert_audio_to_wav_bytes(
                                audio_bytes, transcribe_audio.name, sr=sampling_rate
                            )
                        # st.caption(f"Audio debug (full): {audio_debug_stats(audio)}")
                        if audio is None or audio.size == 0:
                            raise ValueError(
                                "Decoded waveform is empty. This file likely failed ffmpeg decoding "
                                "or contains no valid audio samples."
                            )
                        player_wav_bytes, preview_seconds = create_wavesurfer_preview(wav_bytes, audio, sr=sampling_rate)
                        
                        # Handle Swiss German language code
                        align_language = language
                        if language == "ch_de":
                            align_language = "de"  # Use German alignment for Swiss German
                        
                        # Load model
                        with st.spinner("Loading Whisper model..."):
                            model = whisperx.load_model(whisper_model, device=device, compute_type=compute_type, language=align_language)
                        
                        # Transcribe with optional VAD
                        if use_vad:
                            with st.spinner(f"Running VAD (max_pause={vad_max_pause}s)..."):
                                vad_segments = get_vad_segments(audio, max_pause=vad_max_pause, return_seconds=False, sampling_rate=sampling_rate)
                                # st.info(f"VAD detected {len(vad_segments)} speech segments")
                            
                            with st.spinner("Transcribing VAD segments..."):
                                all_segments = []
                                for idx, vad_seg in enumerate(vad_segments):
                                    start_sample = int(vad_seg['start'])
                                    end_sample = int(vad_seg['end'])
                                    segment_audio = audio[start_sample:end_sample]
                                    # st.caption(
                                    #     f"Audio debug (VAD segment {idx}): "
                                    #     f"{audio_debug_stats(segment_audio)} "
                                    #     f"[samples {start_sample}:{end_sample}]"
                                    # )
                                    if segment_audio.size == 0:
                                        st.warning(
                                            f"Skipping empty VAD segment {idx} "
                                            f"({start_sample}:{end_sample})."
                                        )
                                        continue
                                    
                                    seg_result = model.transcribe(segment_audio, batch_size=16)
                                    
                                    # Adjust timestamps
                                    start_time_offset = start_sample / sampling_rate
                                    for seg in seg_result.get('segments', []):
                                        seg['start'] += start_time_offset
                                        seg['end'] += start_time_offset
                                        all_segments.append(seg)
                                
                                result = {'segments': all_segments, 'language': seg_result.get('language', align_language)}
                        else:
                            with st.spinner("Transcribing audio..."):
                                # st.caption(f"Audio debug (pre-transcribe): {audio_debug_stats(audio)}")
                                result = model.transcribe(audio, batch_size=16)
                        
                        # Always do alignment
                        with st.spinner("Aligning words..."):
                            model_a, metadata = whisperx.load_align_model(language_code=align_language, device=device)
                            aligned = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
                        
                        # Automatic diarization if HF token is present
                        has_speakers = False
                        hf_token = read_hf_token('hf_token.txt')
                        if hf_token:
                            with st.spinner("Running speaker diarization..."):
                                diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=hf_token, device=device)
                                diarize_segments = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
                                aligned = whisperx.assign_word_speakers(diarize_segments, aligned)
                                has_speakers = True
                                # st.info("✅ Speaker diarization completed")
                        else:
                            st.info("ℹ️ HuggingFace token not found - skipping diarization")
                        
                        # Generate CSVs (in memory, not saved)
                        transcription_csv = generate_transcription_csv(aligned, has_speakers=has_speakers)
                        words_csv = generate_words_csv(aligned)
                        
                        # Store results in session state
                        st.session_state.transcription_results = {
                            'transcription_csv': transcription_csv,
                            'words_csv': words_csv,
                            'audio_bytes': audio_bytes,
                            'wav_bytes': wav_bytes,
                            'player_wav_bytes': player_wav_bytes,
                            'preview_seconds': preview_seconds,
                            'wav_filename': wav_filename,
                            'audio_name': transcribe_audio.name,
                            'has_speakers': has_speakers
                        }
                        st.session_state.last_config = current_config
                        
                        st.success("✅ Transcription completed!")
                        
                    except Exception as e:
                        st.error(f"Transcription failed: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
        
        # Display results if they exist in session state
        if st.session_state.get('transcription_results'):
            results = st.session_state.transcription_results
            transcription_csv = results['transcription_csv']
            words_csv = results['words_csv']
            audio_bytes = results['audio_bytes']
            wav_bytes = results.get('wav_bytes')
            player_wav_bytes = results.get('player_wav_bytes')
            preview_seconds = results.get('preview_seconds')
            wav_filename = results.get('wav_filename')
            audio_name = results['audio_name']
            has_speakers = results['has_speakers']
            
            # Display mode selector (only shown when results exist)
            display_mode = st.radio(
                "Display transcript as",
                ["segments", "words"],
                index=0,
                horizontal=True,
                key="transcribe_mode",
                help="Segments = sentence-level view, Words = word-by-word view"
            )
            
            # Display results based on selected mode
            if display_mode == "words":
                items, _ = load_transcript_items(words_csv, "words")
                player_mode = "word"
            else:
                items, _ = load_transcript_items(transcription_csv, "segments")
                player_mode = "diarization" if has_speakers else "segments"
            
            if preview_seconds is not None:
                st.warning(
                    f"Waveform preview limited to the first {format_duration(preview_seconds)} "
                    f"to avoid large in-browser audio. Transcription uses the full audio."
                )

            # Use WAV bytes for playback
            audio_url = audio_data_url(player_wav_bytes or wav_bytes, wav_filename)
            st.components.v1.html(
                build_player_html(audio_url, items, player_mode),
                height=520,
                scrolling=True,
            )
            
            # Download ZIP with all outputs
            zip_buffer = io.BytesIO()
            base_name = os.path.splitext(audio_name)[0]
            text_output = transcription_text_from_csv(transcription_csv)
            with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(f"{base_name}_transcription.csv", transcription_csv)
                zf.writestr(f"{base_name}_words.csv", words_csv)
                zf.writestr(f"{base_name}_text.txt", text_output)
                if wav_bytes:
                    zf.writestr(wav_filename or f"{base_name}.wav", wav_bytes)
            zip_buffer.seek(0)

            col_text, col_zip = st.columns(2)
            with col_text:
                st.download_button(
                    "Download text (.txt)",
                    text_output,
                    file_name=f"{base_name}_text.txt",
                    mime="text/plain",
                    key="download_text_txt",
                    use_container_width=True,
                )
            with col_zip:
                st.download_button(
                    "Download all outputs (ZIP)",
                    zip_buffer.getvalue(),
                    file_name=f"{base_name}_outputs.zip",
                    mime="application/zip",
                    key="download_outputs_zip",
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()

