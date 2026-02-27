import os
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

import io
import csv
import json
import base64
import html
import mimetypes
import uuid
import tempfile
import subprocess
import functools
import numpy as np
import torch
import soundfile as sf
import whisperx
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

# --- Constants ---
WAVESURFER_MAX_BYTES = 75 * 1024 * 1024
WAVESURFER_MAX_SECONDS = 30 * 60

# ==========================================
#        DATA & TIME PARSING
# ==========================================

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:06.3f}"

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
        else:  
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

# ==========================================
#        AUDIO DECODING & PROCESSING
# ==========================================

def audio_bytes_from_path(path):
    with open(path, "rb") as f:
        return f.read()

def decode_audio_bytes(audio_bytes, sr=16000):
    output_args = [
        "-f", "s16le", "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr), "pipe:1",
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
    audio = decode_audio_bytes(audio_bytes, sr=sr)
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, audio, sr, format="WAV", subtype="PCM_16")
    wav_bytes = wav_buffer.getvalue()
    wav_filename = os.path.splitext(original_filename)[0] + ".wav"
    return wav_bytes, wav_filename, audio

@functools.lru_cache(maxsize=1)
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

# ==========================================
#        VAD & WHISPER HELPERS
# ==========================================

def merge_speech_timestamps(speech_timestamps, max_pause=0.5, return_seconds=True, sampling_rate=16000):
    if not speech_timestamps:
        return []
    if not return_seconds:
        max_pause *= sampling_rate 
    
    merged = []
    current_segment = speech_timestamps[0].copy()
    
    for next_segment in speech_timestamps[1:]:
        pause = next_segment['start'] - current_segment['end']
        
        if pause < max_pause:
            current_segment['end'] = next_segment['end']
        else:
            merged.append(current_segment)
            current_segment = next_segment.copy()
    
    merged.append(current_segment)
    return merged

def get_vad_segments(audio_file, max_pause=None, return_seconds=True, sampling_rate=16000):
    model = load_silero_vad()
    if isinstance(audio_file, str):
        wav = read_audio(audio_file, sampling_rate=sampling_rate)
    else:
        wav = audio_file
    
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        return_seconds=return_seconds,
        sampling_rate=sampling_rate
    )
    
    if max_pause:
        speech_timestamps = merge_speech_timestamps(
            speech_timestamps, 
            max_pause=max_pause, 
            return_seconds=return_seconds,
            sampling_rate=sampling_rate
        )
    return speech_timestamps

def read_hf_token(token_arg):
    if os.path.isfile(token_arg):
        try:
            with open(token_arg, 'r') as file:
                return file.read().strip()
        except Exception as e:
            print(f"Warning: Could not read token: {e}")
            return None
    elif isinstance(token_arg, str) and len(token_arg) > 10 and token_arg.startswith("hf_"):
        return token_arg
    return None

def generate_transcription_csv(result, has_speakers=False):
    output = io.StringIO()
    writer = csv.writer(output, delimiter='\t')
    
    if has_speakers:
        writer.writerow(["segment_id", "Start", "End", "Speaker", "Text"])
    else:
        writer.writerow(["segment_id", "Start", "End", "Text"])
    
    segment_id = 1
    
    if has_speakers:
        for segment in result.get('segments', []):
            words = segment.get('words', [])
            if not words: continue
            
            current_speaker = words[0].get('speaker') or "UNKNOWN"
            current_text = words[0]['word']
            start_time = words[0]['start']
            end_time = words[0]['end']
            
            for w in words[1:]:
                speaker = w.get('speaker') or "UNKNOWN"
                if speaker != current_speaker:
                    writer.writerow([segment_id, format_time(start_time), format_time(end_time), current_speaker, current_text.strip()])
                    segment_id += 1
                    current_speaker = speaker
                    current_text = w['word']
                    start_time = w['start']
                else:
                    current_text += " " + w['word']
                end_time = w['end']
            
            writer.writerow([segment_id, format_time(start_time), format_time(end_time), current_speaker, current_text.strip()])
            segment_id += 1
    else:
        for segment in result.get("segments", []):
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            text = segment.get("text", "").strip()
            writer.writerow([segment_id, format_time(start_time), format_time(end_time), text])
            segment_id += 1
    
    return output.getvalue()

def generate_words_csv(result):
    output = io.StringIO()
    writer = csv.writer(output, delimiter='\t')
    writer.writerow(["segment_id", "word_id", "Start", "End", "Word", "Speaker"])
    
    segment_id = 1
    for segment in result.get("segments", []):
        words = segment.get("words", [])
        if not words: continue
        
        word_id = 1
        current_speaker = words[0].get('speaker') or "UNKNOWN"
        
        for w in words:
            speaker = w.get('speaker') or "UNKNOWN"
            if speaker != current_speaker:
                segment_id += 1
                word_id = 1
                current_speaker = speaker
            
            start = w.get("start", 0.0)
            end = w.get("end", 0.0)
            word = w.get("word", w.get("text", ""))
            writer.writerow([segment_id, word_id, format_time(start), format_time(end), word, speaker])
            word_id += 1
        segment_id += 1
    return output.getvalue()


# --- SUBTITLE GENERATORS ---

def _format_timestamp(seconds: float, separator: str = ",") -> str:
    """Formats seconds into SRT/VTT timestamp format: HH:MM:SS,mmm or HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"

def generate_srt(result) -> str:
    """Generates standard SubRip (.srt) format subtitle string."""
    output = io.StringIO()
    segment_id = 1
    
    for segment in result.get('segments', []):
        start_time = segment.get("start", 0)
        end_time = segment.get("end", 0)
        text = segment.get("text", "").strip()
        
        if not text:
            continue
            
        start_str = _format_timestamp(start_time, separator=",")
        end_str = _format_timestamp(end_time, separator=",")
        
        output.write(f"{segment_id}\n")
        output.write(f"{start_str} --> {end_str}\n")
        output.write(f"{text}\n\n")
        
        segment_id += 1
        
    return output.getvalue()

def generate_vtt(result) -> str:
    """Generates standard WebVTT (.vtt) format subtitle string."""
    output = io.StringIO()
    output.write("WEBVTT\n\n")
    
    for segment in result.get('segments', []):
        start_time = segment.get("start", 0)
        end_time = segment.get("end", 0)
        text = segment.get("text", "").strip()
        
        if not text:
            continue
            
        start_str = _format_timestamp(start_time, separator=".")
        end_str = _format_timestamp(end_time, separator=".")
        
        output.write(f"{start_str} --> {end_str}\n")
        output.write(f"{text}\n\n")
        
    return output.getvalue()

# ==========================================
#        WAVESURFER HTML GENERATOR
# ==========================================

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
    if isinstance(audio_path_or_bytes, str):
        with open(audio_path_or_bytes, 'rb') as f:
            audio_bytes = f.read()
    else:
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
        if (!items.length) return;
        let idx = -1;
        for (let i = 0; i < items.length; i++) {{
          if (currentTime >= items[i].start && currentTime <= items[i].end) {{
            idx = i;
            break;
          }}
        }}
        if (idx === activeIndex) return;
        if (activeIndex >= 0) {{
          const prev = transcriptEl.children[activeIndex];
          if (prev) prev.classList.remove("ws-active");
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
      .ws-container {{ font-family: Arial, sans-serif; }}
      .ws-controls {{ display: flex; align-items: center; gap: 12px; margin: 8px 0 16px 0; color: #ffffff; }}
      .ws-time {{ font-variant-numeric: tabular-nums; background: #1b1b1b; border: 1px solid #3a3a3a; padding: 4px 8px; border-radius: 6px; }}
      .ws-transcript {{ border: 1px solid #e0e0e0; padding: 12px; max-height: 360px; overflow-y: auto; background: #fafafa; line-height: 1.5; }}
      .ws-seg {{ padding: 6px 8px; margin-bottom: 6px; border-radius: 6px; cursor: pointer; }}
      .ws-meta {{ font-size: 12px; color: #666666; margin-bottom: 4px; }}
      .ws-text {{ font-size: 14px; color: #000; }}
      .ws-word {{ cursor: pointer; padding: 2px 2px; border-radius: 4px; color: #000;}}
      .ws-active {{ background: #ffe8d6; }}
      button {{ border: 1px solid #333333; background: #ffffff; padding: 6px 10px; cursor: pointer; color: #000;}}
    </style>
    """