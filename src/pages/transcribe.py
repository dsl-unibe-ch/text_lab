import streamlit as st
import whisper
import tempfile
import json
import zipfile
import io

def seconds_to_srt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def make_srt(segments) -> str:
    srt_lines = []
    for i, seg in enumerate(segments, start=1):
        start_time = seconds_to_srt_time(seg["start"])
        end_time = seconds_to_srt_time(seg["end"])
        text = seg["text"].strip()

        srt_lines.append(str(i))
        srt_lines.append(f"{start_time} --> {end_time}")
        srt_lines.append(text)
        srt_lines.append("")  # blank line

    return "\n".join(srt_lines)

def make_csv(segments) -> str:
    lines = ["start,end,text"]
    for seg in segments:
        start = seg["start"]
        end = seg["end"]
        text = seg["text"].strip().replace('"', '""')  # Escape double quotes
        lines.append(f'{start},{end},"{text}"')
    return "\n".join(lines)

def make_json(result) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2)

def create_all_formats_zip(text: str, segments, result_dict) -> bytes:
    """
    Create an in-memory ZIP containing all four output formats:
    - transcription.txt
    - transcription.srt
    - transcription.csv
    - transcription.json
    Returns raw bytes of the ZIP for downloading.
    """
    srt_data = make_srt(segments)
    csv_data = make_csv(segments)
    json_data = make_json(result_dict)

    # Create an in-memory buffer
    zip_buffer = io.BytesIO()

    # Build the ZIP
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("transcription.txt", text)
        zf.writestr("transcription.srt", srt_data)
        zf.writestr("transcription.csv", csv_data)
        zf.writestr("transcription.json", json_data)

    # Move back to start so we can read the data from the beginning
    zip_buffer.seek(0)
    return zip_buffer.read()

def run_transcribe():
    st.title("Whisper Transcription")

    # Model selection
    model_options = ["tiny", "small", "medium", "large-v2", "large-v3"]
    model_name = st.selectbox("Select Whisper Model:", model_options, index=4)

    # Language selection
    LANG_DICT = whisper.tokenizer.LANGUAGES
    language_labels = ["Detect language automatically"] + sorted(LANG_DICT.keys())
    selected_label = st.selectbox("Select Language:", language_labels, index=0)

    # File uploader
    audio_file = st.file_uploader(
        "Upload an audio file",
        type=["m4a","mp3","webm","mp4","mpga","wav","mpeg","ogg","flac"]
    )

    if audio_file is not None:
        st.audio(audio_file, format="audio/*", start_time=0)

    # Initialize storage for the last transcription result
    if "whisper_result" not in st.session_state:
        st.session_state["whisper_result"] = None

    # Transcription button
    if st.button("Transcribe"):
        if audio_file is None:
            st.warning("Please upload an audio file first.")
        else:
            with st.spinner("Loading Whisper model and transcribing. Please wait..."):
                model = whisper.load_model(model_name)

                transcribe_options = {}
                if selected_label != "Detect language automatically":
                    transcribe_options["language"] = LANG_DICT[selected_label]

                # Write the uploaded file to a temp file for Whisper
                audio_bytes = audio_file.read()
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio_bytes)
                    tmp.flush()
                    result = model.transcribe(tmp.name, **transcribe_options)

            st.success("Transcription complete!")
            st.session_state["whisper_result"] = result

    # If we have a stored transcription, show download options
    if st.session_state["whisper_result"] is not None:
        result = st.session_state["whisper_result"]
        text = result["text"]
        segments = result.get("segments", [])

        st.subheader("Download your transcription")
        format_option = st.selectbox("Select a single format:", ["Plain text", "SRT", "CSV", "JSON"])

        # Single-format download
        if format_option == "Plain text":
            st.download_button(
                label="Download .txt",
                data=text,
                file_name="transcription.txt",
                mime="text/plain",
            )
        elif format_option == "SRT":
            srt_data = make_srt(segments)
            st.download_button(
                label="Download .srt",
                data=srt_data,
                file_name="transcription.srt",
                mime="text/plain",
            )
        elif format_option == "CSV":
            csv_data = make_csv(segments)
            st.download_button(
                label="Download .csv",
                data=csv_data,
                file_name="transcription.csv",
                mime="text/csv",
            )
        elif format_option == "JSON":
            json_data = make_json(result)
            st.download_button(
                label="Download .json",
                data=json_data,
                file_name="transcription.json",
                mime="application/json",
            )

        # Download ALL formats in one ZIP
        if st.button("Download All Formats (ZIP)"):
            zip_bytes = create_all_formats_zip(text, segments, result)
            st.download_button(
                label="Click to download ZIP",
                data=zip_bytes,
                file_name="transcription_outputs.zip",
                mime="application/zip",
            )

def main():
    run_transcribe()

if __name__ == "__main__":
    main()
