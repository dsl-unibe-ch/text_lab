import streamlit as st
import whisper
from io import BytesIO

def run_transcribe():
    st.title("Whisper Transcription")

    # Model selection
    model_name = st.selectbox(
        "Select Whisper Model:",
        [
            "tiny", "base", "small", "medium", "large"
            # or you can include "large-v2" etc. if you prefer
        ],
        index=1  # default to "base"
    )

    # Language selection
    language_options = [
        "Detect language automatically",
        "en", "de", "fr", "es", "it", "zh", "ar"
        # Add more if you need
    ]
    language = st.selectbox("Select Language:", language_options, index=0)

    # File uploader
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "ogg", "flac"])

    if audio_file is not None:
        st.audio(audio_file, format="audio/*", start_time=0)

    # Transcription button
    if st.button("Transcribe"):
        if audio_file is None:
            st.warning("Please upload an audio file first.")
        else:
            with st.spinner("Loading Whisper model and transcribing..."):
                # Load the selected Whisper model
                model = whisper.load_model(model_name)

                # If user chose "Detect language automatically", do not pass a language
                # Otherwise pass the chosen language code
                transcribe_options = {}
                if language != "Detect language automatically":
                    transcribe_options["language"] = language

                # Read the uploaded file as bytes and transcribe
                audio_bytes = audio_file.read()

                # Whisper can work on a file path or bytes. 
                # The simplest might be to save to a temp file, but let's try direct I/O:
                # Some versions of whisper might require an actual file path. If that's the case,
                # you'd write the bytes to a NamedTemporaryFile. For now, let's do that approach:

                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio_bytes)
                    tmp.flush()  # ensure all data is written
                    result = model.transcribe(tmp.name, **transcribe_options)

                text = result["text"]

            st.success("Transcription complete!")
            st.write(text)

            # Provide a download button
            st.download_button(
                label="Download Transcription",
                data=text,
                file_name="transcription.txt",
                mime="text/plain",
            )

def main():
    run_transcribe()

if __name__ == "__main__":
    main()
