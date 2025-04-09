import streamlit as st
import whisper
from io import BytesIO
import tempfile

def run_transcribe():
    st.title("Whisper Transcription")

    # Model selection: only list models you actually downloaded
    model_options = [
        "tiny", "small", "medium", "large-v2", "large-v3"
    ]
    model_name = st.selectbox(
        "Select Whisper Model:",
        model_options,
        index=4  # default to large-v3
    )

    # Gather all Whisper-supported languages
    # This returns a dict like {"english": "en", "chinese": "zh", "german": "de", ...}
    LANG_DICT = whisper.tokenizer.LANGUAGES
    # Make a sorted list of language names for the dropdown
    language_labels = ["Detect language automatically"] + sorted(LANG_DICT.keys())

    # Language selection
    selected_label = st.selectbox("Select Language:", language_labels, index=0)

    # File uploader: allow multiple audio formats
    audio_file = st.file_uploader(
        "Upload an audio file",
        type=["m4a","mp3","webm","mp4","mpga","wav","mpeg","ogg","flac"]
    )

    if audio_file is not None:
        st.audio(audio_file, format="audio/*", start_time=0)

    # Transcription button
    if st.button("Transcribe"):
        if audio_file is None:
            st.warning("Please upload an audio file first.")
        else:
            with st.spinner("Loading Whisper model and transcribing..."):
                # Load the selected Whisper model.
                # By default, whisper.load_model() checks the local cache first,
                # so it should find the .pt file you added in the container
                # and won't re-download from the internet.
                model = whisper.load_model(model_name)

                # Build transcribe options
                transcribe_options = {}
                if selected_label != "Detect language automatically":
                    # Map the human-friendly label (e.g. "english") to the short code ("en")
                    transcribe_options["language"] = LANG_DICT[selected_label]

                # Whisper usually needs a real file path, so we write bytes to a temporary file
                audio_bytes = audio_file.read()
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio_bytes)
                    tmp.flush()

                    # Transcribe
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
