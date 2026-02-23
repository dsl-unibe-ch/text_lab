import os
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

import sys
import zipfile
import io
import torch
import whisperx
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from auth import check_token
from language_mappings import (
    TRANSCRIBE_LANGUAGE_CODE_TO_NAME as LANGUAGE_CODE_TO_NAME,
    TRANSCRIBE_LANGUAGE_MAPPING as LANGUAGE_MAPPING,
)


from core.transcribe_engine import (
    load_transcript_items,
    transcription_text_from_csv,
    audio_bytes_from_path,
    format_duration,
    convert_audio_to_wav_bytes,
    detect_language_from_audio_bytes,
    create_wavesurfer_preview,
    audio_data_url,
    build_player_html,
    get_vad_segments,
    generate_transcription_csv,
    generate_words_csv,
    read_hf_token
)

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
            with st.spinner("Converting audio to WAV..."):
                wav_bytes, wav_filename, audio_wave = convert_audio_to_wav_bytes(audio_bytes, audio_upload.name)
        elif audio_path:
            if os.path.exists(audio_path):
                audio_bytes = audio_bytes_from_path(audio_path)
                audio_label = audio_path
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

            audio_url = audio_data_url(player_wav_bytes, wav_filename or audio_label)
            st.components.v1.html(
                build_player_html(audio_url, items, player_mode),
                height=520,
                scrolling=True,
            )
    
    else:
        # Transcribe mode - full WhisperX pipeline
        st.write("Upload an audio file to transcribe using WhisperX, then review the results.")

        if "transcribe_language_name" not in st.session_state:
            st.session_state.transcribe_language_name = "German"

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
            
            # Only show detection messages if a file was actually uploaded
            if transcribe_audio is not None:
                detect_msg = st.session_state.get("transcribe_lang_detect_msg")
                detect_level = st.session_state.get("transcribe_lang_detect_level", "info")
                if detect_msg:
                    if detect_level == "warning":
                        st.warning(detect_msg)
                    else:
                        st.info(detect_msg)
        
        # Configuration sections
        col_config1, col_config2 = st.columns(2)
        
        with col_config1:
            st.write("**Model Configuration**")
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
            
            use_vad = st.checkbox("Use VAD pre-filtering", value=False, help="Use Silero VAD to filter speech segments before transcription")
            if use_vad:
                vad_max_pause = st.slider("VAD max pause (seconds)", min_value=0.1, max_value=1.0, value=0.25, step=0.05, help="Maximum pause duration to merge VAD segments")
            else:
                vad_max_pause = 0.5
        
        with col_config2:
            st.write("**Speaker Diarization**")
            
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
        
        current_config = f"{transcribe_audio.name if transcribe_audio else 'none'}_{language}_{whisper_model}_{use_vad}_{vad_max_pause}"
        
        if 'last_config' not in st.session_state:
            st.session_state.last_config = None
        if st.session_state.last_config != current_config:
            st.session_state.transcription_results = None
        
        if st.button("Start Transcription", type="primary"):
            if transcribe_audio is None:
                st.error("Please upload an audio file first.")
            else:
                if torch.cuda.is_available():
                    device = "cuda"
                    compute_type = "float16"
                else:
                    device = "cpu"
                    compute_type = "float32"
                
                with st.spinner(f"Transcribing audio..."):
                    try:
                        audio_bytes = transcribe_audio.getvalue()
                        sampling_rate = 16000
                        with st.spinner("Converting audio to WAV..."):
                            wav_bytes, wav_filename, audio = convert_audio_to_wav_bytes(
                                audio_bytes, transcribe_audio.name, sr=sampling_rate
                            )
                        
                        if audio is None or audio.size == 0:
                            raise ValueError(
                                "Decoded waveform is empty. This file likely failed ffmpeg decoding "
                                "or contains no valid audio samples."
                            )
                        
                        player_wav_bytes, preview_seconds = create_wavesurfer_preview(wav_bytes, audio, sr=sampling_rate)
                        
                        align_language = language
                        if language == "ch_de":
                            align_language = "de" 
                        
                        with st.spinner("Loading Whisper model..."):
                            model = whisperx.load_model(whisper_model, device=device, compute_type=compute_type, language=align_language)
                        
                        if use_vad:
                            with st.spinner(f"Running VAD (max_pause={vad_max_pause}s)..."):
                                vad_segments = get_vad_segments(audio, max_pause=vad_max_pause, return_seconds=False, sampling_rate=sampling_rate)
                            
                            with st.spinner("Transcribing VAD segments..."):
                                all_segments = []
                                for idx, vad_seg in enumerate(vad_segments):
                                    start_sample = int(vad_seg['start'])
                                    end_sample = int(vad_seg['end'])
                                    segment_audio = audio[start_sample:end_sample]
                                    
                                    if segment_audio.size == 0:
                                        st.warning(f"Skipping empty VAD segment {idx} ({start_sample}:{end_sample}).")
                                        continue
                                    
                                    seg_result = model.transcribe(segment_audio, batch_size=16)
                                    
                                    start_time_offset = start_sample / sampling_rate
                                    for seg in seg_result.get('segments', []):
                                        seg['start'] += start_time_offset
                                        seg['end'] += start_time_offset
                                        all_segments.append(seg)
                                
                                result = {'segments': all_segments, 'language': seg_result.get('language', align_language)}
                        else:
                            with st.spinner("Transcribing audio..."):
                                result = model.transcribe(audio, batch_size=16)
                        
                        with st.spinner("Aligning words..."):
                            model_a, metadata = whisperx.load_align_model(language_code=align_language, device=device)
                            aligned = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
                        
                        has_speakers = False
                        hf_token = read_hf_token('hf_token.txt')
                        if hf_token:
                            with st.spinner("Running speaker diarization..."):
                                diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=hf_token, device=device)
                                diarize_segments = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
                                aligned = whisperx.assign_word_speakers(diarize_segments, aligned)
                                has_speakers = True
                        else:
                            st.info("ℹ️ HuggingFace token not found - skipping diarization")
                        
                        transcription_csv = generate_transcription_csv(aligned, has_speakers=has_speakers)
                        words_csv = generate_words_csv(aligned)
                        
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
        
        if st.session_state.get('transcription_results'):
            results = st.session_state.transcription_results
            transcription_csv = results['transcription_csv']
            words_csv = results['words_csv']
            wav_bytes = results.get('wav_bytes')
            player_wav_bytes = results.get('player_wav_bytes')
            preview_seconds = results.get('preview_seconds')
            wav_filename = results.get('wav_filename')
            audio_name = results['audio_name']
            has_speakers = results['has_speakers']
            
            display_mode = st.radio(
                "Display transcript as",
                ["segments", "words"],
                index=0,
                horizontal=True,
                key="transcribe_mode",
                help="Segments = sentence-level view, Words = word-by-word view"
            )
            
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

            audio_url = audio_data_url(player_wav_bytes or wav_bytes, wav_filename)
            st.components.v1.html(
                build_player_html(audio_url, items, player_mode),
                height=520,
                scrolling=True,
            )
            
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
