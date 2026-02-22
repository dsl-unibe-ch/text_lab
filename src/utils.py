import os
import whisperx
import csv

from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

##### Utility functions for WhisperX and Silero VAD #####

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:06.3f}"

def read_hf_token(token_arg):
    if os.path.isfile(token_arg):
        try:
            with open(token_arg, 'r') as file:
                token = file.read().strip()
            return token
        except Exception as e:
            print(f"Warning: Could not read token from file due to error: {e}")
            return None
    elif isinstance(token_arg, str) and len(token_arg) > 10 and token_arg.startswith("hf_"):
        return token_arg
    else:
        print("Warning: Provided argument is neither a valid token nor a valid file path.")
        return None

def generate_transcription_csv(result, has_speakers=False):
    import io
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
            if not words:
                continue
            
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
    import io
    output = io.StringIO()
    writer = csv.writer(output, delimiter='\t')
    
    writer.writerow(["segment_id", "word_id", "Start", "End", "Word", "Speaker"])
    
    segment_id = 1
    
    for segment in result.get("segments", []):
        words = segment.get("words", [])
        if not words:
            continue
        
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


def save_results(result, audio_file_path, output_dir, has_speakers=False):
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
    
    transcription_csv = generate_transcription_csv(result, has_speakers)
    transcription_file = os.path.join(output_dir, f"{base_name}_transcription.csv")
    with open(transcription_file, 'w', encoding='utf-8') as f:
        f.write(transcription_csv)
    print(f"Transcription saved to {transcription_file}")
    
    words_csv = generate_words_csv(result)
    words_file = os.path.join(output_dir, f"{base_name}_words.csv")
    with open(words_file, 'w', encoding='utf-8') as f:
        f.write(words_csv)
    print(f"Word-level timestamps saved to {words_file}")


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