# utils/video_analyzer.py
from transformers import pipeline
import torch
import os
import json
from datetime import datetime

def create_transcript_filename(audio_path):
    """
    Creates a unique filename for the transcript based on the original audio filename
    and current timestamp to avoid overwriting existing files.
    """
    # Get the base name of the audio file without extension
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    # Add timestamp for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}_transcript"

def save_transcript(transcription, audio_path):
    """
    Saves the transcription in both JSON and human-readable formats.
    The JSON format preserves all information including timestamps,
    while the text format is easier for humans to read and for trend analysis.
    """
    # Create the transcripts directory if it doesn't exist
    os.makedirs("data/transcripts", exist_ok=True)
    
    # Generate base filename for our transcript files
    base_filename = create_transcript_filename(audio_path)
    
    # Save the complete transcription data as JSON
    json_path = os.path.join("data/transcripts", f"{base_filename}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(transcription, f, indent=2, ensure_ascii=False)
    
    # Save a human-readable version with timestamps
    text_path = os.path.join("data/transcripts", f"{base_filename}.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("Video Transcript\n")
        f.write("=" * 50 + "\n\n")
        
        # Write the full text first
        if isinstance(transcription, dict):
            f.write("Complete Transcript:\n")
            f.write("-" * 20 + "\n")
            f.write(transcription.get('text', '') + "\n\n")
            
            # Then write the timestamped segments
            f.write("Timestamped Segments:\n")
            f.write("-" * 20 + "\n")
            chunks = transcription.get('chunks', [])
            for chunk in chunks:
                start = chunk.get('timestamp', (0, 0))[0]
                end = chunk.get('timestamp', (0, 0))[1]
                text = chunk.get('text', '')
                f.write(f"[{start:.2f}s - {end:.2f}s]: {text}\n")
    
    return json_path, text_path

def transcribe_long_audio(audio_path):
    """
    Transcribe longer audio files using Whisper with proper timestamp handling.
    The function now includes progress updates to keep users informed during
    longer transcription tasks.
    """
    try:
        print("Initializing transcription model...")
        transcriber = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base",
            chunk_length_s=30,
            stride_length_s=5,
            return_timestamps=True
        )
        
        print(f"Starting transcription of {audio_path}...")
        print("This may take several minutes depending on the file length...")
        
        transcription = transcriber(
            audio_path,
            return_timestamps=True,
            generate_kwargs={
                "task": "transcribe",
                "language": "en"
            }
        )
        
        print("Transcription completed successfully!")
        return transcription
        
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        raise

def analyze_videos():
    """
    Main function for video analysis that processes the audio and saves
    the transcription in formats suitable for both human reading and
    further analysis.
    """
    audio_path = os.path.join("data", "scraped_videos", "foreignKey.mp3")
    
    try:
        print("\nStarting video analysis process...")
        print(f"Processing audio file: {audio_path}")
        
        # Verify file exists before processing
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found at: {audio_path}")
        
        # Process the audio
        transcription = transcribe_long_audio(audio_path)
        
        # Save the transcription
        json_path, text_path = save_transcript(transcription, audio_path)
        print("\nTranscription saved successfully!")
        print(f"JSON format: {json_path}")
        print(f"Text format: {text_path}")
        
        # Display a preview of the transcription
        if isinstance(transcription, dict):
            print("\nTranscription Preview:")
            print("-" * 50)
            preview_text = transcription.get('text', '')[:500]  # First 500 characters
            print(f"{preview_text}...")
            print("\nFull transcription available in the saved files.")
        
        return transcription, text_path  # Return both for trend analysis
        
    except Exception as e:
        print(f"Error in analyze_videos: {str(e)}")
        raise