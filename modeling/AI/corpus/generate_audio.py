#!/usr/bin/env python3
"""
Audio Generator for Oxford CEFR Master Corpus.
Generates MP3 audio files from the text entries using macOS native TTS (`say`)
and converts them to MP3 using `ffmpeg` in parallel.
"""

import os
import sys
import pandas as pd
import subprocess
import tempfile
import concurrent.futures
import argparse

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Voice mapping based on gender
# Female UK voices available on macOS
FEMALE_VOICES = ['Flo (English (UK))', 'Sandy (English (UK))', 'Shelley (English (UK))']
# Male UK voices available on macOS
MALE_VOICES = ['Daniel', 'Eddy (English (UK))', 'Reed (English (UK))', 'Rocko (English (UK))']

def get_voice_for_speaker(speaker_id, gender):
    # Deterministic choice based on speaker_id hash
    voices = FEMALE_VOICES if gender == 'Female' else MALE_VOICES
    if not voices:
        return 'Daniel'
    idx = hash(speaker_id) % len(voices)
    return voices[idx]

def get_speed_rate(speed):
    # Map speed labels to words-per-minute
    if speed == 'Slow':
        return 140
    elif speed == 'Fast':
        return 210
    else: # Normal
        return 175

def generate_single_audio(args):
    entry_id, text, speaker_id, gender, speed, output_dir = args
    output_path = os.path.join(output_dir, f"{entry_id}.mp3")
    
    # Skip if already exists
    if os.path.exists(output_path):
        return True, entry_id, "skipped"
        
    voice = get_voice_for_speaker(speaker_id, gender)
    rate = get_speed_rate(speed)
    
    # Create a temp file for the intermediate AIFF audio
    with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as temp_file:
        temp_aiff = temp_file.name
        
    try:
        # Step 1: Synthesize text to AIFF using macOS say
        cmd_say = ["say", "-v", voice, "-r", str(rate), "-o", temp_aiff, text]
        subprocess.run(cmd_say, check=True, capture_output=True)
        
        # Step 2: Convert AIFF to MP3 using ffmpeg
        # qscale:a 2 is high quality variable bitrate (approx 170-210 kbps)
        # We also downsample to 16kHz mono as specified in config
        cmd_ffmpeg = [
            "ffmpeg", "-y", "-i", temp_aiff,
            "-ar", "16000", "-ac", "1",
            "-codec:a", "libmp3lame", "-qscale:a", "2",
            output_path
        ]
        subprocess.run(cmd_ffmpeg, check=True, capture_output=True)
        return True, entry_id, "success"
    except Exception as e:
        return False, entry_id, str(e)
    finally:
        if os.path.exists(temp_aiff):
            try:
                os.remove(temp_aiff)
            except OSError:
                pass

def main():
    parser = argparse.ArgumentParser(description="Generate MP3 audio for CEFR corpus entries.")
    parser.add_argument("--sample-size", type=int, default=None, help="Limit generation to N random samples.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel processes.")
    parser.add_argument("--output-dir", type=str, default=os.path.join(project_root, 'data', 'corpus', 'audio_mp3'), help="Directory to save MP3s.")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    metadata_path = os.path.join(project_root, 'data', 'corpus', 'audio_metadata.csv')
    if not os.path.exists(metadata_path):
        print(f"Metadata file not found at {metadata_path}. Please run build_corpus.py first.")
        sys.exit(1)
        
    df = pd.read_csv(metadata_path)
    print(f"Loaded metadata with {len(df)} entries.")
    
    if args.sample_size:
        df = df.sample(n=min(args.sample_size, len(df)), random_state=42).copy()
        print(f"Subset selected: {len(df)} entries for generation.")
        
    tasks = []
    for _, row in df.iterrows():
        tasks.append((
            row['id'],
            row['text'],
            row['speaker_id'],
            row['gender'],
            row['speed'],
            args.output_dir
        ))
        
    print(f"Starting audio generation using {args.workers} worker processes...")
    success_count = 0
    skipped_count = 0
    fail_count = 0
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(generate_single_audio, task): task for task in tasks}
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            success, entry_id, status = future.result()
            if success:
                if status == "skipped":
                    skipped_count += 1
                else:
                    success_count += 1
            else:
                fail_count += 1
                print(f"\nFailed to generate audio for {entry_id}: {status}")
                
            if (i + 1) % 100 == 0 or (i + 1) == len(tasks):
                sys.stdout.write(f"\rProgress: {i+1}/{len(tasks)} (Generated: {success_count}, Skipped: {skipped_count}, Failed: {fail_count})")
                sys.stdout.flush()
                
    print("\n\nAudio generation complete!")
    print(f"Successfully generated: {success_count} files")
    print(f"Skipped (already exist): {skipped_count} files")
    print(f"Failed to generate: {fail_count} files")

if __name__ == '__main__':
    main()
