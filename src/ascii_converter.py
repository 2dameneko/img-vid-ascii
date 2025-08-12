import cv2
import shutil
import numpy as np
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
import time
import sys
import subprocess
import zipfile
import tempfile

from charsets import CHAR_SETS, DEFAULT_CHAR_SET

FFMPEG_PATH = 'ffmpeg'
FFPLAY_PATH = 'ffplay'
FFPROBE_PATH = 'ffprobe'

def setup_ffmpeg():
    global FFMPEG_PATH, FFPLAY_PATH, FFPROBE_PATH
    try:
        import ffmpeg_downloader as ffdl
        FFMPEG_PATH = ffdl.ffmpeg_path
        FFPROBE_PATH = ffdl.ffprobe_path
        FFPLAY_PATH = ffdl.ffplay_path
        print(f"Using ffmpeg from: {FFMPEG_PATH}")
        print(f"Using ffplay from: {FFPLAY_PATH}")
        print(f"Using ffprobe from: {FFPROBE_PATH}")
    except ImportError:
        print("ffmpeg-downloader not found. Using system ffmpeg.")
        print("Install it with: pip install ffmpeg-downloader")
    except Exception as e:
        print(f"Warning: Could not get ffmpeg paths from ffmpeg-downloader: {e}")
        print("Falling back to system ffmpeg.")

def get_char_set(name):
    return CHAR_SETS.get(name, CHAR_SETS[DEFAULT_CHAR_SET])

def extract_audio(video_path, output_path):
    try:
        cmd = [
            FFMPEG_PATH, '-i', video_path, 
            '-map', '0:a:0',  
            '-y',  
            '-vn',  
            '-acodec', 'copy',  
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"Warning: Could not extract audio: {e}")
        return False

def compress_file(input_path, output_path):
    try:
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
            zipf.write(input_path, os.path.basename(input_path))
        return True
    except Exception as e:
        print(f"Warning: Could not compress file: {e}")
        return False

def process_frame_direct(frame, width, height_ratio, char_set):
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(grayscale)
    new_height = int(width * height_ratio)
    resized = cv2.resize(enhanced, (width, new_height), interpolation=cv2.INTER_CUBIC)
    chars_array = np.array(list(char_set['chars']))
    scale_factor = 255.0 / (len(chars_array) - 1)
    indices = (resized / scale_factor).astype(np.uint8)
    indices = np.clip(indices, 0, len(chars_array) - 1)
    ascii_frame = chars_array[indices]
    return '\n'.join([''.join(row) for row in ascii_frame])

def write_frames_batch(output_path, ascii_frames, width):
    batch_output = []
    separator = "\n" + "=" * width + "\n"
    for ascii_art in ascii_frames:
        batch_output.append(ascii_art)
        batch_output.append(separator)
    with open(output_path, "a", encoding='utf-8') as f:
        f.write(''.join(batch_output))

def get_default_output_filename(input_path, char_set_name, compressed=False):
    base_name = os.path.splitext(input_path)[0]
    ext = ".ascii"  
    return f"{base_name}_{char_set_name}{ext}"

def get_optimal_ascii_dimensions(video_width, video_height):
    try:
        term_width, term_height = shutil.get_terminal_size()
        max_width = term_width - 1
        max_height = term_height - 1
        if max_width <= 0 or max_height <= 0:
            return 100, 50
        video_aspect = video_width / video_height
        if video_width >= video_height:
            ascii_width = min(max_width, 400)
            ascii_height = int(ascii_width / video_aspect / 2)
            if ascii_height > max_height:
                ascii_height = max_height
                ascii_width = int(ascii_height * video_aspect * 2)
        else:
            ascii_height = min(max_height, 100)
            ascii_width = int(ascii_height * video_aspect * 2)
            if ascii_width > max_width:
                ascii_width = max_width
                ascii_height = int(ascii_width / video_aspect / 2)
        ascii_width = max(20, ascii_width)
        ascii_height = max(10, ascii_height)
        return ascii_width, ascii_height
    except Exception as e:
        print(f"Warning: Could not determine optimal terminal size: {e}")
        return 100, 50

def video_to_ascii(video_path, output_path=None, ascii_width=None, batch_size=1000, 
                   extract_audio_flag=True, compress_flag=True, char_set_name=DEFAULT_CHAR_SET,
                   preserve_vfr=False, demo_all=False):
    start_time = time.time()
    if demo_all:
        char_sets_list = list(CHAR_SETS.keys()) * 2  
        print("Demo mode: Cycling through all character sets 2 times")
        print("Character sets order:", " -> ".join(char_sets_list))
    else:
        char_sets_list = [char_set_name]
    temp_dir = tempfile.mkdtemp()
    temp_ascii_path = os.path.join(temp_dir, "temp.ascii")
    temp_audio_path = os.path.join(temp_dir, "audio.aac")
    final_output_path = output_path
    if final_output_path is None:
        final_output_path = get_default_output_filename(video_path, char_set_name, compress_flag)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    is_image = any(video_path.lower().endswith(ext) for ext in image_extensions)
    if is_image:
        frame = cv2.imread(video_path)
        if frame is None:
            print("Error: Cannot open image file.")
            return
        fps = 1.0
        total_frames = 1
        video_width = frame.shape[1]
        video_height = frame.shape[0]
        cap = None  
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Cannot open video file.")
            return
    if not is_image:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        fps = 1.0
        total_frames = 1
    is_variable_fps = False
    frame_timestamps = []
    if not is_image and preserve_vfr and total_frames > 0:
        print("Analyzing frame timestamps for VFR support...")
        timestamps_available = True
        for i in range(min(100, total_frames)):  
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            if timestamp == 0 and i > 0:
                timestamps_available = False
                break
            cap.read()  
        if timestamps_available:
            is_variable_fps = True
            print("Variable frame rate detected")
        else:
            cap.release()
            cap = cv2.VideoCapture(video_path)  
    if not is_image and total_frames == 0:
        print("Error: Video file appears to be empty.")
        return
    if is_image:
        test_frame = frame  
    else:
        ret, test_frame = cap.read()
        if not ret:
            print("Error: Cannot read video file.")
            return
        cap.release()
        cap = cv2.VideoCapture(video_path)
    if ascii_width is None:
        calc_width, calc_height = get_optimal_ascii_dimensions(video_width, video_height)
        ascii_width = calc_width
        print(f"Auto-calculated ASCII dimensions: {ascii_width}x{calc_height} (video: {video_width}x{video_height})")
    else:
        height, aspect_ratio = test_frame.shape[0], test_frame.shape[1] / test_frame.shape[0]
        calc_height = int(ascii_width / aspect_ratio / 2)
        print(f"Using provided ASCII width: {ascii_width}, calculated height: {calc_height}")
    print(f"Output file: {final_output_path}")
    fps_display = f"{fps:.2f}" if not is_variable_fps else "Variable"
    print(f"Video FPS: {fps_display}, Total Frames: {total_frames}")
    print(f"Batch size: {batch_size} frames")
    height_ratio = test_frame.shape[0] / test_frame.shape[1] / 2
    audio_extracted = False
    if extract_audio_flag and not is_image:
        print("Extracting audio...")
        audio_extracted = extract_audio(video_path, temp_audio_path)
        if audio_extracted:
            print("Audio extracted successfully")
        else:
            print("Failed to extract audio")
    metadata = {
        'fps': fps,  
        'width': ascii_width,
        'height': int(ascii_width * height_ratio),
        'total_frames': total_frames,
        'video_width': video_width,
        'video_height': video_height,
        'has_audio': audio_extracted,
        'char_set': char_set_name if not demo_all else 'demo_all',
        'variable_fps': is_variable_fps,
        'timestamps': frame_timestamps if is_variable_fps else [],
        'demo_all': demo_all
    }
    with open(temp_ascii_path, "w", encoding='utf-8') as f:
        f.write(json.dumps(metadata) + '\n')
    frame_count = 0
    ascii_frames_buffer = []
    if demo_all:
        frames_per_charset = max(1, total_frames // len(char_sets_list))
        print(f"Each charset will be shown for approximately {frames_per_charset} frames")
    current_charset_index = 0
    frames_with_current_charset = 0
    char_set = get_char_set(char_sets_list[0]) if demo_all else get_char_set(char_set_name)
    if demo_all:
        print(f"Starting with charset: {char_set['name']}")
    if is_image:
        ascii_art = process_frame_direct(frame, ascii_width, height_ratio, char_set)
        ascii_frames_buffer.append(ascii_art)
        write_frames_batch(temp_ascii_path, ascii_frames_buffer, ascii_width)
        frame_count = 1
    else:
        with ThreadPoolExecutor(max_workers=4) as executor:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if demo_all:
                    if frames_with_current_charset >= frames_per_charset and current_charset_index < len(char_sets_list) - 1:
                        current_charset_index += 1
                        char_set = get_char_set(char_sets_list[current_charset_index])
                        frames_with_current_charset = 0
                        print(f"Switching to charset: {char_set['name']}")
                    frames_with_current_charset += 1
                if is_variable_fps:
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  
                    frame_timestamps.append(timestamp)
                frame_count += 1
                ascii_art = process_frame_direct(frame, ascii_width, height_ratio, char_set)
                ascii_frames_buffer.append(ascii_art)
                if len(ascii_frames_buffer) >= batch_size:
                    write_frames_batch(temp_ascii_path, ascii_frames_buffer, ascii_width)
                    ascii_frames_buffer.clear()
                    if frame_count % 100 == 0:
                        print(f"Processed {frame_count}/{total_frames} frames")
            if ascii_frames_buffer:
                write_frames_batch(temp_ascii_path, ascii_frames_buffer, ascii_width)
        cap.release()
    metadata['total_frames'] = frame_count
    if is_variable_fps:
        metadata['timestamps'] = frame_timestamps
    with open(temp_ascii_path, "r", encoding='utf-8') as f:
        content = f.read()
    lines = content.split('\n', 1)
    if len(lines) > 1:
        content = lines[1]  
        with open(temp_ascii_path, "w", encoding='utf-8') as f:
            f.write(json.dumps(metadata) + '\n' + content)
    print("Creating final output...")
    try:
        with zipfile.ZipFile(final_output_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
            zipf.write(temp_ascii_path, "video.ascii")
            if audio_extracted:
                zipf.write(temp_audio_path, "audio.aac")
    except Exception as e:
        print(f"Error creating compressed output: {e}")
        return
    try:
        os.remove(temp_ascii_path)
        if audio_extracted:
            os.remove(temp_audio_path)
        os.rmdir(temp_dir)
    except:
        pass
    total_time = time.time() - start_time
    file_size = os.path.getsize(final_output_path) / (1024 * 1024)  
    print(f"ASCII video saved to {final_output_path}")
    print(f"\n=== Conversion Statistics ===")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Frames processed: {frame_count}")
    print(f"Output file size: {file_size:.2f} MB")
    if total_time > 0:
        print(f"Processing speed: {frame_count/total_time:.1f} frames/second")

def main():
    setup_ffmpeg()
    parser = argparse.ArgumentParser(description='Convert images/video to ASCII art')
    parser.add_argument('input_file', help='Path to the input video file')
    parser.add_argument('-o', '--output', help='Output ASCII video file path (default: input_name_charset.ascii.zip)')
    parser.add_argument('-w', '--width', type=int, help='Width of ASCII output (default: auto-calculated based on terminal size)')
    parser.add_argument('-b', '--batch-size', type=int, default=1000, help='Batch size for processing (default: 1000)')
    parser.add_argument('--no-audio', action='store_true', help='Disable audio extraction')
    parser.add_argument('--no-compress', action='store_true', help='Disable compression')
    parser.add_argument('--vfr', action='store_true', help='Preserve variable frame rate (experimental)')
    parser.add_argument('-c', '--charset', choices=CHAR_SETS.keys(), default=DEFAULT_CHAR_SET, 
                        help=f'Character set to use (default: {DEFAULT_CHAR_SET})')
    parser.add_argument('--demo-all', action='store_true', 
                        help='Cycle through all character sets in order during conversion 2 times')
    args = parser.parse_args()
    if not os.path.exists(args.input_file):
        print(f"Error: File '{args.input_file}' not found")
        sys.exit(1)
    video_to_ascii(args.input_file, args.output, args.width, args.batch_size, 
                   not args.no_audio, not args.no_compress, args.charset, args.vfr, args.demo_all)

if __name__ == "__main__":
    main()