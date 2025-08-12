# File: ascii_converter.py
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
# File: ascii_player.py
import time
import os
import sys
import json
import argparse
import shutil
import zipfile
import tempfile
import subprocess
import select
try:
    import msvcrt  
except ImportError:
    msvcrt = None

FFPLAY_PATH = 'ffplay'

def setup_ffmpeg():
    global FFPLAY_PATH
    try:
        import ffmpeg_downloader as ffdl
        FFPLAY_PATH = ffdl.ffplay_path
        print(f"Using ffplay from: {FFPLAY_PATH}")
    except ImportError:
        print("ffmpeg-downloader not found. Using system ffplay.")
        print("Install it with: pip install ffmpeg-downloader")
    except Exception as e:
        print(f"Warning: Could not get ffplay path from ffmpeg-downloader: {e}")
        print("Falling back to system ffplay.")

class ASCIIVideoPlayer:
    def __init__(self, file_path, repeat=False):
        self.file_path = file_path
        self.repeat = repeat
        self.paused = False
        self.audio_process = None
        self.temp_dir = None
        self.audio_file = None
        self.fps = 30.0  
        self.pause_start_time = None
        self.total_pause_time = 0
        self.audio_start_time = None
        self.char_set_name = 'standard'
        self.is_variable_fps = False
        self.frame_timestamps = []  
        self.start_playback_time = None
        self.last_frame_time = 0
        self.is_single_frame = False  

    def extract_files(self):
        try:
            self.temp_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(self.file_path, 'r') as zipf:
                zipf.extractall(self.temp_dir)
            ascii_file = os.path.join(self.temp_dir, "video.ascii")
            if not os.path.exists(ascii_file):
                for file in os.listdir(self.temp_dir):
                    if file.endswith('.ascii'):
                        ascii_file = os.path.join(self.temp_dir, file)
                        break
            audio_file = os.path.join(self.temp_dir, "audio.aac")
            if not os.path.exists(audio_file):
                for ext in ['mp3', 'wav', 'flac', 'm4a']:
                    audio_candidate = os.path.join(self.temp_dir, f"audio.{ext}")
                    if os.path.exists(audio_candidate):
                        audio_file = audio_candidate
                        break
                else:
                    audio_file = None
            return ascii_file, audio_file
        except Exception as e:
            print(f"Error extracting files: {e}")
            return None, None

    def play_audio(self, start_time=0):
        if not self.audio_file:
            return
        try:
            cmd = [FFPLAY_PATH, '-nodisp', '-autoexit', '-ss', str(start_time), self.audio_file]
            self.audio_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.audio_start_time = time.time() - start_time
        except Exception as e:
            print(f"Warning: Could not play audio: {e}")

    def stop_audio_playback(self):
        if self.audio_process and self.audio_process.poll() is None:
            self.audio_process.terminate()
            try:
                self.audio_process.wait(timeout=1)
            except:
                self.audio_process.kill()
        self.audio_process = None

    def get_audio_position(self):
        if self.audio_start_time is None:
            return 0
        return time.time() - self.audio_start_time

    def toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self.stop_audio_playback()
            self.pause_start_time = time.time()
            print("\n[PAUSED] Press SPACE to resume, Ctrl+C to stop")
        else:
            paused_duration = time.time() - self.pause_start_time
            self.total_pause_time += paused_duration
            audio_position = self.get_audio_position()
            self.play_audio(audio_position)
            self.pause_start_time = None
            print("\n[RESUMED]")

    def cleanup(self):
        self.stop_audio_playback()
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
            except:
                pass

    def check_for_keypress(self):
        try:
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                char = sys.stdin.read(1)
                return char
        except:
            pass
        if msvcrt and msvcrt.kbhit():
            try:
                char = msvcrt.getch().decode('utf-8')
                return char
            except:
                pass
        return None

    def wait_for_keypress(self):
        print("\nPress any key to exit...")
        try:
            import tty, termios
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            try:
                if select.select([sys.stdin], [], [], None) == ([sys.stdin], [], []):
                    sys.stdin.read(1)
            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        except:
            if msvcrt:
                msvcrt.getch()
            else:
                input()

    def get_current_frame_vfr(self, elapsed_time):
        if not self.frame_timestamps:
            return 0
        target_time = elapsed_time
        left, right = 0, len(self.frame_timestamps) - 1
        while left <= right:
            mid = (left + right) // 2
            if self.frame_timestamps[mid] <= target_time:
                if mid == len(self.frame_timestamps) - 1 or self.frame_timestamps[mid + 1] > target_time:
                    return mid
                left = mid + 1
            else:
                right = mid - 1
        for i in range(len(self.frame_timestamps) - 1, -1, -1):
            if self.frame_timestamps[i] <= target_time:
                return i
        return 0

    def play(self):
        ascii_file, self.audio_file = self.extract_files()
        if not ascii_file:
            print("Error: Could not extract video file")
            return
        try:
            with open(ascii_file, "r", encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading ASCII file: {e}")
            self.cleanup()
            return
        lines = content.split('\n', 1)
        if len(lines) < 2:
            print("Error: Invalid file format")
            self.cleanup()
            return
        try:
            metadata = json.loads(lines[0])
            self.fps = float(metadata.get('fps', 30.0))  
            frame_width = metadata.get('width', 100)
            frame_height = metadata.get('height', 50)
            self.char_set_name = metadata.get('char_set', 'standard')
            self.is_variable_fps = metadata.get('variable_fps', False)
            self.frame_timestamps = metadata.get('timestamps', [])
            total_frames = metadata.get('total_frames', 1)
            content = lines[1]
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error: Invalid metadata in file: {e}")
            self.cleanup()
            return
        self.is_single_frame = (total_frames == 1)
        frames = content.split("=" * frame_width)
        frames = [frame for frame in frames if frame.strip()]
        if len(frames) != total_frames and total_frames > 1:
            print(f"Warning: Expected {total_frames} frames, found {len(frames)}")
        fps_display = f"{self.fps:.2f}" if not self.is_variable_fps else "Variable"
        print(f"Total Frames: {total_frames}, FPS: {fps_display}, Resolution: {frame_width}x{frame_height}")
        print(f"Character Set: {self.char_set_name}")
        if self.audio_file:
            print("Audio track detected")
        if self.is_variable_fps and self.frame_timestamps:
            print(f"Variable frame rate with {len(self.frame_timestamps)} timestamps")
        if self.is_single_frame:
            print("Single-frame image detected")
        try:
            term_width, term_height = shutil.get_terminal_size()
            if term_width < frame_width or term_height < frame_height:
                print(f"Warning: Terminal size ({term_width}x{term_height}) is smaller than video resolution ({frame_width}x{frame_height})")
        except:
            pass
        try:
            os.system("tput civis")
        except:
            pass
        if self.is_single_frame and frames:
            try:
                print("Starting playback... (Press any key to exit)")
                sys.stdout.write("\033[2J\033[H")
                sys.stdout.flush()
                frame = frames[0]
                sys.stdout.write(f"\033[H{frame}")
                sys.stdout.flush()
                self.wait_for_keypress()
            except KeyboardInterrupt:
                print("\nPlayback stopped.")
            except Exception as e:
                print(f"\nError during playback: {e}")
            finally:
                self.cleanup()
                try:
                    os.system("tput cnorm")
                except:
                    pass
                sys.stdout.write("\033[2J\033[H")
                sys.stdout.flush()
            return
        if self.audio_file:
            self.play_audio()
        try:
            print("Starting playback... (Press SPACE to pause/resume, Ctrl+C to stop)")
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()
            frame_index = 0
            self.start_playback_time = time.time()
            self.total_pause_time = 0
            while True:
                if frame_index >= len(frames):
                    if self.repeat:
                        frame_index = 0
                        self.start_playback_time = time.time()
                        self.total_pause_time = 0
                        if self.audio_file and not self.paused:
                            self.stop_audio_playback()
                            self.play_audio()
                    else:
                        break
                if self.paused:
                    key = self.check_for_keypress()
                    if key == ' ':
                        self.toggle_pause()
                    time.sleep(0.1)
                    continue
                key = self.check_for_keypress()
                if key == ' ':
                    self.toggle_pause()
                    continue
                elif key is not None and (ord(key) == 3 or key.lower() == 'q'):  
                    raise KeyboardInterrupt()
                elapsed_time = (time.time() - self.start_playback_time) - self.total_pause_time
                if self.is_variable_fps and self.frame_timestamps:
                    frame_index = self.get_current_frame_vfr(elapsed_time)
                else:
                    expected_frame = int(elapsed_time * self.fps)
                    if expected_frame > frame_index:
                        frame_index = expected_frame
                if frame_index >= len(frames):
                    if not self.repeat:
                        break
                    else:
                        continue
                frame = frames[frame_index]
                sys.stdout.write(f"\033[H{frame}")
                sys.stdout.flush()
                if self.is_variable_fps and self.frame_timestamps and frame_index < len(self.frame_timestamps) - 1:
                    if frame_index + 1 < len(self.frame_timestamps):
                        next_timestamp = self.frame_timestamps[frame_index + 1]
                        current_timestamp = self.frame_timestamps[frame_index]
                        frame_duration = next_timestamp - current_timestamp
                        sleep_time = max(0, frame_duration - 0.001)  
                    else:
                        sleep_time = 1.0 / self.fps if self.fps > 0 else 0.033
                else:
                    next_frame_time = self.start_playback_time + (frame_index + 1) / self.fps + self.total_pause_time
                    sleep_time = next_frame_time - time.time()
                if sleep_time > 0:
                    time.sleep(min(sleep_time, 0.1))
                frame_index += 1
        except KeyboardInterrupt:
            print("\nPlayback stopped.")
        except Exception as e:
            print(f"\nError during playback: {e}")
        finally:
            self.cleanup()
            try:
                os.system("tput cnorm")
            except:
                pass
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()

def main():
    setup_ffmpeg()
    parser = argparse.ArgumentParser(description='Play ASCII images/video file')
    parser.add_argument('input_file', help='Path to the ASCII image/video file')
    parser.add_argument('--repeat', action='store_true', help='Repeat playback endlessly')
    args = parser.parse_args()
    if not os.path.exists(args.input_file):
        print(f"Error: File '{args.input_file}' not found")
        sys.exit(1)
    try:
        import tty, termios
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
    except:
        old_settings = None
    player = ASCIIVideoPlayer(args.input_file, args.repeat)
    try:
        player.play()
    finally:
        if old_settings:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            except:
                pass

if __name__ == "__main__":
    main()

# File: ascii_recorder.py
import os
import sys
import json
import argparse
import shutil
import zipfile
import tempfile
import subprocess
import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor
import time
from PIL import Image, ImageDraw, ImageFont
from numba import jit, prange
import multiprocessing as mp
from functools import partial

FFMPEG_PATH = 'ffmpeg'

def setup_ffmpeg():
    global FFMPEG_PATH
    try:
        import ffmpeg_downloader as ffdl
        FFMPEG_PATH = ffdl.ffmpeg_path
        print(f"Using ffmpeg from: {FFMPEG_PATH}")
    except ImportError:
        print("ffmpeg-downloader not found. Using system ffmpeg.")
        print("Install it with: pip install ffmpeg-downloader")
    except Exception as e:
        print(f"Warning: Could not get ffmpeg path from ffmpeg-downloader: {e}")
        print("Falling back to system ffmpeg.")

@jit(nopython=True, fastmath=True, parallel=True)
def render_chars_batch_numba(char_indices, font_array_flat, char_height, char_width, height_chars, width_chars):
    batch_size = char_indices.shape[0]
    img_height = height_chars * char_height
    img_width = width_chars * char_width
    img = np.zeros((batch_size, img_height, img_width, 3), dtype=np.uint8)
    for b in prange(batch_size):
        for y in range(height_chars):
            for x in range(width_chars):
                char_idx = char_indices[b, y, x]
                if char_idx >= 0:  
                    y_start = y * char_height
                    y_end = y_start + char_height
                    x_start = x * char_width
                    x_end = x_start + char_width
                    char_start = char_idx * char_height * char_width * 3
                    char_end = char_start + char_height * char_width * 3
                    char_data = font_array_flat[char_start:char_end].reshape(char_height, char_width, 3)
                    img[b, y_start:y_end, x_start:x_end] = char_data
    return img

def is_runic_char(char):
    code_point = ord(char)
    return 0x16A0 <= code_point <= 0x16FF

def prepare_frame_batch_standalone(batch_frames, width_chars, height_chars, char_to_idx):
    batch_size = len(batch_frames)
    char_indices = np.full((batch_size, height_chars, width_chars), -1, dtype=np.int32)
    for i, frame_text in enumerate(batch_frames):
        lines = frame_text.split('\n')[:height_chars]
        for y, line in enumerate(lines):
            line = (line[:width_chars] if len(line) > width_chars else line.ljust(width_chars))
            for x, char in enumerate(line):
                if char in char_to_idx:
                    char_indices[i, y, x] = char_to_idx[char]
    return char_indices

def can_render_rune(font, rune='ᚠ'):  # 'ᚠ' (Fehu) is a common test rune
    try:
        if hasattr(font, 'getbbox'):
            bbox = font.getbbox(rune)
            # If width or height > 0, it's likely rendered
            return bbox[2] - bbox[0] > 0 and bbox[3] - bbox[1] > 0
        else:
            # Fallback for older PIL
            return font.getsize(rune)[0] > 0
    except Exception:
        return False
    
class ASCIIVideoRecorder:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.temp_dir = tempfile.mkdtemp()
        self.audio_file = None
        self.fps = 30.0
        self.char_width_px = 10
        self.char_height_px = 20
        self.video_writer = None
        self.font_array = None
        self.char_to_idx = None
        self.font_array_flat = None
        self.max_width = 1920
        self.max_height = 1080
        self.output_width = 0
        self.output_height = 0
        self.scale_factor = 1.0
        self.frame_timestamps = []
        self.is_variable_fps = False
        self.char_set = None
        self.is_demo_mode = False
        self.cpu_count = min(mp.cpu_count(), 12)  
        self.frame_width = 0
        self.frame_height = 0
        self.gpu_encoder = self.detect_gpu_encoder()

    def detect_gpu_encoder(self):
        try:
            subprocess.run([FFMPEG_PATH, '-version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("FFmpeg not found, using CPU encoding")
            return None
        encoders_to_check = [
            ('h264_nvenc', 'NVIDIA NVENC'),
            ('h264_amf', 'AMD AMF'),
            ('h264_qsv', 'Intel Quick Sync'),
            ('h264_videotoolbox', 'Apple VideoToolbox'),
        ]
        for encoder, name in encoders_to_check:
            try:
                cmd = [
                    FFMPEG_PATH, '-f', 'lavfi', '-i', 'nullsrc',
                    '-c:v', encoder, 
                    '-frames:v', '1', 
                    '-f', 'null', '-'
                ]
                result = subprocess.run(cmd, capture_output=True, timeout=10)
                if result.returncode == 0:
                    print(f"Using GPU encoder: {name} ({encoder})")
                    return encoder
                else:
                    pass
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                continue
            except Exception as e:
                continue
        print("No GPU encoder found, using CPU encoding (libx264)")
        return None

    def get_ffmpeg_quality_settings(self):
        if self.gpu_encoder:
            if 'nvenc' in self.gpu_encoder:
                return [
                    '-c:v', self.gpu_encoder,
                    '-preset', 'p7',  
                    '-qp', '35',  
                    '-profile:v', 'main'
                ]
            elif 'amf' in self.gpu_encoder:
                return [
                    '-c:v', self.gpu_encoder,
                    '-quality', 'quality',  
                    '-qp_i', '31',
                    '-qp_p', '33',
                    '-qp_b', '35',
                    '-profile:v', 'main'
                ]
            elif 'qsv' in self.gpu_encoder:
                return [
                    '-c:v', self.gpu_encoder,
                    '-preset', 'veryslow',  
                    '-global_quality', '35',
                    '-profile:v', 'main'
                ]
            elif 'videotoolbox' in self.gpu_encoder:
                return [
                    '-c:v', self.gpu_encoder,
                    '-profile:v', 'main',
                    '-q:v', '75'  
                ]
        return [
            '-c:v', 'libx264',
            '-crf', '35',
            '-profile:v', 'main',
            '-tune', 'animation',
            '-preset', 'slow'
        ]

    def extract_files(self):
        try:
            with zipfile.ZipFile(self.input_path, 'r') as zipf:
                zipf.extractall(self.temp_dir)
            ascii_file = os.path.join(self.temp_dir, "video.ascii")
            if not os.path.exists(ascii_file):
                for file in os.listdir(self.temp_dir):
                    if file.endswith('.ascii'):
                        ascii_file = os.path.join(self.temp_dir, file)
                        break
            audio_file = os.path.join(self.temp_dir, "audio.aac")
            if not os.path.exists(audio_file):
                for ext in ['mp3', 'wav', 'flac', 'm4a']:
                    audio_candidate = os.path.join(self.temp_dir, f"audio.{ext}")
                    if os.path.exists(audio_candidate):
                        audio_file = audio_candidate
                        break
                else:
                    audio_file = None
            return ascii_file, audio_file
        except Exception as e:
            print(f"Error extracting files: {e}")
            return None, None

    def create_font_array(self):
        if not self.char_set:
            print("Error: Character set not loaded")
            return False
        unique_chars = set(self.char_set)
        print(f"Creating font array for {len(unique_chars)} unique characters")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        potential_paths_cascadia = [
            os.path.join(script_dir, "Inconsolata-Regular.ttf"),
        ]
        potential_paths_segoe_runic = [
            os.path.join(script_dir, "BabelStoneRunic.ttf"),
        ]
        font_cascadia = None
        font_segoe_runic = None
        for path in potential_paths_cascadia:
            try:
                font_cascadia = ImageFont.truetype(path, int(self.char_height_px * 0.8))
                break
            except OSError:
                continue
        if font_cascadia is None:
            font_cascadia = ImageFont.load_default()
        for path in potential_paths_segoe_runic:
            try:
                font_segoe_runic = ImageFont.truetype(path, int(self.char_height_px * 0.8))
                break
            except OSError:
                continue
        if font_segoe_runic is None:
            font_segoe_runic = font_cascadia 
        
        if any(is_runic_char(char) for char in unique_chars):
            supports_runes = False
            supports_runes = can_render_rune(font_segoe_runic)
            print(f"Can use rune: {supports_runes}")

        chars = {}
        chars[' '] = np.zeros((self.char_height_px, self.char_width_px, 3), dtype=np.uint8)
        char_images = np.zeros((len(unique_chars), self.char_height_px, self.char_width_px, 3), dtype=np.uint8)
        char_mapping = {}
        for idx, char in enumerate(unique_chars):
            if char == ' ':
                char_mapping[char] = idx
                continue
            if is_runic_char(char):
                font_to_use = font_segoe_runic
            else:
                font_to_use = font_cascadia
            img_bgr = np.zeros((self.char_height_px, self.char_width_px, 3), dtype=np.uint8)
            pil_image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            try:
                try:
                    bbox = font_to_use.getbbox(char)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                except AttributeError:
                    try:
                        text_size = draw.textsize(char, font=font_to_use)
                        text_width, text_height = text_size
                    except:
                        text_width, text_height = self.char_width_px // 2, self.char_height_px // 2 
                x_offset = max(0, (self.char_width_px - text_width) // 2)
                y_offset = max(0, (self.char_height_px - text_height) // 2)
                draw.text((x_offset, y_offset), char, font=font_to_use, fill=(255, 255, 255))
                char_img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                char_images[idx] = char_img_bgr
                char_mapping[char] = idx
            except Exception as e:
                print(f"Warning: Could not render character '{repr(char)}': {e}")
                char_images[idx] = np.full((self.char_height_px, self.char_width_px, 3), 255, dtype=np.uint8)
                char_mapping[char] = idx
        self.font_array = char_images
        self.char_to_idx = char_mapping
        self.font_array_flat = self.font_array.reshape(-1)
        return True

    def calculate_output_dimensions(self, width_chars, height_chars):
        raw_width = width_chars * self.char_width_px
        raw_height = height_chars * self.char_height_px
        width_ratio = self.max_width / raw_width
        height_ratio = self.max_height / raw_height
        self.scale_factor = min(width_ratio, height_ratio, 1.0)
        self.output_width = int(raw_width * self.scale_factor)
        self.output_height = int(raw_height * self.scale_factor)
        self.output_width = (self.output_width // 2) * 2
        self.output_height = (self.output_height // 2) * 2
        return self.output_width, self.output_height

    def init_video_recording(self, width_chars, height_chars):
        video_width, video_height = self.calculate_output_dimensions(width_chars, height_chars)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            os.path.join(self.temp_dir, "temp_recording.mp4"), 
            fourcc, 
            self.fps,
            (video_width, video_height)
        )
        if not self.video_writer.isOpened():
            print("Error: Failed to initialize video writer")
            return False
        fps_display = f"{self.fps:.2f}" if not self.is_variable_fps else "Variable"
        print(f"Recording initialized: {video_width}x{video_height} @ {fps_display} FPS")
        return True
    
    def render_frame_batch_optimized(self, char_indices):
        if len(char_indices) == 0:
            return np.array([])
        try:
            rendered_frames = render_chars_batch_numba(
                char_indices, 
                self.font_array_flat, 
                self.char_height_px, 
                self.char_width_px, 
                char_indices.shape[1], 
                char_indices.shape[2]
            )
            return rendered_frames
        except Exception as e:
            print(f"Numba rendering failed, falling back to CPU: {e}")
            return self.render_frame_batch_cpu(char_indices)

    def render_frame_batch_cpu(self, char_indices):
        batch_size, height_chars, width_chars = char_indices.shape
        img_height = height_chars * self.char_height_px
        img_width = width_chars * self.char_width_px
        rendered_frames = np.zeros((batch_size, img_height, img_width, 3), dtype=np.uint8)
        for b in range(batch_size):
            for y in range(height_chars):
                for x in range(width_chars):
                    char_idx = char_indices[b, y, x]
                    if char_idx >= 0:
                        y_start = y * self.char_height_px
                        y_end = y_start + self.char_height_px
                        x_start = x * self.char_width_px
                        x_end = x_start + self.char_width_px
                        rendered_frames[b, y_start:y_end, x_start:x_end] = self.font_array[char_idx]
        return rendered_frames

    def finalize_recording_vfr(self):
        temp_video = os.path.join(self.temp_dir, "temp_recording.mp4")
        temp_timestamps = os.path.join(self.temp_dir, "timestamps.txt")
        if not os.path.exists(temp_video):
            print("Error: Temporary video file not found")
            return
        try:
            with open(temp_timestamps, 'w') as f:
                for i, timestamp in enumerate(self.frame_timestamps):
                    f.write(f"{i},{timestamp:.6f}\n")
        except Exception as e:
            print(f"Error writing timestamps: {e}")
            return
        if self.audio_file:
            print("Creating variable frame rate video with audio...")
            try:
                temp_vfr_video = os.path.join(self.temp_dir, "vfr_video.mp4")
                quality_settings = self.get_ffmpeg_quality_settings()
                cmd = [
                    FFMPEG_PATH, '-y',
                    '-i', temp_video,
                    '-vf', 'setpts=N/TB/30',
                    '-vsync', '0'
                ] + quality_settings + [
                    temp_vfr_video
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                cmd = [
                    FFMPEG_PATH, '-y',
                    '-i', temp_vfr_video,
                    '-i', self.audio_file,
                    '-c:v', 'copy',
                    '-c:a', 'copy',
                    '-map', '0:v:0',
                    '-map', '1:a:0',
                    '-shortest',
                    self.output_path
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"Variable frame rate video with audio saved to {self.output_path}")
            except Exception as e:
                print(f"Error creating VFR video with audio: {e}")
                self.finalize_recording_cfr()
        else:
            try:
                quality_settings = self.get_ffmpeg_quality_settings()
                cmd = [
                    FFMPEG_PATH, '-y',
                    '-i', temp_video,
                    '-vf', 'setpts=N/TB/30',
                    '-vsync', '0'
                ] + quality_settings + [
                    self.output_path
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"Variable frame rate video saved to {self.output_path}")
            except Exception as e:
                print(f"Error creating VFR video: {e}")
                self.finalize_recording_cfr()

    def finalize_recording_cfr(self):
        temp_video = os.path.join(self.temp_dir, "temp_recording.mp4")
        if not os.path.exists(temp_video):
            print("Error: Temporary video file not found")
            return
        if self.audio_file:
            print("Muxing audio with video (high quality)...")
            try:
                quality_settings = self.get_ffmpeg_quality_settings()
                cmd = [
                    FFMPEG_PATH, '-y',
                    '-i', temp_video,
                    '-i', self.audio_file
                ] + quality_settings + [
                    '-c:a', 'copy',
                    '-map', '0:v:0',
                    '-map', '1:a:0',
                    '-shortest',
                    self.output_path
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"High quality video with audio saved to {self.output_path}")
            except Exception as e:
                print(f"Error muxing audio: {e}. Saving video without audio.")
                try:
                    quality_settings = self.get_ffmpeg_quality_settings()
                    cmd = [
                        FFMPEG_PATH, '-y',
                        '-i', temp_video
                    ] + quality_settings + [
                        self.output_path
                    ]
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    print(f"High quality video saved to {self.output_path}")
                except:
                    shutil.copy(temp_video, self.output_path)
                    print(f"Video saved to {self.output_path}")
        else:
            try:
                quality_settings = self.get_ffmpeg_quality_settings()
                cmd = [
                    FFMPEG_PATH, '-y',
                    '-i', temp_video
                ] + quality_settings + [
                    self.output_path
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"High quality video saved to {self.output_path}")
            except:
                shutil.copy(temp_video, self.output_path)
                print(f"Video saved to {self.output_path}")

    def finalize_recording(self):
        if self.video_writer:
            self.video_writer.release()
        if self.total_frames == 1:  
            if not self.output_path.endswith(('.png', '.jpg', '.jpeg')):
                if self.output_path.endswith('.ascii'):
                    img_path = self.output_path + ".png"
                else:
                    base_name = os.path.splitext(self.output_path)[0]
                    img_path = base_name + ".png"
            else:
                img_path = self.output_path
            try:
                temp_video = os.path.join(self.temp_dir, "temp_recording.mp4")
                cap = cv2.VideoCapture(temp_video)
                ret, frame = cap.read()
                if ret:
                    cv2.imwrite(img_path, frame)
                    print(f"Image saved to {img_path}")
                cap.release()
            except Exception as e:
                print(f"Error saving image: {e}")
        elif self.is_variable_fps and len(self.frame_timestamps) > 0:
            self.finalize_recording_vfr()
        else:
            self.finalize_recording_cfr()

    def cleanup(self):
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except:
                pass

    def record(self):
        ascii_file, self.audio_file = self.extract_files()
        if not ascii_file:
            print("Error: Could not extract video file")
            return False
        try:
            with open(ascii_file, "r", encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading ASCII file: {e}")
            self.cleanup()
            return False
        lines = content.split('\n', 1)
        if len(lines) < 2:
            print("Error: Invalid file format")
            self.cleanup()
            return False
        try:
            metadata = json.loads(lines[0])
            self.fps = float(metadata.get('fps', 30.0))
            self.frame_width = metadata.get('width', 100)
            self.frame_height = metadata.get('height', 50)
            self.is_variable_fps = metadata.get('variable_fps', False)
            self.frame_timestamps = metadata.get('timestamps', [])
            self.is_demo_mode = metadata.get('demo_all', False)
            self.char_set = metadata.get('char_set', None)
            content = lines[1]
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error: Invalid metadata in file: {e}")
            self.cleanup()
            return False
        expected_count = metadata['total_frames']
        separator = "=" * self.frame_width
        parts = content.split(separator)
        frames = [p for p in parts if p != '']
        if len(frames) >= expected_count:
            frames = frames[:expected_count]
        else:
            print(f"Warning: Only {len(frames)} frames found, padding to {expected_count}")
            blank_line = ' ' * self.frame_width
            blank_frame = '\n'.join([blank_line] * self.frame_height)
            frames += [blank_frame] * (expected_count - len(frames))
        self.total_frames = len(frames)
        total_frames = self.total_frames
        fps_display = f"{self.fps:.2f}" if not self.is_variable_fps else "Variable"
        print(f"Total Frames: {total_frames}, FPS: {fps_display}, Resolution: {self.frame_width}x{self.frame_height}")
        print(f"Using charset: {self.char_set}")
        print("Extracting character set from frames...")
        all_chars = set()
        sample_size = len(frames)
        sample_frames = frames[:sample_size]
        print(f"All charset demo detected: {'Yes' if self.is_demo_mode else 'No'}. Sampling {sample_size} frames.")
        for frame in sample_frames:
            all_chars.update(frame)
        self.char_set = ''.join(sorted(all_chars, key=ord))
        print(f"Extracted {len(self.char_set)} unique characters")
        print("Initializing high-quality font rendering...")
        if not self.create_font_array():
            print("Error: Could not create font array")
            self.cleanup()
            return False
        if self.font_array is None:
            print("Error: Could not create font array")
            self.cleanup()
            return False
        if self.total_frames == 1:
            if not self.output_path.endswith(('.png', '.jpg', '.jpeg')):
                if self.output_path.endswith('.ascii'):
                    self.output_path = self.output_path + ".png"
                else:
                    base_name = os.path.splitext(self.output_path)[0]
                    self.output_path = base_name + ".png"
        else:
            if not self.output_path.endswith(('.mp4', '.avi', '.mkv', '.mov')):
                if self.output_path.endswith('.ascii'):
                    self.output_path = self.output_path + ".mp4"
                else:
                    base_name = os.path.splitext(self.output_path)[0]
                    self.output_path = base_name + ".mp4"
        if not self.init_video_recording(self.frame_width, self.frame_height):
            self.cleanup()
            return False
        print("Recording started (high quality)...")
        start_time = time.time()
        batch_size = min(50, max(30, self.cpu_count * 10))  
        process_batch_size = min(10, max(5, self.cpu_count))  
        for i in range(0, total_frames, batch_size * process_batch_size):
            batch_end = min(i + batch_size * process_batch_size, total_frames)
            large_batch_frames = frames[i:batch_end]
            sub_batches = [large_batch_frames[j:j+batch_size] 
                          for j in range(0, len(large_batch_frames), batch_size)]
            prepare_func = partial(prepare_frame_batch_standalone, 
                                 width_chars=self.frame_width, 
                                 height_chars=self.frame_height,
                                 char_to_idx=self.char_to_idx)
            with ProcessPoolExecutor(max_workers=self.cpu_count) as process_executor:
                char_indices_batches = list(process_executor.map(prepare_func, sub_batches))
                rendered_batches = []
                for char_indices_batch in char_indices_batches:
                    rendered_batch = self.render_frame_batch_optimized(char_indices_batch)
                    rendered_batches.append(rendered_batch)
            frame_counter = 0
            for rendered_batch in rendered_batches:
                if rendered_batch.size == 0:
                    continue
                for img in rendered_batch:
                    if abs(self.scale_factor - 1.0) > 0.01 or \
                       (img.shape[1] != self.output_width or img.shape[0] != self.output_height):
                        if self.output_width > 0 and self.output_height > 0:
                            img = cv2.resize(img, (self.output_width, self.output_height), 
                                           interpolation=cv2.INTER_CUBIC)
                    self.video_writer.write(img)
                    frame_counter += 1
            current_frame = i + len(large_batch_frames)
            if current_frame % (batch_size * 5) == 0 or current_frame == total_frames:
                elapsed = time.time() - start_time
                percent = current_frame / total_frames * 100
                print(f"Recorded {current_frame}/{total_frames} frames ({percent:.1f}%) - {elapsed:.1f}s elapsed")
        self.finalize_recording()
        self.cleanup()
        return True

def main():
    setup_ffmpeg()
    parser = argparse.ArgumentParser(description='Record ASCII image/video to PNG/MP4 (High Quality)')
    parser.add_argument('input_file', help='Path to the ASCII image/video file')
    parser.add_argument('output_file', nargs='?', help='Output file path (optional)')
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: File '{args.input_file}' not found")
        sys.exit(1)
    
    if args.output_file is None:
        output_file = args.input_file
    else:
        output_file = args.output_file
    
    recorder = ASCIIVideoRecorder(args.input_file, output_file)
    success = recorder.record()
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
# File: charsets.py
"""Predefined character sets for ASCII art conversion."""

CHAR_SETS = {
    'standard': {
        'chars': " .:-=+*#%@",
        'name': 'Standard ASCII'
    },
    'standard2': {
        'chars': " ?=#&0%@",
        'name': 'Standard2 ASCII'
    },
    'standard3': {
        'chars': " ';-~|}/+=",
        'name': 'Standard3 ASCII'
    },
    'standard4': {
        'chars': " _*!~)(+^#&$%@",
        'name': 'Standard4 ASCII'
    },
    'standard5': {
        'chars': " `-~+#@",
        'name': 'Standard5 ASCII'
    },
    'standard6': {
        'chars': " ¨'³•µðEÆ",
        'name': 'Standard6 ASCII'
    },
    'standard7': {
        'chars': " `.,-:~;+*#%$@",
        'name': 'Standard7 ASCII'
    },    
    'standard_alt': {
        'chars': " .,:ilwW",
        'name': 'Standard ASCII Alternative'
    },
    'complex': {
        'chars': " `.',:^\";*!²¤/r(?+¿cLª7t1fJCÝy¢zF3±%S2kñ5AZXG$À0Ãm&Q8#RÔßÊNBåMÆØ@¶",
        'name': 'Complex ASCII '
    },
    'complex_alt': {
        'chars': " `.-':_,^=;><+!rc*/z?sLTv)J7(|Fi\{C\}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@",
        'name': 'Complex ASCII Alternative'
    },
    'fine': {
        'chars': " `^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$",
        'name': 'Fine Detail ASCII'
    },
    'runic': {
        'chars': " ᛫ᛌᚲ᛬ᛍᛵᛁᛊᚿᚽᚳᚪᚮᚩᚰᚨᛏᚠᚬᚧᛪᚫᚷᛱᚱᚢᛒᚤᛄᚣᚻᛖᛰᚸᚥᛞᛤᛥ",
        'name': 'Runic'
    },
    'box': {
        'chars': " ╶╴┈┄╌─┊╺┆└╸╭╰┉┬│╾┅┍┕┭━╘┎╱╲┖┰┯┋┸┇┞┗╙┱╧╀┹┢┡┳╅┻╃┃╚┠╈╇╳╂╦┣╩╉║╋╫╠╬",
        'name': 'Box Drawings'
    },
    'blocks': {
        'chars': " ▏▎▍▌▋▊▉█",
        'name': 'Block Elements '
    },
    'blocks_alt': {
        'chars': "  ▏▁░▂▖▃▍▐▒▀▞▚▌▅▆▊▓▇▉█",
        'name': 'Block Elements Alternative'
    },
    'geo': {
        'chars': " ◜◞◟◦◃◠▿▹▱◌▵◅▭▸◁△◹▽▫▷▯□◯◄▰◫◊◮◎◈◖◭◗▬◤▪▼◑◍▮◒◐▤◉▧▨◕◛ Cuomo▣▦●▩■◘◙",
        'name': 'Geometric Shapes'
    },
    'shades': {
        'chars': " ░▒▓█",
        'name': 'Shaded Blocks'
    },
    'shades_alt': {
        'chars': " ░░░▒▒▒▓▓▓███",
        'name': 'Shaded Blocks Alternative'
    },
    'shades_mix': {
        'chars': " .░▒▓█",
        'name': 'Mixed Shaded Blocks'
    },
}

DEFAULT_CHAR_SET = 'standard'
