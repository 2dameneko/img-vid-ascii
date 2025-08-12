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
