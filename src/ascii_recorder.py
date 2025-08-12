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
            os.path.join(script_dir, "CascadiaCode.ttf"),
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