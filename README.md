# img-vid-ascii: Next-Gen ASCII Art Transformation 🔄🎨

**Transform images and videos into stunning ASCII and unicode art and back** with speed speed and quality. This isn't just another converter - it's a complete toolkit that preserves audio, maintains quality, and delivers performance.

[Bad Apple!! PV](https://raw.githubusercontent.com/2dameneko/img-vid-ascii/samples/video/Bad_Apple_PV_demo.ascii_demo.mp4)  

## Why This Stands Out ✨

**🎭 Dual Media Magic**  
Convert **both images AND videos** seamlessly with a single utility. Your favorite images and clips become mesmerizing ASCII art in seconds.

[3_1](https://raw.githubusercontent.com/2dameneko/img-vid-ascii/samples/images/3_shades.ascii.jpg)

**💾 Ultra-Compact .ascii Format**  
Share creations that are **10-100x smaller** than originals! A 50MB video becomes a 1.5MB .ascii file - perfect for messaging and storage.

**🔊 Perfect Audio Preservation**  
Enjoy videos with **original, untouched audio quality**. Soundtracks remain crystal clear and perfectly synced.

**⏩ Lightning-Fast Processing**  
Experience **3-5× realtime conversion speeds**. GPU acceleration makes reconstruction back to mp4 blazing fast.

[1](https://raw.githubusercontent.com/2dameneko/img-vid-ascii/samples/images/1_blocks.ascii.jpg)

## Key Dependencies ⚙️

**OpenCV-Python**  
Handles core media processing - image/video loading, grayscale conversion, and smart contrast enhancement.

**FFmpeg-Downloader**  
Automatically installs and configures FFmpeg for seamless audio extraction and embedding without quality loss.

**Numba**  
Supercharges reconstruction with JIT compilation, delivering GPU-like speeds for ASCII-to-video conversion.

**Pillow**  
Powers professional font rendering during reconstruction, supporting custom character sets and anti-aliased text.

**🔄 Complete Round-Trip Workflow**  
```mermaid
graph LR
    A[Original] -->|2_convert_to_ascii| B[Tiny .ascii File]
    B -->|3_play_ascii| C[Terminal Playback]
    B -->|4_convert_from_ascii| D[Shareable Media]
```

**🎨 Unlimited Creativity**  
Choose from any character sets or create your own! Specialized fonts bring runic symbols and geometric shapes to life.

[5](https://raw.githubusercontent.com/2dameneko/img-vid-ascii/samples/images/5_blocks.ascii.jpg)

## Quick Start 🚀

1. **Install**  
   ```bash
   git clone https://github.com/2dameneko/img-vid-ascii
   cd img-vid-ascii
   0_install
   ```

2. **Convert & Play**  
   ```bash
   2_convert_to_ascii "your_video.mp4"
   3_play_ascii "your_video.ascii"
   ```

[Dance Animation](https://raw.githubusercontent.com/2dameneko/img-vid-ascii/samples/video/dance_mixer_standard.ascii_demo.mp4)  

3. **Share Everywhere**  
   ```bash
   convert_from_ascii "your_video.ascii"
   ```

**Share**  
Send the tiny .ascii file to friends - they'll see the full show with original audio!

[4](https://raw.githubusercontent.com2dameneko/img-vid-ascii/samples/images/4_blocks.ascii.jpg)

**Transform Back**  
Create a viral social media version:

## Technical Details ⚙️

**Audio Perfection**  
Original audio extracted without re-encoding.

**GPU-Powered Reconstruction**  
NVIDIA CUDA/AMD AMF/Intel QSV acceleration makes conversion fast.

**Smart Compression**  
Internal ZIP format with metadata

## Character Gallery 🖼️
[2](https://raw.githubusercontent.com/2dameneko/img-vid-ascii/samples/images/2_standard.ascii.jpg)
[6_2](https://raw.githubusercontent.com/2dameneko/img-vid-ascii/samples/images/6_standard.ascii.jpg)

## Video Showcase 🎥
[ZAKO](https://raw.githubusercontent.com/2dameneko/img-vid-ascii/samples/video/ZAKO_blocks.ascii_demo.mp4)  

## Join the ASCII Revolution! 💻➡️🎨
Rediscover your media through the beauty of ASCII!

## License
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

## Credits
* FFmpeg: [FFmpeg](https://ffmpeg.org/)
* Runic font: [BabelStone Runic](https://www.babelstone.co.uk/Fonts/Runic.html)
* Monospace font: [Inconsolata](https://github.com/googlefonts/Inconsolata)