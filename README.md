# terminal-raymarcher

Zero-dependency, multithreaded real-time terminal raymarcher and plasma engine written in pure C11 and POSIX threads (pthreads).

![terminal-raymarcher](images/neon-marcher-avx2.png)

Instead of outputting a static image or requiring external graphics libraries like `SDL2` or `Raylib`, it hijacks your terminal using raw ANSI escape sequences and Unicode half-block characters (▀) to double the vertical resolution—turning any standard terminal emulator into a 60 FPS true-color (24-bit RGB) 3D viewport.

## What Makes This Engine Tick

- **Sub-Pixel Terminal Rendering**: Every terminal cell contains two vertical pixels by pairing the foreground color (\033[38;2;R;G;Bm) with the upper half-block character ▀ and the background color (\033[48;2;R;G;Bm) for the lower half.
- **Lock-Free Multithreaded Worker Pool**: It queries your CPU core count via sysconf(_SC_NPROCESSORS_ONLN) and spawns persistent worker threads that divide the screen into horizontal scanline bands, synchronized via a custom condition-variable barrier.
- **Signed Distance Field (SDF) Raymarching**: It renders a morphing, twisted 3D torus orbiting a central sphere using smooth minimum (smin) boolean blending, soft ambient occlusion, dynamic shadows, and a procedural psychedelic plasma floor.
- **Zero Flicker Double-Buffering**: A single pre-allocated memory buffer constructs the entire frame's ANSI bitstream before flushing it to stdout in a single atomic write() syscall.
- **Live Terminal Resizing**: Catches SIGWINCH signals to dynamically reallocate framebuffers on the fly when you resize or maximize your terminal window.
