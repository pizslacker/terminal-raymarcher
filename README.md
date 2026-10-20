# terminal-raymarcher

Zero-dependency, multithreaded real-time terminal raymarcher and plasma engine written in pure C11 and POSIX threads (pthreads).

Instead of outputting a static image or requiring external graphics libraries like `SDL2` or `Raylib`, it hijacks your terminal using raw ANSI escape sequences and Unicode half-block characters (▀) to double the vertical resolution—turning any standard terminal emulator into a 60 FPS true-color (24-bit RGB) 3D viewport.
