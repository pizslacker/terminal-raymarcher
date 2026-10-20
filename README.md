# terminal-raymarcher

Zero-dependency, multithreaded real-time terminal raymarcher and plasma engine written in pure C11 and POSIX threads (pthreads).

![terminal-raymarcher](images/neon-marcher-avx2.png)

Instead of outputting a static image or requiring external graphics libraries like `SDL2` or `Raylib`, it hijacks your terminal using raw ANSI escape sequences and Unicode half-block characters (▀) to double the vertical resolution—turning any standard terminal emulator into a 60 FPS true-color (24-bit RGB) 3D viewport.

## Multithreaded TrueColor Terminal 3D Raymarcher

- **Sub-Pixel Terminal Rendering**: Every terminal cell contains two vertical pixels by pairing the foreground color (\033[38;2;R;G;Bm) with the upper half-block character ▀ and the background color (\033[48;2;R;G;Bm) for the lower half.
- **Lock-Free Multithreaded Worker Pool**: It queries your CPU core count via sysconf(_SC_NPROCESSORS_ONLN) and spawns persistent worker threads that divide the screen into horizontal scanline bands, synchronized via a custom condition-variable barrier.
- **Signed Distance Field (SDF) Raymarching**: It renders a morphing, twisted 3D torus orbiting a central sphere using smooth minimum (smin) boolean blending, soft ambient occlusion, dynamic shadows, and a procedural psychedelic plasma floor.
- **Zero Flicker Double-Buffering**: A single pre-allocated memory buffer constructs the entire frame's ANSI bitstream before flushing it to stdout in a single atomic write() syscall.
- **Live Terminal Resizing**: Catches SIGWINCH signals to dynamically reallocate framebuffers on the fly when you resize or maximize your terminal window.

## 8-Wide AVX2 + FMA SIMD Terminal Raymarcher
- **SoA Data Layout (vec3_avx)**: Instead of holding $(x, y, z)$ for 1 pixel in a register, three 256-bit YMM registers hold $[x_0 \dots x_7]$, $[y_0 \dots y_7]$, and $[z_0 \dots z_7]$. Every arithmetic instruction operates across 8 horizontal pixels in lockstep.
- **Branchless Predication & Lane Masking**: In a scalar raymarcher, a single ray breaks out of the for loop as soon as h.dist < 0.0025f or total_dist > 22.0f. In SIMD, some of the 8 rays will hit geometry early while neighboring rays continue marching into the background. We maintain an active_mask using _mm256_cmp_ps(..., _CMP_GT_OQ) and only break the 8-wide packet when _mm256_movemask_ps(active_mask) == 0.
- **Fast Polynomial $\sin/\cos$ (sin256_ps / cos256_ps)**: The standard C library (sinf/cosf) has no portable AVX2 intrinsic in immintrin.h, and unpacking __m256 registers to call scalar sinf 8 times destroys SIMD throughput. Below is an inline range-reduced minimax polynomial that computes 8 sines and cosines simultaneously in ~12 clock cycles using FMA.
- **Precomputed Invariants**: Time-dependent scene constants (torus rotation matrices, twist factors, and satellite sphere centers) do not depend on pixel coordinates. Precomputing them once per frame instead of inside map_scene cuts redundant trig evaluations by over 80%.
