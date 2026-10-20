/*
 * terminal-marcher.c - Multithreaded TrueColor Terminal 3D Raymarcher
 * Copyright (C) k!M/pizslacker 2026
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 * Compile: gcc -O3 -march=native -pthread terminal-marcher.c -o terminal-marcher -lm
 * Run:     ./terminal-marcher
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>
#include <termios.h>
#include <sys/ioctl.h>
#include <pthread.h>
#include <fcntl.h>

/* --- Vector Math Primitives --- */
typedef struct { float x, y, z; } vec3;

static inline vec3 v3(float x, float y, float z) { return (vec3){x, y, z}; }
static inline vec3 v3_add(vec3 a, vec3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
static inline vec3 v3_sub(vec3 a, vec3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
static inline vec3 v3_scale(vec3 a, float s) { return v3(a.x * s, a.y * s, a.z * s); }
static inline vec3 v3_mul(vec3 a, vec3 b) { return v3(a.x * b.x, a.y * b.y, a.z * b.z); }
static inline float v3_dot(vec3 a, vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline float v3_len(vec3 a) { return sqrtf(v3_dot(a, a)); }
static inline vec3 v3_norm(vec3 a) { float l = v3_len(a); return l > 0.0f ? v3_scale(a, 1.0f / l) : v3(0,0,0); }
static inline vec3 v3_max(vec3 a, float m) { return v3(fmaxf(a.x, m), fmaxf(a.y, m), fmaxf(a.z, m)); }
static inline float clampf(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }

static inline vec3 v3_rot_y(vec3 p, float a) {
    float c = cosf(a), s = sinf(a);
    return v3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
}

static inline vec3 v3_rot_x(vec3 p, float a) {
    float c = cosf(a), s = sinf(a);
    return v3(p.x, c * p.y - s * p.z, s * p.y + c * p.z);
}

/* Smooth minimum for organic metaball blending */
static inline float smin(float a, float b, float k) {
    float h = clampf(0.5f + 0.5f * (b - a) / k, 0.0f, 1.0f);
    return b * (1.0f - h) + a * h - k * h * (1.0f - h);
}

/* --- Scene Signed Distance Field (SDF) --- */
typedef struct {
    float dist;
    int material_id;
} Hit;

static Hit map_scene(vec3 p, float t) {
    /* 1. Ground plane with sine-wave ripple */
    float wave = 0.15f * sinf(p.x * 2.5f + t * 2.0f) * cosf(p.z * 2.5f + t * 1.5f);
    float floor_dist = p.y + 1.8f + wave;

    /* 2. Rotating twisted torus */
    vec3 tp = v3_sub(p, v3(0.0f, 0.1f, 0.0f));
    tp = v3_rot_y(tp, t * 0.8f);
    tp = v3_rot_x(tp, t * 0.6f);

    /* Twist deformation along Y */
    float twist = sinf(t * 0.7f) * 1.2f;
    float c = cosf(twist * tp.y), s = sinf(twist * tp.y);
    vec3 q = v3(c * tp.x - s * tp.z, tp.y, s * tp.x + c * tp.z);

    float torus_r1 = 1.05f, torus_r2 = 0.32f;
    float qx = sqrtf(q.x * q.x + q.z * q.z) - torus_r1;
    float torus_dist = sqrtf(qx * qx + q.y * q.y) - torus_r2;

    /* 3. Orbiting satellite spheres */
    float blobs = 1e9f;
    for (int i = 0; i < 3; i++) {
        float ang = t * 1.4f + i * 2.094395f; /* 2*PI / 3 */
        vec3 center = v3(cosf(ang) * 1.35f, sinf(ang * 1.5f) * 0.65f, sinf(ang) * 1.35f);
        float sp_dist = v3_len(v3_sub(p, center)) - 0.38f;
        blobs = smin(blobs, sp_dist, 0.45f);
    }

    /* Smoothly fuse the torus and orbiting spheres */
    float obj_dist = smin(torus_dist, blobs, 0.35f);

    if (floor_dist < obj_dist) {
        return (Hit){floor_dist, 1};
    }
    return (Hit){obj_dist, 2};
}

/* Calculate surface normal via finite differences */
static vec3 calc_normal(vec3 p, float t) {
    const float e = 0.0015f;
    float d = map_scene(p, t).dist;
    return v3_norm(v3(
        map_scene(v3(p.x + e, p.y, p.z), t).dist - d,
        map_scene(v3(p.x, p.y + e, p.z), t).dist - d,
        map_scene(v3(p.x, p.y, p.z + e), t).dist - d
    ));
}

/* Soft shadow march */
static float calc_soft_shadow(vec3 ro, vec3 rd, float t) {
    float res = 1.0f;
    float dist = 0.05f;
    for (int i = 0; i < 24; i++) {
        float h = map_scene(v3_add(ro, v3_scale(rd, dist)), t).dist;
        if (h < 0.002f) return 0.0f;
        res = fminf(res, 10.0f * h / dist);
        dist += clampf(h, 0.02f, 0.25f);
        if (dist > 12.0f) break;
    }
    return clampf(res, 0.0f, 1.0f);
}

/* Cosine-based iridescent color palette */
static vec3 palette(float t) {
    vec3 a = v3(0.5f, 0.5f, 0.5f);
    vec3 b = v3(0.5f, 0.5f, 0.5f);
    vec3 c = v3(1.0f, 1.0f, 1.0f);
    vec3 d = v3(0.00f, 0.33f, 0.67f);
    return v3(
        a.x + b.x * cosf(6.28318f * (c.x * t + d.x)),
        a.y + b.y * cosf(6.28318f * (c.y * t + d.y)),
        a.z + b.z * cosf(6.28318f * (c.z * t + d.z))
    );
}

/* --- Per-Pixel Raymarch Shader --- */
static vec3 render_pixel(float uv_x, float uv_y, float time_sec, float cam_yaw, float cam_zoom) {
    vec3 ro = v3(sinf(cam_yaw) * cam_zoom, 0.7f, cosf(cam_yaw) * cam_zoom);
    vec3 target = v3(0.0f, -0.1f, 0.0f);

    vec3 forward = v3_norm(v3_sub(target, ro));
    vec3 right   = v3_norm(v3(forward.z, 0.0f, -forward.x));
    vec3 up      = v3(
        right.y * forward.z - right.z * forward.y,
        right.z * forward.x - right.x * forward.z,
        right.x * forward.y - right.y * forward.x
    );

    vec3 rd = v3_norm(v3_add(v3_add(v3_scale(right, uv_x), v3_scale(up, uv_y)), v3_scale(forward, 1.6f)));

    /* Sky / background gradient */
    vec3 col = v3_scale(v3(0.05f, 0.04f, 0.12f), 1.0f - 0.3f * uv_y);

    float total_dist = 0.0f;
    int mat_id = 0;

    for (int step = 0; step < 64; step++) {
        vec3 p = v3_add(ro, v3_scale(rd, total_dist));
        Hit h = map_scene(p, time_sec);
        if (h.dist < 0.0025f) {
            mat_id = h.material_id;
            break;
        }
        total_dist += h.dist;
        if (total_dist > 22.0f) break;
    }

    if (mat_id > 0) {
        vec3 p = v3_add(ro, v3_scale(rd, total_dist));
        vec3 n = calc_normal(p, time_sec);
        vec3 light_pos = v3(3.0f * cosf(time_sec * 0.5f), 4.0f, 3.0f * sinf(time_sec * 0.5f));
        vec3 l = v3_norm(v3_sub(light_pos, p));

        float diff = fmaxf(v3_dot(n, l), 0.0f);
        float shadow = calc_soft_shadow(v3_add(p, v3_scale(n, 0.02f)), l, time_sec);
        diff *= shadow;

        /* Specular highlight (Blinn-Phong) */
        vec3 view_dir = v3_scale(rd, -1.0f);
        vec3 half_dir = v3_norm(v3_add(l, view_dir));
        float spec = powf(fmaxf(v3_dot(n, half_dir), 0.0f), 32.0f) * shadow;

        /* Fresnel rim lighting */
        float fresnel = powf(1.0f - fmaxf(v3_dot(n, view_dir), 0.0f), 3.0f);

        vec3 base_col;
        if (mat_id == 1) {
            /* Neon cyber grid floor */
            float gx = fabsf(fmodf(p.x * 1.5f, 1.0f));
            float gz = fabsf(fmodf(p.z * 1.5f, 1.0f));
            float line = (gx < 0.06f || gz < 0.06f) ? 1.0f : 0.0f;
            vec3 grid_glow = palette(p.x * 0.1f + p.z * 0.1f - time_sec * 0.3f);
            base_col = v3_add(v3(0.06f, 0.06f, 0.10f), v3_scale(grid_glow, line * 0.85f));
        } else {
            /* Iridescent morphing sculpture */
            base_col = palette(v3_dot(n, v3(0.5f, 0.8f, 0.3f)) * 0.5f + time_sec * 0.25f);
        }

        vec3 ambient = v3(0.08f, 0.07f, 0.14f);
        col = v3_add(v3_mul(base_col, v3_add(ambient, v3_scale(v3(1.0f, 0.95f, 0.9f), diff))),
                     v3_scale(v3(1.0f, 1.0f, 1.0f), spec * 0.7f));
        col = v3_add(col, v3_scale(v3(0.2f, 0.6f, 1.0f), fresnel * 0.45f));

        /* Exponential distance fog */
        float fog = expf(-0.008f * total_dist * total_dist);
        vec3 fog_col = v3(0.04f, 0.03f, 0.09f);
        col = v3_add(v3_scale(col, fog), v3_scale(fog_col, 1.0f - fog));
    }

    /* Gamma correction (approx sqrt) */
    return v3(sqrtf(clampf(col.x, 0.0f, 1.0f)),
              sqrtf(clampf(col.y, 0.0f, 1.0f)),
              sqrtf(clampf(col.z, 0.0f, 1.0f)));
}

/* --- Global Engine & Thread Pool State --- */
typedef struct { uint8_t r, g, b; } RGB;

static volatile sig_atomic_t g_running = 1;
static volatile sig_atomic_t g_resized = 1;

static int g_term_w = 80;
static int g_term_h = 24;
static RGB *g_framebuffer = NULL; /* Size: g_term_w * (g_term_h * 2) */
static char *g_ansibuf = NULL;

static float g_time = 0.0f;
static float g_cam_yaw = 0.4f;
static float g_cam_zoom = 3.6f;
static bool  g_paused = false;

/* Thread synchronization */
static int g_num_threads = 4;
static pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_cond_work = PTHREAD_COND_INITIALIZER;
static pthread_cond_t  g_cond_done = PTHREAD_COND_INITIALIZER;
static int g_frame_gen = 0;
static int g_active_workers = 0;
static bool g_shutdown_threads = false;

static void handle_sigint(int sig) { (void)sig; g_running = 0; }
static void handle_sigwinch(int sig) { (void)sig; g_resized = 1; }

/* Worker thread function: renders assigned horizontal scanlines */
static void *render_worker(void *arg) {
    int thread_id = *(int *)arg;
    int local_gen = 0;

    while (1) {
        pthread_mutex_lock(&g_mutex);
        while (g_frame_gen == local_gen && !g_shutdown_threads) {
            pthread_cond_wait(&g_cond_work, &g_mutex);
        }
        if (g_shutdown_threads) {
            pthread_mutex_unlock(&g_mutex);
            break;
        }
        local_gen = g_frame_gen;
        pthread_mutex_unlock(&g_mutex);

        int w = g_term_w;
        int pix_h = g_term_h * 2;
        float aspect = (float)w / (float)pix_h;

        for (int py = thread_id; py < pix_h; py += g_num_threads) {
            float uv_y = 1.0f - 2.0f * ((float)py / (float)pix_h);
            RGB *row = &g_framebuffer[py * w];
            for (int px = 0; px < w; px++) {
                float uv_x = (2.0f * ((float)px / (float)w) - 1.0f) * aspect;
                vec3 c = render_pixel(uv_x, uv_y, g_time, g_cam_yaw, g_cam_zoom);
                row[px] = (RGB){
                    (uint8_t)(c.x * 255.0f),
                    (uint8_t)(c.y * 255.0f),
                    (uint8_t)(c.z * 255.0f)
                };
            }
        }

        pthread_mutex_lock(&g_mutex);
        g_active_workers--;
        if (g_active_workers == 0) {
            pthread_cond_signal(&g_cond_done);
        }
        pthread_mutex_unlock(&g_mutex);
    }
    return NULL;
}

/* Fast integer-to-ASCII helper for RGB values (0-255) */
static inline char *fast_utoa(char *p, uint8_t v) {
    if (v >= 100) {
        *p++ = '0' + (v / 100);
        v %= 100;
        *p++ = '0' + (v / 10);
        *p++ = '0' + (v % 10);
    } else if (v >= 10) {
        *p++ = '0' + (v / 10);
        *p++ = '0' + (v % 10);
    } else {
        *p++ = '0' + v;
    }
    return p;
}

/* Raw terminal configuration */
static struct termios g_orig_termios;

static void restore_terminal(void) {
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &g_orig_termios);
    const char *cleanup = "\033[0m\033[?25h\033[?1049l";
    if (write(STDOUT_FILENO, cleanup, strlen(cleanup)) < 0) {}
}

static void init_terminal(void) {
    tcgetattr(STDIN_FILENO, &g_orig_termios);
    atexit(restore_terminal);

    struct termios raw = g_orig_termios;
    raw.c_lflag &= ~(ECHO | ICANON);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);

    /* Non-blocking stdin */
    int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);

    /* Switch to alternate screen buffer & hide cursor */
    const char *init_seq = "\033[?1049h\033[?25l\033[2J";
    if (write(STDOUT_FILENO, init_seq, strlen(init_seq)) < 0) {}
}

static void update_dimensions(void) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 10 && ws.ws_row > 5) {
        g_term_w = ws.ws_col;
        g_term_h = ws.ws_row - 1; /* Leave 1 row at bottom for telemetry HUD */
    }
    free(g_framebuffer);
    free(g_ansibuf);
    g_framebuffer = malloc((size_t)g_term_w * (size_t)(g_term_h * 2) * sizeof(RGB));
    /* Worst-case ~48 bytes per terminal cell + HUD margin */
    g_ansibuf = malloc((size_t)g_term_w * (size_t)g_term_h * 64 + 1024);
    g_resized = 0;
}

int main(void) {
    signal(SIGINT, handle_sigint);
    signal(SIGWINCH, handle_sigwinch);

    init_terminal();
    update_dimensions();

    long cores = sysconf(_SC_NPROCESSORS_ONLN);
    g_num_threads = (cores > 1 && cores <= 64) ? (int)cores : 4;

    pthread_t *threads = malloc((size_t)g_num_threads * sizeof(pthread_t));
    int *thread_ids = malloc((size_t)g_num_threads * sizeof(int));
    for (int i = 0; i < g_num_threads; i++) {
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, render_worker, &thread_ids[i]);
    }

    struct timespec ts_prev, ts_now;
    clock_gettime(CLOCK_MONOTONIC, &ts_prev);
    float fps_smooth = 60.0f;

    while (g_running) {
        if (g_resized) update_dimensions();

        /* Read non-blocking keyboard input */
        char ch;
        while (read(STDIN_FILENO, &ch, 1) > 0) {
            if (ch == 'q' || ch == 'Q' || ch == 27) g_running = 0;
            else if (ch == ' ') g_paused = !g_paused;
            else if (ch == 'a' || ch == 'A') g_cam_yaw -= 0.12f;
            else if (ch == 'd' || ch == 'D') g_cam_yaw += 0.12f;
            else if (ch == 'w' || ch == 'W') g_cam_zoom = fmaxf(1.8f, g_cam_zoom - 0.2f);
            else if (ch == 's' || ch == 'S') g_cam_zoom = fminf(7.0f, g_cam_zoom + 0.2f);
        }

        clock_gettime(CLOCK_MONOTONIC, &ts_now);
        float dt = (float)(ts_now.tv_sec - ts_prev.tv_sec) +
                   (float)(ts_now.tv_nsec - ts_prev.tv_nsec) * 1e-9f;
        ts_prev = ts_now;

        if (!g_paused) {
            g_time += dt;
            g_cam_yaw += dt * 0.25f;
        }
        if (dt > 0.0001f) {
            fps_smooth = fps_smooth * 0.9f + (1.0f / dt) * 0.1f;
        }

        /* Dispatch frame to worker threads */
        pthread_mutex_lock(&g_mutex);
        g_active_workers = g_num_threads;
        g_frame_gen++;
        pthread_cond_broadcast(&g_cond_work);
        while (g_active_workers > 0) {
            pthread_cond_wait(&g_cond_done, &g_mutex);
        }
        pthread_mutex_unlock(&g_mutex);

        /* Assemble ANSI output stream with half-block characters */
        char *p = g_ansibuf;
        memcpy(p, "\033[H", 3);
        p += 3;

        for (int row = 0; row < g_term_h; row++) {
            RGB *top_row = &g_framebuffer[(row * 2) * g_term_w];
            RGB *bot_row = &g_framebuffer[(row * 2 + 1) * g_term_w];

            for (int col = 0; col < g_term_w; col++) {
                RGB top = top_row[col];
                RGB bot = bot_row[col];

                /* \033[38;2;R;G;B;48;2;R;G;Bm▀ */
                memcpy(p, "\033[38;2;", 7); p += 7;
                p = fast_utoa(p, top.r); *p++ = ';';
                p = fast_utoa(p, top.g); *p++ = ';';
                p = fast_utoa(p, top.b);
                memcpy(p, ";48;2;", 6); p += 6;
                p = fast_utoa(p, bot.r); *p++ = ';';
                p = fast_utoa(p, bot.g); *p++ = ';';
                p = fast_utoa(p, bot.b); *p++ = 'm';
                /* UTF-8 Upper Half Block U+2580: 0xE2 0x96 0x80 */
                *p++ = (char)0xE2; *p++ = (char)0x96; *p++ = (char)0x80;
            }
            memcpy(p, "\033[0m\r\n", 6);
            p += 6;
        }

        /* Telemetry HUD status bar */
        int hud_len = snprintf(p, 256,
            "\033[48;2;20;20;35;38;2;0;255;200m ⚡ NEON SDF RAYMARCHER | %dx%d px | %d Threads | %5.1f FPS | [W/A/S/D] Orbit [SPACE] Pause [Q] Quit \033[K\033[0m",
            g_term_w, g_term_h * 2, g_num_threads, fps_smooth);
        if (hud_len > 0) p += hud_len;

        /* Single atomic write to stdout */
        if (write(STDOUT_FILENO, g_ansibuf, (size_t)(p - g_ansibuf)) < 0) {}

        /* Cap around ~60 FPS (16ms target) */
        struct timespec ts_end;
        clock_gettime(CLOCK_MONOTONIC, &ts_end);
        long elapsed_us = (ts_end.tv_sec - ts_now.tv_sec) * 1000000L +
                          (ts_end.tv_nsec - ts_now.tv_nsec) / 1000L;
        if (elapsed_us < 16000L) {
            struct timespec sleep_ts = {0, (16000L - elapsed_us) * 1000L};
            nanosleep(&sleep_ts, NULL);
        }
    }

    /* Clean shutdown of worker pool */
    pthread_mutex_lock(&g_mutex);
    g_shutdown_threads = true;
    pthread_cond_broadcast(&g_cond_work);
    pthread_mutex_unlock(&g_mutex);

    for (int i = 0; i < g_num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    free(threads);
    free(thread_ids);
    free(g_framebuffer);
    free(g_ansibuf);
    return 0;
}
