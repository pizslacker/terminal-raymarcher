/*
 * neon-marcher-avx2.c - 8-Wide AVX2 + FMA SIMD Terminal Raymarcher
 * Features: DEC Mode 2026 VSync, ANSI RLE State Deduplication, Absolute Frame Pacing
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
 * Compile: gcc -O3 -march=native -mavx2 -mfma -pthread neon_marcher_avx2.c -o neon_marcher_avx2 -lm
 * Run:     ./neon_marcher_avx2
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdalign.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#include <termios.h>
#include <sys/ioctl.h>
#include <pthread.h>
#include <fcntl.h>
#include <immintrin.h>

/* --- Scalar Helper Primitives --- */
typedef struct { float x, y, z; } vec3;
static inline vec3 v3(float x, float y, float z) { return (vec3){x, y, z}; }
static inline vec3 v3_sub(vec3 a, vec3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
static inline float v3_dot(vec3 a, vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline vec3 v3_norm(vec3 a) {
    float l = sqrtf(v3_dot(a, a));
    return l > 0.0f ? v3(a.x / l, a.y / l, a.z / l) : v3(0, 0, 0);
}

/* --- 8-Wide SoA SIMD Vector Primitives --- */
typedef struct {
    __m256 x, y, z;
} vec3_avx;

static inline vec3_avx v3a_set1(float x, float y, float z) {
    return (vec3_avx){ _mm256_set1_ps(x), _mm256_set1_ps(y), _mm256_set1_ps(z) };
}
static inline vec3_avx v3a_add(vec3_avx a, vec3_avx b) {
    return (vec3_avx){ _mm256_add_ps(a.x, b.x), _mm256_add_ps(a.y, b.y), _mm256_add_ps(a.z, b.z) };
}
static inline vec3_avx v3a_sub(vec3_avx a, vec3_avx b) {
    return (vec3_avx){ _mm256_sub_ps(a.x, b.x), _mm256_sub_ps(a.y, b.y), _mm256_sub_ps(a.z, b.z) };
}
static inline vec3_avx v3a_scale(vec3_avx a, __m256 s) {
    return (vec3_avx){ _mm256_mul_ps(a.x, s), _mm256_mul_ps(a.y, s), _mm256_mul_ps(a.z, s) };
}
static inline vec3_avx v3a_fmadd(vec3_avx a, __m256 s, vec3_avx b) {
    return (vec3_avx){
        _mm256_fmadd_ps(a.x, s, b.x),
        _mm256_fmadd_ps(a.y, s, b.y),
        _mm256_fmadd_ps(a.z, s, b.z)
    };
}
static inline __m256 v3a_dot(vec3_avx a, vec3_avx b) {
    return _mm256_fmadd_ps(a.x, b.x, _mm256_fmadd_ps(a.y, b.y, _mm256_mul_ps(a.z, b.z)));
}
static inline __m256 v3a_len(vec3_avx a) {
    return _mm256_sqrt_ps(v3a_dot(a, a));
}
static inline vec3_avx v3a_norm(vec3_avx a) {
    __m256 inv_len = _mm256_rsqrt_ps(_mm256_max_ps(v3a_dot(a, a), _mm256_set1_ps(1e-12f)));
    return v3a_scale(a, inv_len);
}
static inline __m256 clamp256_ps(__m256 x, __m256 lo, __m256 hi) {
    return _mm256_min_ps(_mm256_max_ps(x, lo), hi);
}
static inline __m256 abs256_ps(__m256 x) {
    return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x);
}

/* --- Fast 8-Wide AVX2 Sine & Cosine --- */
static inline void sincos256_ps(__m256 x, __m256 *out_sin, __m256 *out_cos) {
    const __m256 inv_twopi = _mm256_set1_ps(0.15915494309f);
    const __m256 twopi     = _mm256_set1_ps(6.28318530718f);
    const __m256 half_pi   = _mm256_set1_ps(1.57079632679f);
    const __m256 pi        = _mm256_set1_ps(3.14159265359f);

    __m256 k = _mm256_round_ps(_mm256_mul_ps(x, inv_twopi), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    x = _mm256_fnmadd_ps(k, twopi, x);

    __m256 xc = _mm256_add_ps(x, half_pi);
    __m256 wrap_mask = _mm256_cmp_ps(xc, pi, _CMP_GT_OQ);
    xc = _mm256_sub_ps(xc, _mm256_and_ps(wrap_mask, twopi));

    const __m256 B = _mm256_set1_ps(1.2732395447f);
    const __m256 C = _mm256_set1_ps(-0.4052847345f);
    const __m256 P = _mm256_set1_ps(0.225f);

    __m256 ys = _mm256_fmadd_ps(C, _mm256_mul_ps(x, abs256_ps(x)), _mm256_mul_ps(B, x));
    *out_sin  = _mm256_fmadd_ps(P, _mm256_fnmadd_ps(ys, ys, _mm256_mul_ps(ys, abs256_ps(ys))), ys);

    __m256 yc = _mm256_fmadd_ps(C, _mm256_mul_ps(xc, abs256_ps(xc)), _mm256_mul_ps(B, xc));
    *out_cos  = _mm256_fmadd_ps(P, _mm256_fnmadd_ps(yc, yc, _mm256_mul_ps(yc, abs256_ps(yc))), yc);
}

static inline __m256 smin256_ps(__m256 a, __m256 b, float k_scalar) {
    __m256 k = _mm256_set1_ps(k_scalar);
    __m256 inv_k = _mm256_set1_ps(0.5f / k_scalar);
    __m256 h = clamp256_ps(_mm256_fmadd_ps(_mm256_sub_ps(b, a), inv_k, _mm256_set1_ps(0.5f)),
                           _mm256_setzero_ps(), _mm256_set1_ps(1.0f));
    __m256 one_minus_h = _mm256_sub_ps(_mm256_set1_ps(1.0f), h);
    __m256 mix = _mm256_fmadd_ps(a, h, _mm256_mul_ps(b, one_minus_h));
    return _mm256_fnmadd_ps(_mm256_mul_ps(k, h), one_minus_h, mix);
}

/* --- Precomputed Per-Frame Scene Invariants --- */
typedef struct {
    float t;
    float rot_yc, rot_ys;
    float rot_xc, rot_xs;
    float twist;
    vec3  blob_centers[3];
} SceneFrame;

static SceneFrame prepare_scene_frame(float t) {
    SceneFrame sf;
    sf.t = t;
    sf.rot_yc = cosf(t * 0.8f); sf.rot_ys = sinf(t * 0.8f);
    sf.rot_xc = cosf(t * 0.6f); sf.rot_xs = sinf(t * 0.6f);
    sf.twist  = sinf(t * 0.7f) * 1.2f;
    for (int i = 0; i < 3; i++) {
        float ang = t * 1.4f + i * 2.094395f;
        sf.blob_centers[i] = v3(cosf(ang) * 1.35f, sinf(ang * 1.5f) * 0.65f, sinf(ang) * 1.35f);
    }
    return sf;
}

typedef struct {
    __m256 dist;
    __m256 mat_id;
} Hit_avx;

static Hit_avx map_scene_avx2(vec3_avx p, const SceneFrame *sf) {
    __m256 wx = _mm256_fmadd_ps(p.x, _mm256_set1_ps(2.5f), _mm256_set1_ps(sf->t * 2.0f));
    __m256 wz = _mm256_fmadd_ps(p.z, _mm256_set1_ps(2.5f), _mm256_set1_ps(sf->t * 1.5f));
    __m256 sin_wx, dummy_cos, dummy_sin, cos_wz;
    sincos256_ps(wx, &sin_wx, &dummy_cos);
    sincos256_ps(wz, &dummy_sin, &cos_wz);

    __m256 wave = _mm256_mul_ps(_mm256_set1_ps(0.15f), _mm256_mul_ps(sin_wx, cos_wz));
    __m256 floor_dist = _mm256_add_ps(_mm256_add_ps(p.y, _mm256_set1_ps(1.8f)), wave);

    __m256 tpy = _mm256_sub_ps(p.y, _mm256_set1_ps(0.1f));
    __m256 ryc = _mm256_set1_ps(sf->rot_yc), rys = _mm256_set1_ps(sf->rot_ys);
    __m256 rxc = _mm256_set1_ps(sf->rot_xc), rxs = _mm256_set1_ps(sf->rot_xs);

    __m256 x1 = _mm256_fmadd_ps(ryc, p.x, _mm256_mul_ps(rys, p.z));
    __m256 z1 = _mm256_fnmadd_ps(rys, p.x, _mm256_mul_ps(ryc, p.z));
    __m256 y2 = _mm256_fnmadd_ps(rxs, z1, _mm256_mul_ps(rxc, tpy));
    __m256 z2 = _mm256_fmadd_ps(rxs, tpy, _mm256_mul_ps(rxc, z1));

    __m256 twist_ang = _mm256_mul_ps(_mm256_set1_ps(sf->twist), y2);
    __m256 ts, tc;
    sincos256_ps(twist_ang, &ts, &tc);

    __m256 qx_rot = _mm256_fnmadd_ps(ts, z2, _mm256_mul_ps(tc, x1));
    __m256 qz_rot = _mm256_fmadd_ps(ts, x1, _mm256_mul_ps(tc, z2));

    __m256 qx = _mm256_sub_ps(_mm256_sqrt_ps(_mm256_fmadd_ps(qx_rot, qx_rot, _mm256_mul_ps(qz_rot, qz_rot))),
                              _mm256_set1_ps(1.05f));
    __m256 torus_dist = _mm256_sub_ps(_mm256_sqrt_ps(_mm256_fmadd_ps(qx, qx, _mm256_mul_ps(y2, y2))),
                                      _mm256_set1_ps(0.32f));

    __m256 blobs = _mm256_set1_ps(1e9f);
    for (int i = 0; i < 3; i++) {
        vec3_avx c = v3a_set1(sf->blob_centers[i].x, sf->blob_centers[i].y, sf->blob_centers[i].z);
        __m256 sp_dist = _mm256_sub_ps(v3a_len(v3a_sub(p, c)), _mm256_set1_ps(0.38f));
        blobs = smin256_ps(blobs, sp_dist, 0.45f);
    }

    __m256 obj_dist = smin256_ps(torus_dist, blobs, 0.35f);
    __m256 is_floor = _mm256_cmp_ps(floor_dist, obj_dist, _CMP_LT_OQ);

    Hit_avx res;
    res.dist   = _mm256_min_ps(floor_dist, obj_dist);
    res.mat_id = _mm256_blendv_ps(_mm256_set1_ps(2.0f), _mm256_set1_ps(1.0f), is_floor);
    return res;
}

static vec3_avx calc_normal_avx2(vec3_avx p, const SceneFrame *sf) {
    __m256 eps = _mm256_set1_ps(0.0015f);
    __m256 d   = map_scene_avx2(p, sf).dist;
    vec3_avx px = p; px.x = _mm256_add_ps(px.x, eps);
    vec3_avx py = p; py.y = _mm256_add_ps(py.y, eps);
    vec3_avx pz = p; pz.z = _mm256_add_ps(pz.z, eps);

    vec3_avx n = {
        _mm256_sub_ps(map_scene_avx2(px, sf).dist, d),
        _mm256_sub_ps(map_scene_avx2(py, sf).dist, d),
        _mm256_sub_ps(map_scene_avx2(pz, sf).dist, d)
    };
    return v3a_norm(n);
}

static vec3_avx palette_avx2(__m256 t) {
    const __m256 twopi = _mm256_set1_ps(6.2831853f);
    __m256 s, c0, c1, c2;
    sincos256_ps(_mm256_mul_ps(twopi, t), &s, &c0);
    sincos256_ps(_mm256_mul_ps(twopi, _mm256_add_ps(t, _mm256_set1_ps(0.33f))), &s, &c1);
    sincos256_ps(_mm256_mul_ps(twopi, _mm256_add_ps(t, _mm256_set1_ps(0.67f))), &s, &c2);

    __m256 half = _mm256_set1_ps(0.5f);
    return (vec3_avx){
        _mm256_fmadd_ps(half, c0, half),
        _mm256_fmadd_ps(half, c1, half),
        _mm256_fmadd_ps(half, c2, half)
    };
}

typedef struct { uint8_t r, g, b; } RGB;

static void render_packet8_avx2(
    __m256 uv_x, __m256 uv_y,
    vec3 cam_ro, vec3 cam_right, vec3 cam_up, vec3 cam_fwd,
    const SceneFrame *sf,
    RGB *out_pixels)
{
    vec3_avx ro    = v3a_set1(cam_ro.x, cam_ro.y, cam_ro.z);
    vec3_avx right = v3a_set1(cam_right.x, cam_right.y, cam_right.z);
    vec3_avx up    = v3a_set1(cam_up.x, cam_up.y, cam_up.z);
    vec3_avx fwd   = v3a_set1(cam_fwd.x * 1.6f, cam_fwd.y * 1.6f, cam_fwd.z * 1.6f);

    vec3_avx rd = v3a_norm(v3a_fmadd(right, uv_x, v3a_fmadd(up, uv_y, fwd)));

    __m256 total_dist = _mm256_setzero_ps();
    __m256 hit_mat    = _mm256_setzero_ps();
    __m256 active     = _mm256_castsi256_ps(_mm256_set1_epi32(-1));

    const __m256 eps_hit  = _mm256_set1_ps(0.0025f);
    const __m256 max_dist = _mm256_set1_ps(22.0f);

    for (int step = 0; step < 64; step++) {
        vec3_avx p = v3a_fmadd(rd, total_dist, ro);
        Hit_avx h  = map_scene_avx2(p, sf);

        __m256 hit_now = _mm256_and_ps(active, _mm256_cmp_ps(h.dist, eps_hit, _CMP_LT_OQ));
        hit_mat = _mm256_blendv_ps(hit_mat, h.mat_id, hit_now);

        __m256 still_marching = _mm256_andnot_ps(hit_now, active);
        total_dist = _mm256_add_ps(total_dist, _mm256_and_ps(still_marching, h.dist));

        active = _mm256_and_ps(still_marching, _mm256_cmp_ps(total_dist, max_dist, _CMP_LT_OQ));
        if (_mm256_movemask_ps(active) == 0) break;
    }

    __m256 sky_factor = _mm256_fnmadd_ps(_mm256_set1_ps(0.3f), uv_y, _mm256_set1_ps(1.0f));
    vec3_avx col = v3a_scale(v3a_set1(0.05f, 0.04f, 0.12f), sky_factor);

    __m256 any_hit_mask = _mm256_cmp_ps(hit_mat, _mm256_setzero_ps(), _CMP_GT_OQ);
    if (_mm256_movemask_ps(any_hit_mask) != 0) {
        vec3_avx p = v3a_fmadd(rd, total_dist, ro);
        vec3_avx n = calc_normal_avx2(p, sf);

        vec3_avx light_pos = v3a_set1(3.0f * cosf(sf->t * 0.5f), 4.0f, 3.0f * sinf(sf->t * 0.5f));
        vec3_avx l = v3a_norm(v3a_sub(light_pos, p));

        __m256 diff = _mm256_max_ps(v3a_dot(n, l), _mm256_setzero_ps());

        vec3_avx shadow_ro = v3a_fmadd(n, _mm256_set1_ps(0.03f), p);
        __m256 sh_dist = map_scene_avx2(v3a_fmadd(l, _mm256_set1_ps(0.35f), shadow_ro), sf).dist;
        __m256 shadow  = clamp256_ps(_mm256_mul_ps(sh_dist, _mm256_set1_ps(3.0f)),
                                     _mm256_set1_ps(0.15f), _mm256_set1_ps(1.0f));
        diff = _mm256_mul_ps(diff, shadow);

        vec3_avx view_dir = v3a_scale(rd, _mm256_set1_ps(-1.0f));
        vec3_avx half_dir = v3a_norm(v3a_add(l, view_dir));
        __m256 spec = _mm256_max_ps(v3a_dot(n, half_dir), _mm256_setzero_ps());
        for (int k = 0; k < 5; k++) spec = _mm256_mul_ps(spec, spec);
        spec = _mm256_mul_ps(spec, shadow);

        __m256 fn_base = _mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_max_ps(v3a_dot(n, view_dir), _mm256_setzero_ps()));
        __m256 fresnel = _mm256_mul_ps(fn_base, _mm256_mul_ps(fn_base, fn_base));

        __m256 gx = abs256_ps(_mm256_mul_ps(p.x, _mm256_set1_ps(1.5f)));
        __m256 gz = abs256_ps(_mm256_mul_ps(p.z, _mm256_set1_ps(1.5f)));
        gx = _mm256_sub_ps(gx, _mm256_floor_ps(gx));
        gz = _mm256_sub_ps(gz, _mm256_floor_ps(gz));

        __m256 line_mask = _mm256_or_ps(_mm256_cmp_ps(gx, _mm256_set1_ps(0.06f), _CMP_LT_OQ),
                                        _mm256_cmp_ps(gz, _mm256_set1_ps(0.06f), _CMP_LT_OQ));
        __m256 line_val  = _mm256_and_ps(line_mask, _mm256_set1_ps(0.85f));

        __m256 grid_t = _mm256_fmadd_ps(_mm256_add_ps(p.x, p.z), _mm256_set1_ps(0.1f), _mm256_set1_ps(-sf->t * 0.3f));
        vec3_avx floor_col = v3a_fmadd(palette_avx2(grid_t), line_val, v3a_set1(0.06f, 0.06f, 0.10f));

        __m256 obj_t = _mm256_fmadd_ps(v3a_dot(n, v3a_set1(0.5f, 0.8f, 0.3f)),
                                       _mm256_set1_ps(0.5f), _mm256_set1_ps(sf->t * 0.25f));
        vec3_avx obj_col = palette_avx2(obj_t);

        __m256 is_floor = _mm256_cmp_ps(hit_mat, _mm256_set1_ps(1.5f), _CMP_LT_OQ);
        vec3_avx base_col = {
            _mm256_blendv_ps(obj_col.x, floor_col.x, is_floor),
            _mm256_blendv_ps(obj_col.y, floor_col.y, is_floor),
            _mm256_blendv_ps(obj_col.z, floor_col.z, is_floor)
        };

        vec3_avx lit = {
            _mm256_fmadd_ps(base_col.x, _mm256_fmadd_ps(_mm256_set1_ps(1.00f), diff, _mm256_set1_ps(0.08f)), _mm256_mul_ps(spec, _mm256_set1_ps(0.7f))),
            _mm256_fmadd_ps(base_col.y, _mm256_fmadd_ps(_mm256_set1_ps(0.95f), diff, _mm256_set1_ps(0.07f)), _mm256_mul_ps(spec, _mm256_set1_ps(0.7f))),
            _mm256_fmadd_ps(base_col.z, _mm256_fmadd_ps(_mm256_set1_ps(0.90f), diff, _mm256_set1_ps(0.14f)), _mm256_mul_ps(spec, _mm256_set1_ps(0.7f)))
        };
        lit = v3a_fmadd(v3a_set1(0.2f, 0.6f, 1.0f), _mm256_mul_ps(fresnel, _mm256_set1_ps(0.45f)), lit);

        __m256 fog = _mm256_rcp_ps(_mm256_fmadd_ps(_mm256_set1_ps(0.008f), _mm256_mul_ps(total_dist, total_dist), _mm256_set1_ps(1.0f)));
        __m256 one_minus_fog = _mm256_sub_ps(_mm256_set1_ps(1.0f), fog);
        vec3_avx fogged = v3a_fmadd(lit, fog, v3a_scale(v3a_set1(0.04f, 0.03f, 0.09f), one_minus_fog));

        col.x = _mm256_blendv_ps(col.x, fogged.x, any_hit_mask);
        col.y = _mm256_blendv_ps(col.y, fogged.y, any_hit_mask);
        col.z = _mm256_blendv_ps(col.z, fogged.z, any_hit_mask);
    }

    __m256 zero = _mm256_setzero_ps(), one = _mm256_set1_ps(1.0f), c255 = _mm256_set1_ps(255.0f);
    __m256 r_f = _mm256_mul_ps(_mm256_sqrt_ps(clamp256_ps(col.x, zero, one)), c255);
    __m256 g_f = _mm256_mul_ps(_mm256_sqrt_ps(clamp256_ps(col.y, zero, one)), c255);
    __m256 b_f = _mm256_mul_ps(_mm256_sqrt_ps(clamp256_ps(col.z, zero, one)), c255);

    alignas(32) int32_t r_i[8], g_i[8], b_i[8];
    _mm256_store_si256((__m256i *)r_i, _mm256_cvttps_epi32(r_f));
    _mm256_store_si256((__m256i *)g_i, _mm256_cvttps_epi32(g_f));
    _mm256_store_si256((__m256i *)b_i, _mm256_cvttps_epi32(b_f));

    /* Quantize lowest 2 bits (& 0xFC) to boost RLE compression in smooth gradients */
    for (int lane = 0; lane < 8; lane++) {
        out_pixels[lane] = (RGB){
            (uint8_t)(r_i[lane] & 0xFC),
            (uint8_t)(g_i[lane] & 0xFC),
            (uint8_t)(b_i[lane] & 0xFC)
        };
    }
}

/* --- Engine & Worker Pool State --- */
static volatile sig_atomic_t g_running = 1;
static volatile sig_atomic_t g_resized = 1;

static int g_term_w = 80;
static int g_term_h = 24;
static RGB *g_framebuffer = NULL;
static char *g_ansibuf = NULL;

static float g_time = 0.0f;
static float g_cam_yaw = 0.4f;
static float g_cam_zoom = 3.6f;
static bool  g_paused = false;
static SceneFrame g_scene_frame;

static int g_num_threads = 4;
static pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_cond_work = PTHREAD_COND_INITIALIZER;
static pthread_cond_t  g_cond_done = PTHREAD_COND_INITIALIZER;
static int g_frame_gen = 0;
static int g_active_workers = 0;
static bool g_shutdown_threads = false;

static void handle_sigint(int sig) { (void)sig; g_running = 0; }
static void handle_sigwinch(int sig) { (void)sig; g_resized = 1; }

static void *render_worker_avx2(void *arg) {
    int thread_id = *(int *)arg;
    int local_gen = 0;
    const __m256 lane_offsets = _mm256_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f);

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
        float inv_w2 = (2.0f * aspect) / (float)w;

        vec3 ro      = v3(sinf(g_cam_yaw) * g_cam_zoom, 0.7f, cosf(g_cam_yaw) * g_cam_zoom);
        vec3 target  = v3(0.0f, -0.1f, 0.0f);
        vec3 forward = v3_norm(v3_sub(target, ro));
        vec3 right   = v3_norm(v3(forward.z, 0.0f, -forward.x));
        vec3 up      = v3(
            right.y * forward.z - right.z * forward.y,
            right.z * forward.x - right.x * forward.z,
            right.x * forward.y - right.y * forward.x
        );

        for (int py = thread_id; py < pix_h; py += g_num_threads) {
            __m256 uv_y = _mm256_set1_ps(1.0f - 2.0f * ((float)py / (float)pix_h));
            RGB *row = &g_framebuffer[py * w];

            int px = 0;
            for (; px <= w - 8; px += 8) {
                __m256 px_vec = _mm256_add_ps(_mm256_set1_ps((float)px), lane_offsets);
                __m256 uv_x   = _mm256_fmadd_ps(px_vec, _mm256_set1_ps(inv_w2), _mm256_set1_ps(-aspect));
                render_packet8_avx2(uv_x, uv_y, ro, right, up, forward, &g_scene_frame, &row[px]);
            }

            if (px < w) {
                RGB tmp[8];
                __m256 px_vec = _mm256_add_ps(_mm256_set1_ps((float)px), lane_offsets);
                __m256 uv_x   = _mm256_fmadd_ps(px_vec, _mm256_set1_ps(inv_w2), _mm256_set1_ps(-aspect));
                render_packet8_avx2(uv_x, uv_y, ro, right, up, forward, &g_scene_frame, tmp);
                for (int r = 0; px + r < w; r++) {
                    row[px + r] = tmp[r];
                }
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

static inline char *fast_utoa(char *p, uint8_t v) {
    if (v >= 100) {
        *p++ = '0' + (v / 100); v %= 100;
        *p++ = '0' + (v / 10);  *p++ = '0' + (v % 10);
    } else if (v >= 10) {
        *p++ = '0' + (v / 10);  *p++ = '0' + (v % 10);
    } else {
        *p++ = '0' + v;
    }
    return p;
}

/* Guaranteed complete write loop to prevent partial PTY frame flushes */
static void write_all(int fd, const char *buf, size_t len) {
    size_t written = 0;
    while (written < len) {
        ssize_t n = write(fd, buf + written, len - written);
        if (n > 0) {
            written += (size_t)n;
        } else if (n < 0 && (errno == EINTR || errno == EAGAIN)) {
            continue;
        } else {
            break;
        }
    }
}

static struct termios g_orig_termios;

static void restore_terminal(void) {
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &g_orig_termios);
    const char *cleanup = "\033[?2026l\033[0m\033[?25h\033[?1049l";
    write_all(STDOUT_FILENO, cleanup, strlen(cleanup));
}

static void init_terminal(void) {
    tcgetattr(STDIN_FILENO, &g_orig_termios);
    atexit(restore_terminal);
    struct termios raw = g_orig_termios;
    raw.c_lflag &= ~(ECHO | ICANON);
    raw.c_cc[VMIN] = 0; raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
    int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);
    const char *init_seq = "\033[?1049h\033[?25l\033[2J";
    write_all(STDOUT_FILENO, init_seq, strlen(init_seq));
}

static void update_dimensions(void) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 10 && ws.ws_row > 5) {
        g_term_w = ws.ws_col;
        g_term_h = ws.ws_row - 1;
    }
    free(g_framebuffer);
    free(g_ansibuf);
    g_framebuffer = malloc((size_t)g_term_w * (size_t)(g_term_h * 2) * sizeof(RGB));
    g_ansibuf = malloc((size_t)g_term_w * (size_t)g_term_h * 64 + 2048);
    g_resized = 0;
}

static inline bool rgb_eq(RGB a, RGB b) {
    return a.r == b.r && a.g == b.g && a.b == b.b;
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
        pthread_create(&threads[i], NULL, render_worker_avx2, &thread_ids[i]);
    }

    struct timespec ts_prev, ts_now, ts_next_frame;
    clock_gettime(CLOCK_MONOTONIC, &ts_prev);
    ts_next_frame = ts_prev;
    float fps_compute = 60.0f;

    while (g_running) {
        if (g_resized) update_dimensions();

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
        g_scene_frame = prepare_scene_frame(g_time);

        pthread_mutex_lock(&g_mutex);
        g_active_workers = g_num_threads;
        g_frame_gen++;
        pthread_cond_broadcast(&g_cond_work);
        while (g_active_workers > 0) {
            pthread_cond_wait(&g_cond_done, &g_mutex);
        }
        pthread_mutex_unlock(&g_mutex);

        struct timespec ts_rendered;
        clock_gettime(CLOCK_MONOTONIC, &ts_rendered);
        float render_dt = (float)(ts_rendered.tv_sec - ts_now.tv_sec) +
                          (float)(ts_rendered.tv_nsec - ts_now.tv_nsec) * 1e-9f;
        if (render_dt > 0.00001f) {
            fps_compute = fps_compute * 0.9f + (1.0f / render_dt) * 0.1f;
        }

        /* --- Build Synchronized, RLE-Compressed ANSI Stream --- */
        char *p = g_ansibuf;

        /* DEC Mode 2026: Begin Synchronized Update (\033[?2026h) + Home Cursor (\033[H) */
        memcpy(p, "\033[?2026h\033[H", 11);
        p += 11;

        for (int row = 0; row < g_term_h; row++) {
            RGB *top_row = &g_framebuffer[(row * 2) * g_term_w];
            RGB *bot_row = &g_framebuffer[(row * 2 + 1) * g_term_w];

            bool has_fg = false, has_bg = false;
            RGB prev_fg = {0, 0, 0}, prev_bg = {0, 0, 0};

            for (int col = 0; col < g_term_w; col++) {
                RGB top = top_row[col];
                RGB bot = bot_row[col];

                /* Case 1: Both vertical sub-pixels share the exact same color -> emit single space ' ' */
                if (rgb_eq(top, bot)) {
                    if (!has_bg || !rgb_eq(bot, prev_bg)) {
                        memcpy(p, "\033[48;2;", 7); p += 7;
                        p = fast_utoa(p, bot.r); *p++ = ';';
                        p = fast_utoa(p, bot.g); *p++ = ';';
                        p = fast_utoa(p, bot.b); *p++ = 'm';
                        prev_bg = bot;
                        has_bg = true;
                    }
                    *p++ = ' ';
                    continue;
                }

                /* Case 2: Different top/bottom colors -> only emit changed FG or BG state */
                bool fg_changed = !has_fg || !rgb_eq(top, prev_fg);
                bool bg_changed = !has_bg || !rgb_eq(bot, prev_bg);

                if (fg_changed && bg_changed) {
                    memcpy(p, "\033[38;2;", 7); p += 7;
                    p = fast_utoa(p, top.r); *p++ = ';';
                    p = fast_utoa(p, top.g); *p++ = ';';
                    p = fast_utoa(p, top.b);
                    memcpy(p, ";48;2;", 6); p += 6;
                    p = fast_utoa(p, bot.r); *p++ = ';';
                    p = fast_utoa(p, bot.g); *p++ = ';';
                    p = fast_utoa(p, bot.b); *p++ = 'm';
                    prev_fg = top; prev_bg = bot;
                    has_fg = has_bg = true;
                } else if (fg_changed) {
                    memcpy(p, "\033[38;2;", 7); p += 7;
                    p = fast_utoa(p, top.r); *p++ = ';';
                    p = fast_utoa(p, top.g); *p++ = ';';
                    p = fast_utoa(p, top.b); *p++ = 'm';
                    prev_fg = top;
                    has_fg = true;
                } else if (bg_changed) {
                    memcpy(p, "\033[48;2;", 7); p += 7;
                    p = fast_utoa(p, bot.r); *p++ = ';';
                    p = fast_utoa(p, bot.g); *p++ = ';';
                    p = fast_utoa(p, bot.b); *p++ = 'm';
                    prev_bg = bot;
                    has_bg = true;
                }

                /* UTF-8 Upper Half Block U+2580: ▀ */
                *p++ = (char)0xE2; *p++ = (char)0x96; *p++ = (char)0x80;
            }
            memcpy(p, "\033[0m\r\n", 6);
            p += 6;
        }

        size_t payload_kb = (size_t)(p - g_ansibuf) / 1024;

        /* Telemetry HUD + DEC Mode 2026 End Synchronized Update (\033[?2026l) */
        int hud_len = snprintf(p, 512,
            "\033[48;2;20;20;35;38;2;0;255;200m ⚡ AVX2 VSYNC | %dx%d px | %zu KB/frame | %6.0f Compute FPS | [W/A/S/D] Orbit [SPACE] Pause [Q] Quit \033[K\033[0m\033[?2026l",
            g_term_w, g_term_h * 2, payload_kb, fps_compute);
        if (hud_len > 0) p += hud_len;

        write_all(STDOUT_FILENO, g_ansibuf, (size_t)(p - g_ansibuf));

        /* Absolute 60.0 Hz Monotonic Frame Pacing (16,666,667 ns) */
        ts_next_frame.tv_nsec += 16666667L;
        if (ts_next_frame.tv_nsec >= 1000000000L) {
            ts_next_frame.tv_sec += 1;
            ts_next_frame.tv_nsec -= 1000000000L;
        }

        struct timespec ts_check;
        clock_gettime(CLOCK_MONOTONIC, &ts_check);
        /* If terminal was slow and we fell behind deadline, resync instead of burst-catching up */
        if (ts_check.tv_sec > ts_next_frame.tv_sec ||
           (ts_check.tv_sec == ts_next_frame.tv_sec && ts_check.tv_nsec > ts_next_frame.tv_nsec)) {
            ts_next_frame = ts_check;
        } else {
            clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts_next_frame, NULL);
        }
    }

    pthread_mutex_lock(&g_mutex);
    g_shutdown_threads = true;
    pthread_cond_broadcast(&g_cond_work);
    pthread_mutex_unlock(&g_mutex);

    for (int i = 0; i < g_num_threads; i++) pthread_join(threads[i], NULL);
    free(threads); free(thread_ids); free(g_framebuffer); free(g_ansibuf);
    return 0;
}