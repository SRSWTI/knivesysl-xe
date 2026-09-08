#include "tq_common.hpp"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>

namespace {

constexpr int kMaxSeq = 262144;
constexpr size_t kSkipChunkBytes = 1u << 20;
constexpr size_t kUploadChunkBytes = 64u << 20;

bool checked_mul(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) return false;
    *out = a * b;
    return true;
}

bool checked_add(size_t a, size_t b, size_t *out) {
    if (b > std::numeric_limits<size_t>::max() - a) return false;
    *out = a + b;
    return true;
}

bool checked_int_mul(int a, int b, int *out) {
    if (a < 0 || b < 0) return false;
    const long long product = static_cast<long long>(a) * static_cast<long long>(b);
    if (product > INT_MAX) return false;
    *out = static_cast<int>(product);
    return true;
}

bool selector_has(const char *selector, const char *key) {
    if (!selector || !*selector || !key || !*key) return false;
    const size_t key_len = std::strlen(key);
    const char *p = selector;
    while (*p) {
        while (*p == ',' || *p == ' ' || *p == '\t') ++p;
        const char *begin = p;
        while (*p && *p != ',') ++p;
        const char *end = p;
        while (end > begin && (end[-1] == ' ' || end[-1] == '\t')) --end;
        const size_t len = static_cast<size_t>(end - begin);
        if ((len == 3 && std::memcmp(begin, "all", 3) == 0) ||
            (len == key_len && std::memcmp(begin, key, key_len) == 0)) {
            return true;
        }
    }
    return false;
}

int read_exact(FILE *f, void *dst, size_t nbytes, const char *what) {
    if (nbytes == 0) return 0;
    const size_t got = std::fread(dst, 1, nbytes, f);
    if (got != nbytes) {
        std::fprintf(stderr, "TQF read failed for %s: expected %zu bytes, got %zu\n",
                     what, nbytes, got);
        return -1;
    }
    return 0;
}

int skip_exact(FILE *f, size_t nbytes, const char *what) {
    if (nbytes == 0) return 0;
    const size_t capacity = std::min(nbytes, kSkipChunkBytes);
    auto *buffer = static_cast<uint8_t *>(std::malloc(capacity));
    if (!buffer) {
        std::fprintf(stderr, "malloc failed while skipping %s (%zu bytes)\n", what, nbytes);
        return -1;
    }

    size_t left = nbytes;
    while (left != 0) {
        const size_t now = std::min(left, capacity);
        if (read_exact(f, buffer, now, what) != 0) {
            std::free(buffer);
            return -1;
        }
        left -= now;
    }
    std::free(buffer);
    return 0;
}

int upload_payload(FILE *f, void **dst, size_t nbytes, const char *what, int upload) {
    if (nbytes == 0) {
        *dst = nullptr;
        return 0;
    }
    if (!upload) return skip_exact(f, nbytes, what);

    try {
        *dst = sycl::malloc_device(nbytes, tq_q());
    } catch (const std::exception &e) {
        std::fprintf(stderr, "device allocation failed for %s (%zu bytes): %s\n",
                     what, nbytes, e.what());
        *dst = nullptr;
        return -1;
    } catch (...) {
        std::fprintf(stderr, "device allocation failed for %s (%zu bytes)\n", what, nbytes);
        *dst = nullptr;
        return -1;
    }
    if (!*dst) {
        std::fprintf(stderr, "device allocation returned null for %s (%zu bytes)\n", what, nbytes);
        return -1;
    }

    const size_t capacity = std::min(nbytes, kUploadChunkBytes);
    // Host-USM staging: the H2D copy from imported host memory runs ~1.7x
    // faster than from plain malloc (opt guide measured 26.9 -> 45.4 GB/s;
    // level-up queue item 10). Fall back to malloc if USM alloc fails.
    bool host_is_usm = true;
    auto *host = static_cast<uint8_t *>(sycl::malloc_host(capacity, tq_q()));
    if (!host) {
        host_is_usm = false;
        host = static_cast<uint8_t *>(std::malloc(capacity));
    }
    if (!host) {
        std::fprintf(stderr, "staging alloc failed for %s (%zu bytes)\n", what,
                     capacity);
        return -1;
    }
    auto free_host = [&]() {
        if (host_is_usm) sycl::free(host, tq_q());
        else std::free(host);
    };

    size_t offset = 0;
    while (offset != nbytes) {
        const size_t now = std::min(nbytes - offset, capacity);
        if (read_exact(f, host, now, what) != 0) {
            free_host();
            return -1;
        }
        try {
            tq_h2d(static_cast<uint8_t *>(*dst) + offset, host, now);
            // The staging allocation is immediately reused/freed. Waiting here is
            // required even though all device work otherwise uses the in-order queue.
            tq_q().wait_and_throw();
        } catch (const std::exception &e) {
            std::fprintf(stderr, "device upload failed for %s at byte %zu: %s\n",
                         what, offset, e.what());
            free_host();
            return -1;
        } catch (...) {
            std::fprintf(stderr, "device upload failed for %s at byte %zu\n", what, offset);
            free_host();
            return -1;
        }
        offset += now;
    }

    free_host();
    return 0;
}

int bytes_for_elements(size_t count, size_t element_bytes, size_t *nbytes,
                       const char *what) {
    if (!checked_mul(count, element_bytes, nbytes)) {
        std::fprintf(stderr, "%s byte size overflows size_t\n", what);
        return -1;
    }
    return 0;
}

int read_non_quant(FILE *f, uint16_t **dst, size_t count, const char *what, int upload) {
    size_t nbytes = 0;
    if (bytes_for_elements(count, sizeof(uint16_t), &nbytes, what) != 0) return -1;
    return upload_payload(f, reinterpret_cast<void **>(dst), nbytes, what, upload);
}

int read_f32(FILE *f, float **dst, size_t count, const char *what, int upload) {
    size_t nbytes = 0;
    if (bytes_for_elements(count, sizeof(float), &nbytes, what) != 0) return -1;
    return upload_payload(f, reinterpret_cast<void **>(dst), nbytes, what, upload);
}

int read_qmma(FILE *f, tq_qmma_weight_t *w, int M, int K, const char *what, int upload) {
    if (M <= 0 || K <= 0 || (M % 16) != 0 || (K % 32) != 0) {
        std::fprintf(stderr, "%s: unaligned QMMA shape %dx%d\n", what, M, K);
        return -1;
    }

    w->M = M;
    w->K = K;
    w->Mt = M / 16;
    w->Kt = K / 32;
    w->block_scaled = (g_qwen.flags & TQ_FLAG_BLOCK_SCALED_QMMA) != 0;
    w->row_major = (g_qwen.flags & TQ_FLAG_BLOCK_SCALED_ROWMAJOR) != 0;
    w->word_major = (g_qwen.flags & TQ_FLAG_BLOCK_SCALED_QMMA_WORDMAJOR) != 0;
    w->e2m3 = (g_qwen.flags & TQ_FLAG_BLOCK_SCALED_E2M3) != 0;
    w->e2m1 = 0;
    w->e2m3_byte = 0;
    w->d_s4 = nullptr;
    w->d_s4_scale = nullptr;
    w->s4_ready = 0;
    w->d_s8 = nullptr;
    w->d_s8_scale = nullptr;
    w->s8_ready = 0;

    if (!w->block_scaled || !w->e2m3) {
        std::fprintf(stderr, "%s: Phase-1 XPU requires block-scaled dense E2M3 QMMA\n", what);
        return -1;
    }

    uint32_t scale_rows = 0;
    uint32_t scale_cols = 0;
    if (read_exact(f, &scale_rows, sizeof(scale_rows), what) != 0 ||
        read_exact(f, &scale_cols, sizeof(scale_cols), what) != 0) {
        return -1;
    }
    const uint32_t expected_rows = static_cast<uint32_t>((M + 127) / 128);
    const uint32_t expected_cols = static_cast<uint32_t>((K + 127) / 128);
    const uint32_t expected_cols_k32 = static_cast<uint32_t>((K + 31) / 32);
    if (scale_rows != expected_rows ||
        (scale_cols != expected_cols && scale_cols != expected_cols_k32)) {
        std::fprintf(stderr, "%s: invalid block scale shape %ux%u expected %ux%u\n",
                     what, scale_rows, scale_cols, expected_rows, expected_cols);
        return -1;
    }
    w->scale_rows = static_cast<int>(scale_rows);
    w->scale_cols = static_cast<int>(scale_cols);

    size_t scale_count = 0;
    size_t scale_bytes = 0;
    if (!checked_mul(static_cast<size_t>(scale_rows), static_cast<size_t>(scale_cols),
                     &scale_count) ||
        bytes_for_elements(scale_count, sizeof(float), &scale_bytes, what) != 0) {
        std::fprintf(stderr, "%s: block scale size overflows size_t\n", what);
        return -1;
    }
    if (upload_payload(f, reinterpret_cast<void **>(&w->d_block_scale_inv), scale_bytes,
                       what, upload) != 0) {
        return -1;
    }
    w->inv_scale = 1.0f;

    size_t tile_count = 0;
    size_t payload_bytes = 0;
    if (!checked_mul(static_cast<size_t>(w->Mt), static_cast<size_t>(w->Kt),
                     &tile_count) ||
        !checked_mul(tile_count, tq_w_tile_bytes(w), &payload_bytes)) {
        std::fprintf(stderr, "%s: QMMA payload size overflows size_t\n", what);
        return -1;
    }
    return upload_payload(f, reinterpret_cast<void **>(&w->d_A), payload_bytes, what, upload);
}

template <typename T>
void free_device(T *&ptr) {
    if (ptr) tq_dev_free(ptr);
    ptr = nullptr;
}

void free_qmma(tq_qmma_weight_t *w) {
    free_device(w->d_A);
    free_device(w->d_block_scale_inv);
    free_device(w->d_s4);
    free_device(w->d_s4_scale);
    free_device(w->d_s8);
    free_device(w->d_s8_scale);
    std::memset(w, 0, sizeof(*w));
}

int parse_failure(FILE *f) {
    if (f) std::fclose(f);
    qwn_free();
    return -1;
}

int validate_header_fields(const uint32_t u[16]) {
    for (int i = 3; i < 16; ++i) {
        if (u[i] > static_cast<uint32_t>(INT_MAX)) {
            std::fprintf(stderr, "TQF integer header field %d exceeds INT_MAX\n", i);
            return -1;
        }
    }
    if (g_qwen.H <= 0 || g_qwen.I <= 0 || g_qwen.L <= 0 ||
        g_qwen.L > TQ_MAX_LAYERS || g_qwen.V <= 0 || g_qwen.nh <= 0 ||
        g_qwen.nkv <= 0 || g_qwen.hd <= 0) {
        std::fprintf(stderr,
                     "Invalid TQF dimensions: H=%d I=%d L=%d V=%d nh=%d nkv=%d hd=%d\n",
                     g_qwen.H, g_qwen.I, g_qwen.L, g_qwen.V,
                     g_qwen.nh, g_qwen.nkv, g_qwen.hd);
        return -1;
    }
    if (g_qwen.non_quant_dtype != TQ_DTYPE_BF16) {
        std::fprintf(stderr, "Phase-1 XPU requires BF16 non-quant tensors (dtype=%u)\n",
                     g_qwen.non_quant_dtype);
        return -1;
    }
    const uint32_t required = TQ_FLAG_FP8_QMMA_WEIGHTS |
                              TQ_FLAG_BLOCK_SCALED_QMMA |
                              TQ_FLAG_BLOCK_SCALED_E2M3;
    if ((g_qwen.flags & required) != required) {
        std::fprintf(stderr,
                     "Phase-1 XPU requires FP8_QMMA_WEIGHTS|BLOCK_SCALED_QMMA|"
                     "BLOCK_SCALED_E2M3 (flags=0x%x)\n",
                     g_qwen.flags);
        return -1;
    }
    if (g_qwen.flags & (TQ_FLAG_BLOCK_SCALED_ROWMAJOR |
                        TQ_FLAG_BLOCK_SCALED_QMMA_WORDMAJOR |
                        TQ_FLAG_SPARSE_24_E2M3)) {
        std::fprintf(stderr,
                     "Phase-1 XPU supports only dense E2M3 QMMA-fragment payloads "
                     "(flags=0x%x)\n",
                     g_qwen.flags);
        return -1;
    }
    return 0;
}

int skip_mtp_section(FILE *f, long start_offset) {
    const size_t H = static_cast<size_t>(g_qwen.H);
    const size_t I = static_cast<size_t>(g_qwen.I);
    const size_t hd = static_cast<size_t>(g_qwen.hd);
    size_t q_rows = 0;
    size_t kv_rows = 0;
    size_t attn_rows = 0;
    size_t fc_count = 0;
    size_t q_count = 0;
    size_t kv_count = 0;
    size_t o_count = 0;
    size_t mlp_in_count = 0;
    size_t mlp_down_count = 0;

    if (!checked_mul(static_cast<size_t>(g_qwen.nh), hd, &attn_rows) ||
        !checked_mul(attn_rows, 2, &q_rows) ||
        !checked_mul(static_cast<size_t>(g_qwen.nkv), hd, &kv_rows) ||
        !checked_mul(H, 2, &fc_count) ||
        !checked_mul(H, fc_count, &fc_count) ||
        !checked_mul(q_rows, H, &q_count) ||
        !checked_mul(kv_rows, H, &kv_count) ||
        !checked_mul(H, attn_rows, &o_count) ||
        !checked_mul(I, H, &mlp_in_count) ||
        !checked_mul(H, I, &mlp_down_count)) {
        std::fprintf(stderr, "MTP tensor size overflows size_t\n");
        return -1;
    }

    struct TensorToSkip {
        const char *name;
        size_t count;
    };
    const TensorToSkip tensors[] = {
        {"mtp.pre_fc_norm_emb", H},
        {"mtp.pre_fc_norm_hidden", H},
        {"mtp.fc", fc_count},
        {"mtp.input_ln", H},
        {"mtp.post_ln", H},
        {"mtp.q_norm", hd},
        {"mtp.k_norm", hd},
        {"mtp.q_proj", q_count},
        {"mtp.k_proj", kv_count},
        {"mtp.v_proj", kv_count},
        {"mtp.o_proj", o_count},
        {"mtp.mlp_gate", mlp_in_count},
        {"mtp.mlp_up", mlp_in_count},
        {"mtp.mlp_down", mlp_down_count},
        {"mtp.norm", H},
    };

    size_t skipped_bytes = 0;
    for (const TensorToSkip &tensor : tensors) {
        size_t tensor_bytes = 0;
        size_t new_total = 0;
        if (bytes_for_elements(tensor.count, sizeof(uint16_t), &tensor_bytes,
                               tensor.name) != 0 ||
            !checked_add(skipped_bytes, tensor_bytes, &new_total)) {
            std::fprintf(stderr, "MTP section size overflows size_t\n");
            return -1;
        }
        if (skip_exact(f, tensor_bytes, tensor.name) != 0) return -1;
        skipped_bytes = new_total;
    }

    const long end_offset = std::ftell(f);
    std::printf("TQF1 MTP section skipped (Phase 1): 15 BF16 tensors, %zu bytes, "
                "offset %ld -> %ld\n",
                skipped_bytes, start_offset, end_offset);
    return 0;
}

static int parse_tqf(const char *path, int upload) {
    qwn_free();
    if (!path || !*path) {
        std::fprintf(stderr, "TQF path is empty\n");
        return -1;
    }

    FILE *f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "Cannot open %s\n", path);
        return -1;
    }

    char magic[4] = {};
    uint32_t header_bytes = 0;
    uint32_t u[16] = {};
    float fl[3] = {};
    uint32_t tail[8] = {};
    if (read_exact(f, magic, sizeof(magic), "magic") != 0 ||
        std::memcmp(magic, "TQF1", sizeof(magic)) != 0) {
        if (std::memcmp(magic, "TQF1", sizeof(magic)) != 0)
            std::fprintf(stderr, "Bad TQF magic\n");
        return parse_failure(f);
    }
    if (read_exact(f, &header_bytes, sizeof(header_bytes), "header_bytes") != 0 ||
        read_exact(f, u, sizeof(u), "fixed uint header") != 0 ||
        read_exact(f, fl, sizeof(fl), "fixed float header") != 0 ||
        read_exact(f, tail, sizeof(tail), "fixed tail header") != 0) {
        return parse_failure(f);
    }

    g_qwen.flags = u[0];
    g_qwen.model_family = u[1];
    g_qwen.non_quant_dtype = u[2];
    g_qwen.H = static_cast<int>(u[3]);
    g_qwen.I = static_cast<int>(u[4]);
    g_qwen.L = static_cast<int>(u[5]);
    g_qwen.V = static_cast<int>(u[6]);
    g_qwen.nh = static_cast<int>(u[7]);
    g_qwen.nkv = static_cast<int>(u[8]);
    g_qwen.hd = static_cast<int>(u[9]);
    g_qwen.linear_num_key_heads = static_cast<int>(u[10]);
    g_qwen.linear_key_head_dim = static_cast<int>(u[11]);
    g_qwen.linear_num_value_heads = static_cast<int>(u[12]);
    g_qwen.linear_value_head_dim = static_cast<int>(u[13]);
    g_qwen.linear_conv_kernel_dim = static_cast<int>(u[14]);
    g_qwen.max_position_embeddings = static_cast<int>(u[15]);
    g_qwen.eps = fl[0];
    g_qwen.rope_theta = fl[1];
    g_qwen.partial_rotary_factor = fl[2];
    g_qwen.mrope_interleaved = static_cast<int>(tail[0]);
    g_qwen.mrope_section[0] = static_cast<int>(tail[1]);
    g_qwen.mrope_section[1] = static_cast<int>(tail[2]);
    g_qwen.mrope_section[2] = static_cast<int>(tail[3]);
    g_qwen.tie_word_embeddings = static_cast<int>(tail[4]);
    g_qwen.text_only = static_cast<int>(tail[5]);
    g_qwen.has_vision_config = static_cast<int>(tail[6]);
    g_qwen.has_mtp = static_cast<int>(tail[7]);
    g_qwen.has_mtp_section = (g_qwen.flags & TQ_FLAG_HAS_MTP) != 0;

    if (validate_header_fields(u) != 0) return parse_failure(f);
    const uint32_t expected_header = 4u + 4u + 16u * 4u + 3u * 4u +
                                     8u * 4u + static_cast<uint32_t>(g_qwen.L);
    if (header_bytes != expected_header) {
        std::fprintf(stderr, "Unexpected header_bytes=%u expected=%u\n",
                     header_bytes, expected_header);
        return parse_failure(f);
    }
    if (read_exact(f, g_qwen.layer_types, static_cast<size_t>(g_qwen.L),
                   "layer types") != 0) {
        return parse_failure(f);
    }

    bool has_linear_layer = false;
    for (int i = 0; i < g_qwen.L; ++i) {
        const uint8_t type = g_qwen.layer_types[i];
        if (type == TQ_LAYER_LINEAR_ATTENTION) {
            has_linear_layer = true;
        } else if (type != TQ_LAYER_FULL_ATTENTION) {
            std::fprintf(stderr, "Unsupported layer type byte at layer %d: %u\n", i, type);
            return parse_failure(f);
        }
    }
    if (has_linear_layer &&
        (g_qwen.linear_num_key_heads <= 0 || g_qwen.linear_key_head_dim <= 0 ||
         g_qwen.linear_num_value_heads <= 0 || g_qwen.linear_value_head_dim <= 0 ||
         g_qwen.linear_conv_kernel_dim <= 0)) {
        std::fprintf(stderr, "Invalid linear-attention dimensions in TQF header\n");
        return parse_failure(f);
    }

    int linear_key_dim = 0;
    int linear_value_dim = 0;
    int twice_linear_key_dim = 0;
    int linear_conv_dim = 0;
    int full_q_rows = 0;
    int full_kv_rows = 0;
    int full_attn_rows = 0;
    if (!checked_int_mul(g_qwen.linear_num_key_heads, g_qwen.linear_key_head_dim,
                         &linear_key_dim) ||
        !checked_int_mul(g_qwen.linear_num_value_heads, g_qwen.linear_value_head_dim,
                         &linear_value_dim) ||
        !checked_int_mul(linear_key_dim, 2, &twice_linear_key_dim) ||
        twice_linear_key_dim > INT_MAX - linear_value_dim ||
        !checked_int_mul(g_qwen.nh, g_qwen.hd, &full_attn_rows) ||
        !checked_int_mul(full_attn_rows, 2, &full_q_rows) ||
        !checked_int_mul(g_qwen.nkv, g_qwen.hd, &full_kv_rows)) {
        std::fprintf(stderr, "Derived TQF dimensions overflow int\n");
        return parse_failure(f);
    }
    linear_conv_dim = twice_linear_key_dim + linear_value_dim;

    std::printf("TQF1: H=%d I=%d L=%d V=%d nh=%d nkv=%d hd=%d dtype=%u flags=0x%x\n",
                g_qwen.H, g_qwen.I, g_qwen.L, g_qwen.V, g_qwen.nh, g_qwen.nkv,
                g_qwen.hd, g_qwen.non_quant_dtype, g_qwen.flags);

    size_t embed_count = 0;
    if (!checked_mul(static_cast<size_t>(g_qwen.V), static_cast<size_t>(g_qwen.H),
                     &embed_count)) {
        std::fprintf(stderr, "Embedding tensor size overflows size_t\n");
        return parse_failure(f);
    }
    if (read_non_quant(f, &g_qwen.d_embed, embed_count, "embed", upload) != 0 ||
        read_non_quant(f, &g_qwen.d_norm, static_cast<size_t>(g_qwen.H), "norm", upload) != 0) {
        return parse_failure(f);
    }
    if (!g_qwen.tie_word_embeddings &&
        read_qmma(f, &g_qwen.lm_head, g_qwen.V, g_qwen.H, "lm_head", upload) != 0) {
        return parse_failure(f);
    }

    for (int i = 0; i < g_qwen.L; ++i) {
        tq_layer_t *layer = &g_qwen.layers[i];
        char what[128];
        std::snprintf(what, sizeof(what), "layer%d.input_ln", i);
        if (read_non_quant(f, &layer->d_input_ln, static_cast<size_t>(g_qwen.H),
                           what, upload) != 0) return parse_failure(f);
        std::snprintf(what, sizeof(what), "layer%d.post_ln", i);
        if (read_non_quant(f, &layer->d_post_ln, static_cast<size_t>(g_qwen.H),
                           what, upload) != 0) return parse_failure(f);
        std::snprintf(what, sizeof(what), "layer%d.mlp_gate", i);
        if (read_qmma(f, &layer->mlp_gate, g_qwen.I, g_qwen.H, what, upload) != 0)
            return parse_failure(f);
        std::snprintf(what, sizeof(what), "layer%d.mlp_up", i);
        if (read_qmma(f, &layer->mlp_up, g_qwen.I, g_qwen.H, what, upload) != 0)
            return parse_failure(f);
        std::snprintf(what, sizeof(what), "layer%d.mlp_down", i);
        if (read_qmma(f, &layer->mlp_down, g_qwen.H, g_qwen.I, what, upload) != 0)
            return parse_failure(f);

        if (g_qwen.layer_types[i] == TQ_LAYER_FULL_ATTENTION) {
            std::snprintf(what, sizeof(what), "layer%d.q_norm", i);
            if (read_non_quant(f, &layer->d_q_norm, static_cast<size_t>(g_qwen.hd),
                               what, upload) != 0) return parse_failure(f);
            std::snprintf(what, sizeof(what), "layer%d.k_norm", i);
            if (read_non_quant(f, &layer->d_k_norm, static_cast<size_t>(g_qwen.hd),
                               what, upload) != 0) return parse_failure(f);
            std::snprintf(what, sizeof(what), "layer%d.q_proj", i);
            if (read_qmma(f, &layer->q_proj, full_q_rows, g_qwen.H, what, upload) != 0)
                return parse_failure(f);
            std::snprintf(what, sizeof(what), "layer%d.k_proj", i);
            if (read_qmma(f, &layer->k_proj, full_kv_rows, g_qwen.H, what, upload) != 0)
                return parse_failure(f);
            std::snprintf(what, sizeof(what), "layer%d.v_proj", i);
            if (read_qmma(f, &layer->v_proj, full_kv_rows, g_qwen.H, what, upload) != 0)
                return parse_failure(f);
            std::snprintf(what, sizeof(what), "layer%d.o_proj", i);
            if (read_qmma(f, &layer->o_proj, g_qwen.H, full_attn_rows, what, upload) != 0)
                return parse_failure(f);
        } else {
            std::snprintf(what, sizeof(what), "layer%d.linear_A_log", i);
            if (read_f32(f, &layer->d_linear_A_log,
                         static_cast<size_t>(g_qwen.linear_num_value_heads), what, upload) != 0)
                return parse_failure(f);
            std::snprintf(what, sizeof(what), "layer%d.linear_dt_bias", i);
            if (read_non_quant(f, &layer->d_linear_dt_bias,
                               static_cast<size_t>(g_qwen.linear_num_value_heads), what,
                               upload) != 0) return parse_failure(f);
            std::snprintf(what, sizeof(what), "layer%d.linear_norm", i);
            if (read_f32(f, &layer->d_linear_norm,
                         static_cast<size_t>(g_qwen.linear_value_head_dim), what, upload) != 0)
                return parse_failure(f);

            size_t conv_count = 0;
            if (!checked_mul(static_cast<size_t>(linear_conv_dim),
                             static_cast<size_t>(g_qwen.linear_conv_kernel_dim),
                             &conv_count)) {
                std::fprintf(stderr, "%s: convolution tensor size overflows size_t\n", what);
                return parse_failure(f);
            }
            std::snprintf(what, sizeof(what), "layer%d.linear_conv1d", i);
            if (read_non_quant(f, &layer->d_linear_conv1d, conv_count, what, upload) != 0)
                return parse_failure(f);
            std::snprintf(what, sizeof(what), "layer%d.linear_in_a", i);
            if (read_qmma(f, &layer->linear_in_a, g_qwen.linear_num_value_heads,
                          g_qwen.H, what, upload) != 0) return parse_failure(f);
            std::snprintf(what, sizeof(what), "layer%d.linear_in_b", i);
            if (read_qmma(f, &layer->linear_in_b, g_qwen.linear_num_value_heads,
                          g_qwen.H, what, upload) != 0) return parse_failure(f);
            std::snprintf(what, sizeof(what), "layer%d.linear_in_qkv", i);
            if (read_qmma(f, &layer->linear_in_qkv, linear_conv_dim,
                          g_qwen.H, what, upload) != 0) return parse_failure(f);
            std::snprintf(what, sizeof(what), "layer%d.linear_in_z", i);
            if (read_qmma(f, &layer->linear_in_z, linear_value_dim,
                          g_qwen.H, what, upload) != 0) return parse_failure(f);
            std::snprintf(what, sizeof(what), "layer%d.linear_out", i);
            if (read_qmma(f, &layer->linear_out, g_qwen.H,
                          linear_value_dim, what, upload) != 0) return parse_failure(f);
        }
        if ((i % 6) == 0) std::printf("  layer %d/%d parsed\n", i, g_qwen.L);
    }

    if (g_qwen.has_mtp_section) {
        const long mtp_start = std::ftell(f);
        if (skip_mtp_section(f, mtp_start) != 0) return parse_failure(f);
    }

    const long consumed = std::ftell(f);
    if (consumed < 0 || std::fseek(f, 0, SEEK_END) != 0) {
        std::fprintf(stderr, "Unable to determine TQF payload position\n");
        return parse_failure(f);
    }
    const long end = std::ftell(f);
    std::fclose(f);
    if (end < 0 || consumed != end) {
        std::fprintf(stderr, "TQF payload mismatch: stopped at %ld of %ld\n", consumed, end);
        qwn_free();
        return -1;
    }

    g_qwen.initialized = upload ? 1 : 0;
    std::printf("TQF1 %s complete: consumed %ld bytes\n", upload ? "load" : "probe", end);
    return 0;
}

template <typename T>
int allocate_elements(T **dst, size_t count, const char *what) {
    if (count == 0) {
        *dst = nullptr;
        return 0;
    }
    size_t nbytes = 0;
    if (bytes_for_elements(count, sizeof(T), &nbytes, what) != 0) return -1;
    try {
        *dst = static_cast<T *>(sycl::malloc_device(nbytes, tq_q()));
    } catch (const std::exception &e) {
        std::fprintf(stderr, "device allocation failed for %s (%zu bytes): %s\n",
                     what, nbytes, e.what());
        *dst = nullptr;
        return -1;
    } catch (...) {
        std::fprintf(stderr, "device allocation failed for %s (%zu bytes)\n", what, nbytes);
        *dst = nullptr;
        return -1;
    }
    if (!*dst) {
        std::fprintf(stderr, "device allocation returned null for %s (%zu bytes)\n", what, nbytes);
        return -1;
    }
    return 0;
}

struct tq_state_sizes_t {
    size_t conv;
    size_t recurrent;
    size_t kv;
    size_t kv_rows;
};

bool get_state_sizes(tq_state_sizes_t *out) {
    int linear_key_dim = 0;
    int linear_value_dim = 0;
    int twice_linear_key_dim = 0;
    int linear_conv_dim = 0;
    int full_kv_dim = 0;
    if (!checked_int_mul(g_qwen.linear_num_key_heads, g_qwen.linear_key_head_dim,
                         &linear_key_dim) ||
        !checked_int_mul(g_qwen.linear_num_value_heads,
                         g_qwen.linear_value_head_dim, &linear_value_dim) ||
        !checked_int_mul(linear_key_dim, 2, &twice_linear_key_dim) ||
        twice_linear_key_dim > INT_MAX - linear_value_dim ||
        !checked_int_mul(g_qwen.nkv, g_qwen.hd, &full_kv_dim)) {
        return false;
    }
    linear_conv_dim = twice_linear_key_dim + linear_value_dim;

    size_t recurrent_heads_dims = 0;
    return checked_mul(static_cast<size_t>(linear_conv_dim),
                       static_cast<size_t>(g_qwen.linear_conv_kernel_dim),
                       &out->conv) &&
           checked_mul(static_cast<size_t>(g_qwen.linear_num_value_heads),
                       static_cast<size_t>(g_qwen.linear_key_head_dim),
                       &recurrent_heads_dims) &&
           checked_mul(recurrent_heads_dims,
                       static_cast<size_t>(g_qwen.linear_value_head_dim),
                       &out->recurrent) &&
           checked_mul(static_cast<size_t>(g_qwen.max_seq),
                       static_cast<size_t>(full_kv_dim), &out->kv) &&
           checked_mul(static_cast<size_t>(g_qwen.max_seq),
                       static_cast<size_t>(g_qwen.nkv), &out->kv_rows);
}

bool state_buffers_ready(bool paged) {
    for (int i = 0; i < g_qwen.L; ++i) {
        const tq_layer_t *layer = &g_qwen.layers[i];
        if (g_qwen.layer_types[i] == TQ_LAYER_LINEAR_ATTENTION) {
            if (!layer->d_linear_conv_state ||
                !layer->d_linear_recurrent_state) {
                return false;
            }
        } else if (g_qwen.layer_types[i] == TQ_LAYER_FULL_ATTENTION &&
                   !paged &&
                   (!layer->d_k_cache || !layer->d_v_cache ||
                    !layer->d_k_scale || !layer->d_v_scale)) {
            return false;
        }
    }
    return true;
}

int paged_mode_from_env() {
    const char *mode = std::getenv("TQ_XPU_PAGED");
    if (!mode) return 0;
    if (mode[0] == '0' && mode[1] == '\0') return 0;
    if (mode[0] == '1' && mode[1] == '\0') return 1;
    std::fprintf(stderr, "TQ_XPU_PAGED must be exactly 0 or 1\n");
    return -1;
}

int parse_paged_int_env(const char *name, int default_value,
                        int minimum, int maximum, int *out) {
    const char *value = std::getenv(name);
    if (!value) {
        *out = default_value;
        return 0;
    }
    char *end = nullptr;
    errno = 0;
    const long long parsed = std::strtoll(value, &end, 10);
    if (errno == ERANGE || end == value || *end != '\0' ||
        parsed < minimum || parsed > maximum) {
        std::fprintf(stderr, "%s must be an integer in [%d, %d] (got '%s')\n",
                     name, minimum, maximum, value);
        return -1;
    }
    *out = static_cast<int>(parsed);
    return 0;
}

int paged_prefill_warm_tokens(int max_seq, int *out) {
    long long configured = 512;
    if (const char *env = std::getenv("TQ_XPU_PREFILL_CHUNK")) {
        if (!*env) {
            std::fprintf(stderr, "TQ_XPU_PREFILL_CHUNK must not be empty\n");
            return -1;
        }
        char *end = nullptr;
        errno = 0;
        const long long parsed = std::strtoll(env, &end, 10);
        if (errno == ERANGE || end == env || *end != '\0' ||
            parsed < 8 || parsed > INT_MAX || (parsed % 8) != 0) {
            std::fprintf(stderr,
                         "TQ_XPU_PREFILL_CHUNK must be a positive multiple "
                         "of 8 not exceeding INT_MAX (got '%s')\n",
                         env);
            return -1;
        }
        configured = parsed;
    }
    const int context_bound = max_seq - (max_seq % 8);
    *out = static_cast<int>(
        std::min<long long>(configured, static_cast<long long>(context_bound)));
    return 0;
}

}  // namespace

tq_model_t g_qwen_ranks[kMaxTP];
tq_model_t *g_qwen_active = &g_qwen_ranks[0];

extern "C" void qwn_free(void) {
    const bool model_has_device_storage =
        g_qwen.initialized || g_qwen.d_embed || g_qwen.d_norm ||
        g_qwen.lm_head.d_A;
    if (model_has_device_storage || tq_paged_ready()) {
        try {
            tq_q().wait_and_throw();
        } catch (const std::exception &e) {
            std::fprintf(stderr, "device queue error while freeing Qwen model: %s\n", e.what());
        } catch (...) {
            std::fprintf(stderr, "device queue error while freeing Qwen model\n");
        }
    }

    // Decode scratch and paged ownership use this model's queue and shape.
    // Destroy them before releasing model storage or clearing g_qwen.
    tq_decode_runtime_free();
    tq_paged_free();

    free_device(g_qwen.d_embed);
    free_device(g_qwen.d_norm);
    free_qmma(&g_qwen.lm_head);

    for (int i = 0; i < g_qwen.L && i < TQ_MAX_LAYERS; ++i) {
        tq_layer_t *layer = &g_qwen.layers[i];
        free_device(layer->d_input_ln);
        free_device(layer->d_post_ln);
        free_qmma(&layer->mlp_gate);
        free_qmma(&layer->mlp_up);
        free_qmma(&layer->mlp_down);
        free_device(layer->d_q_norm);
        free_device(layer->d_k_norm);
        free_qmma(&layer->q_proj);
        free_qmma(&layer->k_proj);
        free_qmma(&layer->v_proj);
        free_qmma(&layer->o_proj);
        free_device(layer->d_linear_A_log);
        free_device(layer->d_linear_dt_bias);
        free_device(layer->d_linear_norm);
        free_device(layer->d_linear_conv1d);
        free_qmma(&layer->linear_in_a);
        free_qmma(&layer->linear_in_b);
        free_qmma(&layer->linear_in_qkv);
        free_qmma(&layer->linear_in_z);
        free_qmma(&layer->linear_out);
        free_device(layer->d_linear_conv_state);
        free_device(layer->d_linear_recurrent_state);
        free_device(layer->d_k_cache);
        free_device(layer->d_v_cache);
        free_device(layer->d_k_scale);
        free_device(layer->d_v_scale);
    }

    free_device(g_qwen.d_x);
    free_device(g_qwen.d_norm_out);
    free_device(g_qwen.d_qkv);
    free_device(g_qwen.d_z);
    free_device(g_qwen.d_b);
    free_device(g_qwen.d_a);
    free_device(g_qwen.d_core);
    free_device(g_qwen.d_resid);
    free_device(g_qwen.d_post_norm);
    free_device(g_qwen.d_gate);
    free_device(g_qwen.d_up);
    free_device(g_qwen.d_mlp_hidden);
    free_device(g_qwen.d_layer_out);
    free_device(g_qwen.d_proj_out);
    free_device(g_qwen.d_logits);
    free_device(g_qwen.d_scores);
    free_device(g_qwen.d_argmax_vals);
    free_device(g_qwen.d_argmax_ids);
    std::memset(&g_qwen, 0, sizeof(g_qwen));
}

extern "C" int qwn_probe(const char *path) {
    return parse_tqf(path, 0);
}

extern "C" int qwn_init(const char *path) {
    // Reject unsupported paged TP before loading weights: parse_tqf's first
    // device allocation creates the process-wide SYCL context and queues.
    const int paged_mode = paged_mode_from_env();
    int requested_tp = 1;
    if (paged_mode < 0 ||
        (paged_mode &&
         (parse_paged_int_env("TQ_XPU_TP", 1, 1, INT_MAX,
                              &requested_tp) != 0 ||
          requested_tp != 1))) {
        if (paged_mode > 0 && requested_tp != 1)
            std::fprintf(stderr, "paged KV requires TQ_XPU_TP=1\n");
        qwn_free();
        return -4;
    }

    const int parse_result = parse_tqf(path, 1);
    if (parse_result != 0) return parse_result;

    long long context = 4096;
    const char *env = std::getenv("TQ_CTX");
    if (paged_mode) {
        int paged_context = 4096;
        if (parse_paged_int_env("TQ_CTX", 4096, 8, kMaxSeq,
                                &paged_context) != 0) {
            qwn_free();
            return -4;
        }
        context = paged_context;
    } else {
        if (env && *env) {
            char *end = nullptr;
            errno = 0;
            const long long parsed = std::strtoll(env, &end, 10);
            context = (end == env) ? 0 : parsed;
        }
        context = std::max<long long>(
            2048, std::min<long long>(context, kMaxSeq));
    }
    g_qwen.max_seq = static_cast<int>(context);
    std::fprintf(stderr, "context limit: %d tokens (TQ_CTX, cap %d)\n",
                 g_qwen.max_seq, kMaxSeq);

    int full_attn_rows = 0;
    int full_q_rows = 0;
    int linear_key_dim = 0;
    int linear_decode_key_dim = 0;
    int linear_value_dim = 0;
    int linear_conv_dim = 0;
    int linear_decode_conv_dim = 0;
    int twice_linear_key_dim = 0;
    int twice_linear_decode_key_dim = 0;
    if (!checked_int_mul(g_qwen.nh, g_qwen.hd, &full_attn_rows) ||
        !checked_int_mul(full_attn_rows, 2, &full_q_rows) ||
        !checked_int_mul(g_qwen.linear_num_key_heads, g_qwen.linear_key_head_dim,
                         &linear_key_dim) ||
        !checked_int_mul(g_qwen.linear_num_key_heads, g_qwen.linear_value_head_dim,
                         &linear_decode_key_dim) ||
        !checked_int_mul(g_qwen.linear_num_value_heads, g_qwen.linear_value_head_dim,
                         &linear_value_dim) ||
        !checked_int_mul(linear_key_dim, 2, &twice_linear_key_dim) ||
        !checked_int_mul(linear_decode_key_dim, 2, &twice_linear_decode_key_dim) ||
        twice_linear_key_dim > INT_MAX - linear_value_dim ||
        twice_linear_decode_key_dim > INT_MAX - linear_value_dim) {
        std::fprintf(stderr, "Scratch dimensions overflow int\n");
        qwn_free();
        return -2;
    }
    linear_conv_dim = twice_linear_key_dim + linear_value_dim;
    linear_decode_conv_dim = twice_linear_decode_key_dim + linear_value_dim;
    const size_t qkv_count = std::max(static_cast<size_t>(full_q_rows),
                                      static_cast<size_t>(linear_decode_conv_dim));
    int full_kv_rows = 0;
    if (!checked_int_mul(g_qwen.nkv, g_qwen.hd, &full_kv_rows)) {
        std::fprintf(stderr, "Scratch KV dimensions overflow int\n");
        qwn_free();
        return -2;
    }
    const size_t z_count = std::max(static_cast<size_t>(full_kv_rows),
                                    static_cast<size_t>(linear_value_dim));
    const size_t b_count = std::max(static_cast<size_t>(g_qwen.linear_num_value_heads),
                                    static_cast<size_t>(full_kv_rows));
    const size_t core_count = std::max(static_cast<size_t>(full_attn_rows),
                                       static_cast<size_t>(linear_value_dim));
    const size_t projection_count = std::max({
        static_cast<size_t>(g_qwen.H),
        static_cast<size_t>(g_qwen.I),
        static_cast<size_t>(full_q_rows),
        static_cast<size_t>(full_kv_rows),
        static_cast<size_t>(g_qwen.linear_num_value_heads),
        static_cast<size_t>(linear_conv_dim),
        static_cast<size_t>(linear_value_dim),
    });
    size_t score_count = 0;
    if (!checked_mul(static_cast<size_t>(g_qwen.nh),
                     static_cast<size_t>(g_qwen.max_seq), &score_count)) {
        std::fprintf(stderr, "Attention score scratch size overflows size_t\n");
        qwn_free();
        return -2;
    }
    const size_t argmax_count = static_cast<size_t>(TQ_ARGMAX_BLOCKS) + 1;

    if (allocate_elements(&g_qwen.d_x, static_cast<size_t>(g_qwen.H), "forward.x") != 0 ||
        allocate_elements(&g_qwen.d_norm_out, static_cast<size_t>(g_qwen.H),
                          "forward.norm_out") != 0 ||
        allocate_elements(&g_qwen.d_qkv, qkv_count, "forward.qkv") != 0 ||
        allocate_elements(&g_qwen.d_z, z_count, "forward.z") != 0 ||
        allocate_elements(&g_qwen.d_b, b_count, "forward.b") != 0 ||
        allocate_elements(&g_qwen.d_a,
                          static_cast<size_t>(g_qwen.linear_num_value_heads),
                          "forward.a") != 0 ||
        allocate_elements(&g_qwen.d_core, core_count, "forward.core") != 0 ||
        allocate_elements(&g_qwen.d_resid, static_cast<size_t>(g_qwen.H),
                          "forward.resid") != 0 ||
        allocate_elements(&g_qwen.d_post_norm, static_cast<size_t>(g_qwen.H),
                          "forward.post_norm") != 0 ||
        allocate_elements(&g_qwen.d_gate, static_cast<size_t>(g_qwen.I),
                          "forward.gate") != 0 ||
        allocate_elements(&g_qwen.d_up, static_cast<size_t>(g_qwen.I),
                          "forward.up") != 0 ||
        allocate_elements(&g_qwen.d_mlp_hidden, static_cast<size_t>(g_qwen.I),
                          "forward.mlp_hidden") != 0 ||
        allocate_elements(&g_qwen.d_layer_out, static_cast<size_t>(g_qwen.H),
                          "forward.layer_out") != 0 ||
        allocate_elements(&g_qwen.d_proj_out, projection_count,
                          "forward.proj_out") != 0 ||
        allocate_elements(&g_qwen.d_logits, static_cast<size_t>(g_qwen.V),
                          "forward.logits") != 0 ||
        allocate_elements(&g_qwen.d_scores, score_count, "forward.scores") != 0 ||
        allocate_elements(&g_qwen.d_argmax_vals, argmax_count,
                          "forward.argmax_vals") != 0 ||
        allocate_elements(&g_qwen.d_argmax_ids, argmax_count,
                          "forward.argmax_ids") != 0) {
        qwn_free();
        return -2;
    }

    // Phase 2: repack every projection into a native integer DPAS tier unless
    // TQ_XPU_W4A8=0 explicitly selects the scalar reference.
    //
    // Default is W4 everywhere except lm_head. Measured on a B70: lm_head is
    // the ONLY tensor that earns 8 bits - it emits the logits, so its argmax
    // margins are what teacher-forced agreement actually scores. Putting
    // q/k/v/o/out in W8 as well cost 9.6% throughput for no measurable quality
    // (94.16% vs 94.55%, a one-position difference over 257). TQ_XPU_W8
    // overrides with a comma-separated exact selector, "all", or "" for none.
    const char *integer_tier = std::getenv("TQ_XPU_W4A8");
    if (!integer_tier || integer_tier[0] != '0') {
        const char *w8_override = std::getenv("TQ_XPU_W8");
        const char *w8_selector = w8_override ? w8_override : "lm_head";
        // TQ_XPU_W4_K16 is a selector over the same labels ("all" for every
        // W4 matrix): CUDA's NVFP4 scale granularity on the Intel INT4 grid.
        const char *k16_selector = std::getenv("TQ_XPU_W4_K16");
        // TQ_XPU_K64 is a selector over the same labels ("all", or e.g.
        // "gate,up,down" for the MLP trio; "1" is accepted as "all"): one
        // FP16 scale per K64 tile - the dpas.s4.s4.8.8 granularity that
        // unlocks the W4A4 prefill GEMM, at +13.6% weight rel-L2 on the
        // matrices it covers (grid probe scheme 10, level-up item 4).
        // Measured: k64 on ALL weights prices decode TF at 89.9% (below the
        // 0.90 gate); the MLP-only selector is the shipping candidate.
        // w8 and k16 selectors win over k64 for any matrix naming both.
        const char *k64_env = std::getenv("TQ_XPU_K64");
        const char *k64_selector =
            (k64_env && k64_env[0] == '1' && k64_env[1] == '\0') ? "all"
                                                                 : k64_env;
        int n_w4 = 0, n_w4k16 = 0, n_w4k64 = 0, n_w8 = 0, n_fail = 0;
        auto rep = [&](tq_qmma_weight_t *w, const char *kind) {
            if (!w->d_A) return;
            const bool use_w8 = selector_has(w8_selector, kind);
            const bool use_k16 = !use_w8 && selector_has(k16_selector, kind);
            const bool use_k64 =
                !use_w8 && !use_k16 && selector_has(k64_selector, kind);
            const int mode = use_k16 ? 1 : (use_k64 ? 2 : 0);
            const int rc = use_w8 ? x_w8_repack_weight(w)
                                  : x_w4_repack_weight_mode(w, mode);
            if (rc == 0) {
                if (use_w8) ++n_w8;
                else if (use_k16) ++n_w4k16;
                else if (use_k64) ++n_w4k64;
                else ++n_w4;
            } else {
                ++n_fail;
                std::fprintf(stderr, "%s repack failed for %s rc=%d\n",
                             use_w8 ? "W8" : "W4", kind, rc);
            }
        };
        rep(&g_qwen.lm_head, "lm_head");
        for (int i = 0; i < g_qwen.L; ++i) {
            tq_layer_t *l = &g_qwen.layers[i];
            rep(&l->mlp_gate, "gate"); rep(&l->mlp_up, "up"); rep(&l->mlp_down, "down");
            if (g_qwen.layer_types[i] == TQ_LAYER_FULL_ATTENTION) {
                rep(&l->q_proj, "q"); rep(&l->k_proj, "k");
                rep(&l->v_proj, "v"); rep(&l->o_proj, "o");
            } else {
                rep(&l->linear_in_a, "in_a"); rep(&l->linear_in_b, "in_b");
                rep(&l->linear_in_qkv, "in_qkv"); rep(&l->linear_in_z, "in_z");
                rep(&l->linear_out, "out");
            }
            if ((i % 8) == 0) std::printf("  integer repack layer %d/%d\n", i, g_qwen.L);
        }
        std::printf("integer DPAS repack: W4=%d W4k16=%d W4k64=%d W8=%d "
                    "failed=%d w8=%s k16=%s k64=%s\n",
                    n_w4, n_w4k16, n_w4k64, n_w8, n_fail,
                    w8_selector ? w8_selector : "<none>",
                    k16_selector ? k16_selector : "<none>",
                    k64_selector ? k64_selector : "<none>");
        if (n_fail) { qwn_free(); return -3; }
    }

    // Slot count is fixed at init (state allocations depend on it). Paged
    // mode honors the requested supported value exactly; flat mode preserves
    // its historical atoi-and-clamp behavior.
    const char *slots_env = std::getenv("TQ_XPU_SLOTS");
    int slots = 1;
    if (paged_mode) {
        if (parse_paged_int_env("TQ_XPU_SLOTS", 1, 1, 8, &slots) != 0) {
            qwn_free();
            return -4;
        }
    } else {
        slots = slots_env ? std::atoi(slots_env) : 1;
        if (slots < 1) slots = 1;
        if (slots > 8) slots = 8;
    }
    g_qwen.slots = slots;
    g_qwen.active_slot = 0;
    for (int s = 0; s < 8; ++s) g_qwen.state_pos[s] = 0;

    const int paged_init = tq_paged_init_from_env();
    if (paged_init != 0) {
        std::fprintf(stderr, "paged KV initialization failed: %d\n", paged_init);
        qwn_free();
        return -4;
    }
    if (qwn_paged_enabled()) {
        if (!tq_paged_ready()) {
            std::fprintf(stderr, "paged KV initialization did not become ready\n");
            qwn_free();
            return -4;
        }
        const int reset = qwn_reset_state();
        if (reset != 0) {
            std::fprintf(stderr, "paged state initialization failed: %d\n", reset);
            qwn_free();
            return -4;
        }
        int warm_tokens = 0;
        if (paged_prefill_warm_tokens(g_qwen.max_seq, &warm_tokens) != 0 ||
            tq_decode_runtime_warm(warm_tokens) != 0) {
            std::fprintf(stderr, "paged prefill scratch warm failed\n");
            qwn_free();
            return -4;
        }
    }
    return 0;
}

extern "C" int qwn_reset_state(void) {
    if (!g_qwen.initialized || g_qwen.max_seq <= 0) return -1;
    const bool paged = qwn_paged_enabled() != 0;
    if (paged && !tq_paged_ready()) return -1;

    tq_state_sizes_t sizes{};
    if (!get_state_sizes(&sizes)) {
        std::fprintf(stderr, "State dimensions overflow\n");
        return -2;
    }

    const size_t slots =
        static_cast<size_t>(g_qwen.slots < 1 ? 1 : g_qwen.slots);
    size_t conv_total = 0;
    size_t recurrent_total = 0;
    size_t kv_total = 0;
    size_t kv_rows_total = 0;
    if (!checked_mul(sizes.conv, slots, &conv_total) ||
        !checked_mul(sizes.recurrent, slots, &recurrent_total) ||
        (!paged &&
         (!checked_mul(sizes.kv, slots, &kv_total) ||
          !checked_mul(sizes.kv_rows, slots, &kv_rows_total)))) {
        std::fprintf(stderr, "State allocation size overflows size_t\n");
        return -2;
    }

    for (int i = 0; i < g_qwen.L; ++i) {
        tq_layer_t *layer = &g_qwen.layers[i];
        if (g_qwen.layer_types[i] == TQ_LAYER_LINEAR_ATTENTION) {
            if ((!layer->d_linear_conv_state &&
                 allocate_elements(&layer->d_linear_conv_state, conv_total,
                                   "linear.conv_state") != 0) ||
                (!layer->d_linear_recurrent_state &&
                 allocate_elements(&layer->d_linear_recurrent_state,
                                   recurrent_total,
                                   "linear.recurrent_state") != 0)) {
                return -2;
            }
        } else if (g_qwen.layer_types[i] == TQ_LAYER_FULL_ATTENTION && !paged) {
            // E4M3 K/V use one byte per element plus one FP16 scale per row.
            if ((!layer->d_k_cache &&
                 allocate_elements(&layer->d_k_cache, kv_total,
                                   "attention.k_cache") != 0) ||
                (!layer->d_v_cache &&
                 allocate_elements(&layer->d_v_cache, kv_total,
                                   "attention.v_cache") != 0) ||
                (!layer->d_k_scale &&
                 allocate_elements(&layer->d_k_scale, kv_rows_total,
                                   "attention.k_scale") != 0) ||
                (!layer->d_v_scale &&
                 allocate_elements(&layer->d_v_scale, kv_rows_total,
                                   "attention.v_scale") != 0)) {
                return -3;
            }
        }
    }

    if (paged) {
        const int reset = tq_paged_reset_all();
        if (reset != 0) return reset;
    }

    try {
        for (int i = 0; i < g_qwen.L; ++i) {
            tq_layer_t *layer = &g_qwen.layers[i];
            if (g_qwen.layer_types[i] == TQ_LAYER_LINEAR_ATTENTION) {
                tq_dev_zero(layer->d_linear_conv_state,
                            conv_total * sizeof(float));
                tq_dev_zero(layer->d_linear_recurrent_state,
                            recurrent_total * sizeof(float));
            } else if (g_qwen.layer_types[i] == TQ_LAYER_FULL_ATTENTION &&
                       !paged) {
                tq_dev_zero(layer->d_k_cache, kv_total);
                tq_dev_zero(layer->d_v_cache, kv_total);
                tq_dev_zero(layer->d_k_scale,
                            kv_rows_total * sizeof(uint16_t));
                tq_dev_zero(layer->d_v_scale,
                            kv_rows_total * sizeof(uint16_t));
            }
        }
        tq_q().wait_and_throw();
    } catch (const std::exception &e) {
        std::fprintf(stderr, "device state reset failed: %s\n", e.what());
        return -5;
    } catch (...) {
        std::fprintf(stderr, "device state reset failed\n");
        return -5;
    }

    for (int s = 0; s < 8; ++s) g_qwen.state_pos[s] = 0;
    g_qwen.active_slot = 0;
    return 0;
}

// ---- tensor-parallel model sharding -------------------------------------
// Turns a fully-loaded rank-0 model into one shard per rank. Called after
// qwn_init, before any forward. Two passes so rank 0's FULL weights are the
// source for everyone:
//   pass 1: ranks 1..tp-1 copy the struct, allocate their own scratch, and
//           shard/duplicate/slice every tensor out of rank 0's copy.
//   pass 2: rank 0 sheds its own weights in place (shard -> free full), which
//           is what actually halves each card's footprint.
// Head counts are divided LAST so the scratch sizing above still sees the
// full dimensions - the buffers stay full-sized on purpose (a few MB) rather
// than risk undersizing a shard.
//
// Modes per family are the ones qwn_shard_check_all validated:
//   column: q/k/v, lin_qkv (grouped [key|key|val]), lin_z, mlp_gate, mlp_up
//   row:    o_proj, lin_out, mlp_down
//   replicate: lin_a, lin_b (M=48; 24 rows/rank is not 16-aligned)
// d_embed and lm_head stay on rank 0 only: the embed feeds a 20 KB broadcast
// of d_x, and a vocab-parallel lm_head is a later item.
namespace {

template <typename T>
T *tp_dup(const T *src, size_t count, const char *what) {
    if (!src || count == 0) return nullptr;
    T *dst = static_cast<T *>(tq_dev_alloc(count * sizeof(T), what));
    tq_q().memcpy(dst, src, count * sizeof(T));
    return dst;
}

template <typename T>
T *tp_slice(const T *src, size_t off, size_t count, const char *what) {
    if (!src || count == 0) return nullptr;
    T *dst = static_cast<T *>(tq_dev_alloc(count * sizeof(T), what));
    tq_q().memcpy(dst, src + off, count * sizeof(T));
    return dst;
}

// conv1d is [conv_dim, ks] channel-major and its channels follow the same
// [key|key|val] grouping as linear_in_qkv, so it slices per group.
uint16_t *tp_slice_conv(const uint16_t *src, const int *groups, int ng,
                        int rank, int tp, int ks) {
    if (!src) return nullptr;
    size_t total = 0;
    for (int g = 0; g < ng; ++g) total += (size_t)groups[g] / tp;
    uint16_t *dst = static_cast<uint16_t *>(
        tq_dev_alloc(total * (size_t)ks * sizeof(uint16_t), "tp.conv1d"));
    size_t src_c = 0, dst_c = 0;
    for (int g = 0; g < ng; ++g) {
        const size_t per = (size_t)groups[g] / tp;
        tq_q().memcpy(dst + dst_c * ks,
                      src + (src_c + per * (size_t)rank) * ks,
                      per * (size_t)ks * sizeof(uint16_t));
        src_c += (size_t)groups[g];
        dst_c += per;
    }
    return dst;
}

int tp_shard_layer(tq_layer_t *dst, const tq_layer_t *src, int type, int rank,
                   int tp, int H, int hd, int vhd, int nv, int key_dim,
                   int val_dim, int ks) {
    const int groups_qkv[3] = {key_dim, key_dim, val_dim};
    struct Job { tq_qmma_weight_t *d; const tq_qmma_weight_t *s; int mode;
                 const int *g; int ng; };
    Job jobs[12];
    int n = 0;
    if (type == TQ_LAYER_FULL_ATTENTION) {
        jobs[n++] = {&dst->q_proj, &src->q_proj, 0, nullptr, 0};
        jobs[n++] = {&dst->k_proj, &src->k_proj, 0, nullptr, 0};
        jobs[n++] = {&dst->v_proj, &src->v_proj, 0, nullptr, 0};
        jobs[n++] = {&dst->o_proj, &src->o_proj, 1, nullptr, 0};
    } else {
        jobs[n++] = {&dst->linear_in_qkv, &src->linear_in_qkv, 0, groups_qkv, 3};
        jobs[n++] = {&dst->linear_in_z, &src->linear_in_z, 0, nullptr, 0};
        jobs[n++] = {&dst->linear_in_b, &src->linear_in_b, 2, nullptr, 0};
        jobs[n++] = {&dst->linear_in_a, &src->linear_in_a, 2, nullptr, 0};
        jobs[n++] = {&dst->linear_out, &src->linear_out, 1, nullptr, 0};
    }
    jobs[n++] = {&dst->mlp_gate, &src->mlp_gate, 0, nullptr, 0};
    jobs[n++] = {&dst->mlp_up, &src->mlp_up, 0, nullptr, 0};
    jobs[n++] = {&dst->mlp_down, &src->mlp_down, 1, nullptr, 0};
    for (int i = 0; i < n; ++i) {
        tq_qmma_weight_t out{};
        const int rc = x_w4_shard(&out, jobs[i].s, rank, tp, jobs[i].mode,
                                  jobs[i].g, jobs[i].ng);
        if (rc != 0) {
            std::fprintf(stderr, "tp shard failed: job %d mode %d rc %d\n",
                         i, jobs[i].mode, rc);
            return -1;
        }
        *jobs[i].d = out;
    }
    dst->d_input_ln = tp_dup(src->d_input_ln, (size_t)H, "tp.input_ln");
    dst->d_post_ln = tp_dup(src->d_post_ln, (size_t)H, "tp.post_ln");
    if (type == TQ_LAYER_FULL_ATTENTION) {
        dst->d_q_norm = tp_dup(src->d_q_norm, (size_t)hd, "tp.q_norm");
        dst->d_k_norm = tp_dup(src->d_k_norm, (size_t)hd, "tp.k_norm");
    } else {
        dst->d_linear_norm = tp_dup(src->d_linear_norm, (size_t)vhd, "tp.lnorm");
        const size_t per = (size_t)nv / tp;
        dst->d_linear_A_log =
            tp_slice(src->d_linear_A_log, per * rank, per, "tp.A_log");
        dst->d_linear_dt_bias =
            tp_slice(src->d_linear_dt_bias, per * rank, per, "tp.dt_bias");
        dst->d_linear_conv1d =
            tp_slice_conv(src->d_linear_conv1d, groups_qkv, 3, rank, tp, ks);
    }
    // per-sequence state is (re)allocated by qwn_reset_state from this rank's
    // own head counts, so drop the inherited pointers rather than free them
    // (they belong to rank 0).
    dst->d_linear_conv_state = nullptr;
    dst->d_linear_recurrent_state = nullptr;
    dst->d_k_cache = dst->d_v_cache = nullptr;
    dst->d_k_scale = dst->d_v_scale = nullptr;
    tq_q().wait_and_throw();
    return 0;
}

}  // namespace

extern "C" int qwn_tp_shard(void) {
    const int tp = tq_tp_size();
    if (tp < 2) return 0;
    tq_model_t *m0 = &g_qwen_ranks[0];
    if (!m0->initialized) return -1;
    const int nh0 = m0->nh, nkv0 = m0->nkv, I0 = m0->I;
    const int nv0 = m0->linear_num_value_heads;
    const int nk0 = m0->linear_num_key_heads;
    if (nh0 % tp || nkv0 % tp || I0 % tp || nv0 % tp || nk0 % tp) {
        std::fprintf(stderr, "tp shard: head/intermediate counts not "
                             "divisible by tp=%d\n", tp);
        return -2;
    }
    const int H = m0->H, hd = m0->hd, ks = m0->linear_conv_kernel_dim;
    const int vhd = m0->linear_value_head_dim;
    const int key_dim = nk0 * m0->linear_key_head_dim;
    const int val_dim = nv0 * vhd;
    const int L = m0->L;

    // ---- pass 1: the other ranks, sourced from rank 0's full weights -----
    for (int r = 1; r < tp; ++r) {
        tq_set_rank(r);
        tq_model_t *mr = &g_qwen_ranks[r];
        *mr = *m0;                          // scalars + rank-0 pointers
        mr->d_embed = nullptr;              // rank 0 embeds and broadcasts d_x
        mr->lm_head = tq_qmma_weight_t{};   // rank 0 owns the head for now
        mr->d_norm = tp_dup(m0->d_norm, (size_t)H, "tp.final_norm");
        for (int i = 0; i < L; ++i) {
            if (tp_shard_layer(&mr->layers[i], &m0->layers[i],
                               m0->layer_types[i], r, tp, H, hd, vhd, nv0,
                               key_dim, val_dim, ks) != 0)
                return -3;
        }
        // Fresh scratch on this rank, still at FULL dimensions: a few MB, and
        // it cannot be undersized for a shard.
        mr->d_x = mr->d_norm_out = mr->d_qkv = mr->d_z = mr->d_b = mr->d_a =
            mr->d_core = mr->d_resid = mr->d_post_norm = mr->d_gate =
            mr->d_up = mr->d_mlp_hidden = mr->d_layer_out = mr->d_proj_out =
            mr->d_logits = mr->d_scores = mr->d_argmax_vals = nullptr;
        mr->d_argmax_ids = nullptr;
        // Scratch sized from rank 0's FULL dims: a few MB, and it cannot be
        // undersized for a shard. (qwn_init's own sizing lives inline in a
        // long overflow-checked block; duplicating the ceilings here is
        // cheaper and lower-risk than refactoring it.)
        {
            const int conv_dim_f = 2 * key_dim + val_dim;
            const size_t big = (size_t)std::max(
                std::max(H, I0), std::max(2 * nh0 * hd, conv_dim_f));
            auto fa = [&](size_t cnt, const char *what) {
                return static_cast<float *>(
                    tq_dev_alloc(cnt * sizeof(float), what));
            };
            mr->d_x = fa((size_t)H, "tp.x");
            mr->d_norm_out = fa((size_t)H, "tp.norm_out");
            mr->d_resid = fa((size_t)H, "tp.resid");
            mr->d_post_norm = fa((size_t)H, "tp.post_norm");
            mr->d_layer_out = fa((size_t)H, "tp.layer_out");
            mr->d_qkv = fa((size_t)std::max(2 * nh0 * hd, conv_dim_f), "tp.qkv");
            mr->d_z = fa((size_t)std::max(nkv0 * hd, val_dim), "tp.z");
            mr->d_b = fa((size_t)std::max(nv0, nkv0 * hd), "tp.b");
            mr->d_a = fa((size_t)nv0, "tp.a");
            mr->d_core = fa((size_t)std::max(nh0 * hd, val_dim), "tp.core");
            mr->d_gate = fa((size_t)I0, "tp.gate");
            mr->d_up = fa((size_t)I0, "tp.up");
            mr->d_mlp_hidden = fa((size_t)I0, "tp.mlp_hidden");
            mr->d_proj_out = fa(big, "tp.proj_out");
            mr->d_logits = fa((size_t)m0->V, "tp.logits");
            mr->d_scores = fa((size_t)nh0 * (size_t)m0->max_seq, "tp.scores");
            mr->d_argmax_vals = fa((size_t)TQ_ARGMAX_BLOCKS + 1, "tp.amax_v");
            mr->d_argmax_ids = static_cast<int *>(tq_dev_alloc(
                ((size_t)TQ_ARGMAX_BLOCKS + 1) * sizeof(int), "tp.amax_i"));
            tq_q().wait_and_throw();
        }
        mr->nh = nh0 / tp;
        mr->nkv = nkv0 / tp;
        mr->I = I0 / tp;
        mr->linear_num_value_heads = nv0 / tp;
        mr->linear_num_key_heads = nk0 / tp;
        mr->initialized = 1;
    }

    // ---- pass 2: rank 0 sheds its own weights in place -------------------
    tq_set_rank(0);
    for (int i = 0; i < L; ++i) {
        tq_layer_t shard = m0->layers[i];
        if (tp_shard_layer(&shard, &m0->layers[i], m0->layer_types[i], 0, tp,
                           H, hd, vhd, nv0, key_dim, val_dim, ks) != 0)
            return -5;
        // free the full copies now that every rank has its slice
        tq_layer_t *full = &m0->layers[i];
        tq_qmma_weight_t *fw[8];
        int nf = 0;
        if (m0->layer_types[i] == TQ_LAYER_FULL_ATTENTION) {
            fw[nf++] = &full->q_proj; fw[nf++] = &full->k_proj;
            fw[nf++] = &full->v_proj; fw[nf++] = &full->o_proj;
        } else {
            fw[nf++] = &full->linear_in_qkv; fw[nf++] = &full->linear_in_z;
            fw[nf++] = &full->linear_in_b;   fw[nf++] = &full->linear_in_a;
            fw[nf++] = &full->linear_out;
        }
        fw[nf++] = &full->mlp_gate; fw[nf++] = &full->mlp_up;
        fw[nf++] = &full->mlp_down;
        for (int j = 0; j < nf; ++j) {
            tq_dev_free(fw[j]->d_s4);
            tq_dev_free(fw[j]->d_s4_scale);
        }
        tq_dev_free(full->d_linear_A_log);
        tq_dev_free(full->d_linear_dt_bias);
        tq_dev_free(full->d_linear_conv1d);
        // keep rank 0's replicated norms; the shard reuses fresh copies
        tq_dev_free(full->d_input_ln);
        tq_dev_free(full->d_post_ln);
        tq_dev_free(full->d_q_norm);
        tq_dev_free(full->d_k_norm);
        tq_dev_free(full->d_linear_norm);
        m0->layers[i] = shard;
    }
    m0->nh = nh0 / tp;
    m0->nkv = nkv0 / tp;
    m0->I = I0 / tp;
    m0->linear_num_value_heads = nv0 / tp;
    m0->linear_num_key_heads = nk0 / tp;
    std::printf("tp shard: %d ranks, per-rank nh=%d nkv=%d I=%d nv=%d nk=%d\n",
                tp, m0->nh, m0->nkv, m0->I, m0->linear_num_value_heads,
                m0->linear_num_key_heads);
    return tp;
}

// Tensor-parallel substrate self-test. Independent of the model, so it can
// run before any weights exist: allocate a vector per rank, seed rank r with
// (r+1)*i, all-reduce, and check every rank holds sum_r (r+1)*i. Returns the
// rank count on success, negative on a mismatch. This gates the substrate
// (shared context, per-rank queues, peer exchange) before the sharded
// forward is built on top of it.
extern "C" int qwn_tp_selftest(int n) {
    const int tp = tq_tp_size();
    if (tp < 2) return tp;                    // nothing to reduce
    if (n <= 0 || n > (1 << 22)) return -1;
    std::vector<float *> bufs(tp), scratch(tp);
    std::vector<float> host(n);
    const int saved = tq_rank();
    for (int r = 0; r < tp; ++r) {
        tq_set_rank(r);
        bufs[r] = static_cast<float *>(tq_dev_alloc((size_t)n * 4, "tp.buf"));
        scratch[r] = static_cast<float *>(tq_dev_alloc((size_t)n * 4, "tp.scr"));
        for (int i = 0; i < n; ++i) host[i] = (float)(r + 1) * (float)i;
        tq_h2d(bufs[r], host.data(), (size_t)n * 4);
    }
    tq_allreduce(bufs.data(), scratch.data(), n);
    float want_mul = 0.0f;
    for (int r = 0; r < tp; ++r) want_mul += (float)(r + 1);
    int bad = 0;
    for (int r = 0; r < tp && !bad; ++r) {
        tq_set_rank(r);
        tq_d2h(host.data(), bufs[r], (size_t)n * 4);
        for (int i = 0; i < n; ++i)
            if (host[i] != want_mul * (float)i) { bad = 1; break; }
    }
    for (int r = 0; r < tp; ++r) {
        tq_set_rank(r);
        tq_dev_free(bufs[r]);
        tq_dev_free(scratch[r]);
    }
    tq_set_rank(saved);
    return bad ? -2 : tp;
}

extern "C" int qwn_num_slots(void) { return g_qwen.slots < 1 ? 1 : g_qwen.slots; }
extern "C" int qwn_get_slot(void) { return g_qwen.active_slot; }
extern "C" int qwn_set_slot(int slot) {
    if (!g_qwen.initialized) return -1;
    if (slot < 0 || slot >= (g_qwen.slots < 1 ? 1 : g_qwen.slots)) return -2;
    g_qwen.active_slot = slot;
    return 0;
}

// Zero ONE slot's per-sequence state (KV + scales + GDN conv/recurrent) and
// its committed position, leaving every other slot untouched. The full
// qwn_reset_state remains the whole-engine reset (all slots).
extern "C" int qwn_reset_slot(int slot) {
    if (!g_qwen.initialized || g_qwen.max_seq <= 0) return -1;
    const int slot_count = g_qwen.slots < 1 ? 1 : g_qwen.slots;
    if (slot < 0 || slot >= slot_count) return -2;

    const bool paged = qwn_paged_enabled() != 0;
    if (paged && !tq_paged_ready()) return -1;
    if (!state_buffers_ready(paged)) {
        // Flat mode historically allocates state on its first reset. Paged
        // startup eagerly prepares GDN state, so missing state is an error
        // rather than a null-flat-pointer signal to reset every slot.
        if (paged) return -4;
        return qwn_reset_state();
    }

    tq_state_sizes_t sizes{};
    if (!get_state_sizes(&sizes)) return -3;
    const size_t slots = static_cast<size_t>(slot_count);
    size_t ignored = 0;
    if (!checked_mul(sizes.conv, slots, &ignored) ||
        !checked_mul(sizes.recurrent, slots, &ignored) ||
        (!paged &&
         (!checked_mul(sizes.kv, slots, &ignored) ||
          !checked_mul(sizes.kv_rows, slots, &ignored)))) {
        return -3;
    }

    if (paged) {
        // This drops only this slot's page references. Unlike retirement,
        // reset-at-zero preserves the scheduler's reservation credits.
        const int reset = tq_paged_reset_slot(slot);
        if (reset != 0) return reset;
    }

    try {
        for (int i = 0; i < g_qwen.L; ++i) {
            tq_layer_t *layer = &g_qwen.layers[i];
            if (g_qwen.layer_types[i] == TQ_LAYER_LINEAR_ATTENTION) {
                tq_dev_zero(
                    layer->d_linear_conv_state +
                        static_cast<size_t>(slot) * sizes.conv,
                    sizes.conv * sizeof(float));
                tq_dev_zero(
                    layer->d_linear_recurrent_state +
                        static_cast<size_t>(slot) * sizes.recurrent,
                    sizes.recurrent * sizeof(float));
            } else if (g_qwen.layer_types[i] == TQ_LAYER_FULL_ATTENTION &&
                       !paged) {
                tq_dev_zero(
                    layer->d_k_cache + static_cast<size_t>(slot) * sizes.kv,
                    sizes.kv);
                tq_dev_zero(
                    layer->d_v_cache + static_cast<size_t>(slot) * sizes.kv,
                    sizes.kv);
                tq_dev_zero(
                    layer->d_k_scale +
                        static_cast<size_t>(slot) * sizes.kv_rows,
                    sizes.kv_rows * sizeof(uint16_t));
                tq_dev_zero(
                    layer->d_v_scale +
                        static_cast<size_t>(slot) * sizes.kv_rows,
                    sizes.kv_rows * sizeof(uint16_t));
            }
        }
        tq_q().wait_and_throw();
    } catch (const std::exception &e) {
        std::fprintf(stderr, "device slot reset failed: %s\n", e.what());
        return -5;
    } catch (...) {
        std::fprintf(stderr, "device slot reset failed\n");
        return -5;
    }
    g_qwen.state_pos[slot] = 0;
    return 0;
}

extern "C" int qwn_hidden_size(void) { return g_qwen.H; }
extern "C" int qwn_intermediate_size(void) { return g_qwen.I; }
extern "C" int qwn_vocab_size(void) { return g_qwen.V; }
extern "C" int qwn_num_layers(void) { return g_qwen.L; }
extern "C" int qwn_num_attention_heads(void) { return g_qwen.nh; }
extern "C" int qwn_num_key_value_heads(void) { return g_qwen.nkv; }
extern "C" int qwn_head_dim(void) { return g_qwen.hd; }
extern "C" int qwn_max_seq(void) { return g_qwen.max_seq; }
extern "C" int qwn_layer_type(int layer) {
    if (layer < 0 || layer >= g_qwen.L) return 0;
    return static_cast<int>(g_qwen.layer_types[layer]);
}
extern "C" int qwn_has_mtp(void) { return 0; }
extern "C" int qwn_last_argmax_id(void) { return g_qwen.last_argmax_id; }
extern "C" float qwn_last_argmax_logit(void) { return g_qwen.last_argmax_logit; }
