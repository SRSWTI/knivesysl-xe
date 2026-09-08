#pragma once

#include "tq_common.hpp"

// Returns false only for auto/disabled selection. TQ_XPU_PREFILL_XMX=1
// requires the supported Xe2/HD256 path and throws rather than falling back.
bool x_prefill_attn_xmx(float *d_out, const float *d_qg_proj,
                        const uint16_t *d_q_norm, const uint8_t *d_k_cache,
                        const uint8_t *d_v_cache, const uint16_t *d_k_scale,
                        const uint16_t *d_v_scale, int pos0, int T, int nh,
                        int nkv, int hd, float eps, float rope_theta,
                        float partial_rotary_factor, tq_kv_layout_t layout);

bool x_prefill_attn_xmx_packed(const tq_prefill_attn_request_t *requests,
                               int n, const uint16_t *qnorm,
                               int nh, int nkv, int hd, float eps,
                               float rope_theta, float partial_rotary_factor);
