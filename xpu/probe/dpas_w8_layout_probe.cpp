// dpas_w8_layout_probe.cpp — one-hot layout proof for Xe2 W8A8 GEMV.
//
// Validates dpas.s8.s8.8.1 as D[1][16] = A[1][32] x B[32][16]:
//   A: flat byte k is logical k; SIMD16 lane l supplies bytes 2l, 2l+1.
//   B: (k,n) is byte (k%4) of flat dword (k/4)*16+n.
//   D: lane n owns output column n.
//
// Build: icpx -fsycl -O2 -o dpas_w8_layout_probe dpas_w8_layout_probe.cpp
#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdio>

using uintv8 = uint32_t __attribute__((ext_vector_type(8)));

struct probe_s8s8_rc1 {
    const uint8_t *a_bytes;    // 32-byte flat A image
    const uint32_t *b_words;   // 128-dword flat B image
    int32_t *d_out;            // 16-dword flat D image

    [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> it) const {
        const auto sg = it.get_sub_group();
        const int lane = (int)sg.get_local_linear_id();
        if (it.get_group_linear_id() != 0 || sg.get_group_linear_id() != 0) return;

        const uint16_t a = *reinterpret_cast<const uint16_t *>(a_bytes + 2 * lane);
        uintv8 b;
        for (int i = 0; i < 8; ++i) b[i] = b_words[i * 16 + lane];
        int32_t d = 0;
#ifdef __SYCL_DEVICE_ONLY__
        asm("{\n"
            ".decl DST     v_type=G type=D num_elts=16 alias=<%0,0>\n"
            ".decl SRC1_UD v_type=G type=UD num_elts=128 alias=<%2,0>\n"
            ".decl SRC2_UD v_type=G type=UD num_elts=8 alias=<%1,0>\n"
            "dpas.s8.s8.8.1 (M1, 16) DST.0 DST.0 SRC1_UD.0 SRC2_UD(0,0)\n"
            "}\n"
            : "+rw"(d) : "rw"(a), "rw"(b));
#endif
        d_out[lane] = d;
    }
};

int main() {
    sycl::queue q({sycl::device(sycl::gpu_selector_v)},
                  sycl::property::queue::in_order{});
    std::printf("device: %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());

    auto *a = sycl::malloc_shared<uint8_t>(32, q);
    auto *b = sycl::malloc_shared<uint32_t>(128, q);
    auto *d = sycl::malloc_shared<int32_t>(16, q);

    auto clear = [&] {
        for (int i = 0; i < 32; ++i) a[i] = 0;
        for (int i = 0; i < 128; ++i) b[i] = 0;
        for (int i = 0; i < 16; ++i) d[i] = 0;
    };
    auto set_b = [&](int k, int n, int8_t value) {
        const int dw = (k >> 2) * 16 + n;
        reinterpret_cast<int8_t *>(b)[dw * 4 + (k & 3)] = value;
    };
    auto run = [&] {
        q.parallel_for(sycl::nd_range<1>(16, 16), probe_s8s8_rc1{a, b, d}).wait();
    };

    int alignment_errors = 0;
    for (int ak = 0; ak < 32; ++ak) {
        int paired = -1;
        for (int bk = 0; bk < 32; ++bk) {
            clear();
            a[ak] = 3;
            set_b(bk, 0, 2);
            run();
            if (d[0] == 6) {
                paired = bk;
                break;
            }
        }
        if (paired != ak) {
            if (alignment_errors < 5)
                std::printf("k alignment: A k=%d paired with B k=%d\n", ak, paired);
            ++alignment_errors;
        }
    }
    std::printf("A/B K alignment and B VNNI byte map: %s\n",
                alignment_errors ? "FAIL" : "PASS");

    clear();
    a[7] = 4;
    set_b(7, 11, 3);
    run();
    int route_errors = 0;
    for (int n = 0; n < 16; ++n) {
        const int expected = n == 11 ? 12 : 0;
        if (d[n] != expected) ++route_errors;
    }
    std::printf("D lane routing: %s\n", route_errors ? "FAIL" : "PASS");

    clear();
    a[0] = 2;
    set_b(0, 0, -3);
    run();
    const bool sign_ok = d[0] == -6;
    std::printf("S8 sign check (want -6): got %d  %s\n", d[0],
                sign_ok ? "PASS" : "FAIL");

    sycl::free(a, q);
    sycl::free(b, q);
    sycl::free(d, q);
    return alignment_errors || route_errors || !sign_ok;
}
