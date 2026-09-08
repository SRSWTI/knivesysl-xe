// dpas_layout_probe.cpp — one-hot operand-layout decode for the W4A8 GEMV
// instruction dpas.s4.s8.8.<rc> on Xe2 (B70).
//
// Question being answered (the Phase-0 rate probe used uniform patterns and
// cannot see layout): for D[M][N=16] = A[M][K=32](s8) x B[K=32][N=16](s4),
//   - where does A's k index live in the 8*M-dword A region?
//   - where does B's (k, n) element live in the 128-dword B region?
// Method: zero both operands, set exactly one A element and one B element,
// check which D element lights up and with what value — sweeping k, n, and rc.
//
// Build: icpx -fsycl -O2 -o dpas_layout_probe dpas_layout_probe.cpp
//
// Hypothesis under test (from cute XE_DPAS_TT operand shapes):
//   A (rc=1): 32 s8 bytes, lane-major: lane l bytes [2l, 2l+1] hold k = 2l, 2l+1.
//   A (rc=8): row m at byte offset 32*m... actually lane l dwords, k = within-lane
//             byte index; verified by sweep.
//   B: lane l = output column n; lane's 32B region holds K s4 codes; position of
//      code k within the region = k nibbles from the start (k/2 byte, k%2 nibble).
#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstdio>
#include <vector>

using uintv4 = uint32_t __attribute__((ext_vector_type(4)));
using uintv8 = uint32_t __attribute__((ext_vector_type(8)));
using intv8  = int32_t  __attribute__((ext_vector_type(8)));

// rc=8 s8 x s4: A = 8x32 s8 (256B = 64 dw), B = 32x16 s4 (fixed 512B region).
// The host buffers are FLAT GRF images: a per-lane ext_vector element i lives
// at flat GRF dword i*16 + lane, so loading a[i] = flat[i*16+lane] reproduces
// the flat image bit-for-bit in the operand region. One-hot positions in the
// host buffer therefore decode the raw hardware layout.
struct probe_rc8 {
    const uint32_t *a_words;   // 64 dwords: flat A GRF image
    const uint32_t *b_words;   // 128 dwords: flat B GRF image
    int *d_out;                // 128 ints: D flat image (dword m*16+lane)
    [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> it) const {
        auto sg = it.get_sub_group();
        const int lane = sg.get_local_linear_id();
        if (it.get_group_linear_id() != 0 || sg.get_group_linear_id() != 0) return;
        uintv4 a;
        uintv8 b;
        for (int i = 0; i < 4; i++) a[i] = a_words[i * 16 + lane];
        for (int i = 0; i < 8; i++) b[i] = b_words[i * 16 + lane];
        intv8 d;
        for (int i = 0; i < 8; i++) d[i] = 0;
#ifdef __SYCL_DEVICE_ONLY__
        asm("{\n"
            ".decl DST     v_type=G type=D num_elts=128 alias=<%0,0>\n"
            ".decl SRC1_UD v_type=G type=UD num_elts=128 alias=<%2,0>\n"
            ".decl SRC2_UD v_type=G type=UD num_elts=64 alias=<%1,0>\n"
            "dpas.s4.s8.8.8 (M1, 16) DST.0 DST.0 SRC1_UD.0 SRC2_UD(0,0)\n"
            "}\n"
            : "+rw"(d) : "rw"(a), "rw"(b));
#endif
        for (int i = 0; i < 8; i++) d_out[i * 16 + lane] = d[i];
    }
};

int main() {
    sycl::queue q({sycl::device(sycl::gpu_selector_v)},
                  sycl::property::queue::in_order{});
    printf("device: %s\n",
           q.get_device().get_info<sycl::info::device::name>().c_str());

    uint32_t *a = sycl::malloc_shared<uint32_t>(64, q);
    uint32_t *b = sycl::malloc_shared<uint32_t>(128, q);
    int *d = sycl::malloc_shared<int>(128, q);

    auto run = [&]() {
        q.parallel_for(sycl::nd_range<1>(16, 16), probe_rc8{a, b, d}).wait();
    };
    auto clr = [&]() {
        for (int i = 0; i < 64; i++) a[i] = 0;
        for (int i = 0; i < 128; i++) b[i] = 0;
        for (int i = 0; i < 128; i++) d[i] = 0;
    };

    // Hypotheses under test (canonical Intel DPAS operand layouts):
    //   A flat: row m = flat bytes [m*32, m*32+32), byte-in-row = k.
    //   B flat: (k, n) at flat dword (k/8)*16 + n, nibble k%8 (VNNI, opc=8).
    //   D flat: (m, n) at flat dword m*16 + n.
    auto b_set_nibble = [&](int k, int n, uint8_t val) {
        const int dw = (k >> 3) * 16 + n;
        const int nib = k & 7;
        uint8_t *p = reinterpret_cast<uint8_t *>(b) + dw * 4 + (nib >> 1);
        *p |= (nib & 1) ? (val << 4) : val;
    };

    // --- Probe 1: A byte j -> (row, k). B col 0 = all-ones over k. Expect the
    // hit at D[m=j/32][n=0] i.e. flat dword (j/32)*16.
    printf("\nProbe 1: A row map (expect byte j -> row j/32, contiguous):\n");
    int p1_bad = 0;
    for (int j = 0; j < 256; j++) {
        clr();
        for (int k = 0; k < 32; k++) b_set_nibble(k, 0, 1);
        reinterpret_cast<uint8_t *>(a)[j] = 3;
        run();
        int hits = 0, hit_dw = -1;
        for (int i = 0; i < 128; i++)
            if (d[i] != 0) { hits++; hit_dw = i; }
        const int expect_dw = (j / 32) * 16;
        if (hits != 1 || hit_dw != expect_dw || d[expect_dw] != 3) {
            if (p1_bad < 5)
                printf("  byte %d: hits=%d dw=%d (expect %d) val=%d\n",
                       j, hits, hit_dw, expect_dw, hit_dw >= 0 ? d[hit_dw] : 0);
            p1_bad++;
        }
    }
    printf("  A rows contiguous 32B, byte==k: %s (%d mismatches)\n",
           p1_bad ? "FAIL" : "PASS", p1_bad);

    // --- Probe 2: k alignment. A one-hot (row 0, k=j) x B one-hot (k=g, n=0):
    // fires iff j == g under the hypothesized B map.
    printf("Probe 2: A/B k alignment (expect pair g == j):\n");
    int p2_bad = 0;
    for (int j = 0; j < 32; j++) {
        int paired = -1;
        for (int g = 0; g < 32; g++) {
            clr();
            reinterpret_cast<uint8_t *>(a)[j] = 2;
            b_set_nibble(g, 0, 1);
            run();
            if (d[0] == 2) { paired = g; break; }
        }
        if (paired != j) {
            printf("  k=%d paired with g=%d\n", j, paired);
            p2_bad++;
        }
    }
    printf("  B VNNI map (dword (k/8)*16+n, nibble k%%8): %s\n",
           p2_bad ? "FAIL" : "PASS");

    // --- Probe 3: output routing. B one-hot (k=0, n=5), A row 2 all-ones.
    // Expect exactly D[2][5] == 1 (flat dword 2*16+5).
    clr();
    for (int k = 0; k < 32; k++) reinterpret_cast<uint8_t *>(a)[2 * 32 + k] = 1;
    b_set_nibble(0, 5, 1);
    run();
    int p3_bad = 0;
    for (int i = 0; i < 128; i++) {
        const int expect = (i == 2 * 16 + 5) ? 1 : 0;
        if (d[i] != expect) p3_bad++;
    }
    printf("Probe 3: D routing (m,n)->dword m*16+n: %s\n", p3_bad ? "FAIL" : "PASS");

    // --- Probe 4: negative s4 code. B code 0xF (= -1) x A +2 -> D[0][0] == -2.
    clr();
    reinterpret_cast<uint8_t *>(a)[0] = 2;
    b_set_nibble(0, 0, 0xF);
    run();
    printf("Probe 4: s4 sign (want -2): got %d  %s\n", d[0],
           d[0] == -2 ? "PASS" : "FAIL");

    sycl::free(a, q); sycl::free(b, q); sycl::free(d, q);
    return 0;
}
