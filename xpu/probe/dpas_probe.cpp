// dpas_probe.cpp — Phase 0 probe harness for knivesysl-xe (Intel Arc Pro B70, Xe2/BMG-G31)
//
// Measures, per device:
//   1. DPAS issue rate for every precision the hardware might support:
//      hf, bf, tf32, s8s8, u8u8, s4s4, u4u4, s8s4, s4s8  (M=8, N=16, K=256/max(bits))
//      - all-ones operand pattern makes the expected accumulator value == iters*K,
//        which validates BOTH that the MAC executed and the actual K depth.
//      - each config is JIT-compiled independently; an unsupported precision shows
//        up as a caught exception, not a dead binary.
//   2. DRAM bandwidth (triad + pure-read).
//   3. Kernel submission latency (in-order queue, async chain vs per-launch sync).
//   4. Cross-device copy bandwidth (P2P or staged, whatever the runtime gives us).
//
// Build: icpx -fsycl -O2 -o dpas_probe dpas_probe.cpp
// Run:   ./dpas_probe [device_index|all]
//
// The inline-vISA idiom below is copied from sycl-tla's cute/arch/mma_xe.hpp
// (XE_DPAS_TT), which is the same mechanism the engine kernels will use.

#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>

namespace syclex = sycl::ext::oneapi::experimental;

using uintv4  = uint32_t __attribute__((ext_vector_type(4)));   // A operand: 4 dwords/lane (M=8, any prec)
using uintv8  = uint32_t __attribute__((ext_vector_type(8)));   // B operand: 8 dwords/lane (any prec)
using floatv8 = float    __attribute__((ext_vector_type(8)));   // D/C fp accumulator: 8/lane
using intv8   = int32_t  __attribute__((ext_vector_type(8)));   // D/C int accumulator: 8/lane

static constexpr int SG   = 16;    // subgroup size (DPAS execution width)
static constexpr int M    = 8;     // repeat count
static constexpr int NCH  = 8;     // independent accumulator chains (hide systolic latency)
static constexpr int ITER = 4096;  // inner iterations; dpas count = ITER*NCH

// One DPAS probe kernel per precision pair. TB/TA are vISA precision tokens
// (order in the mnemonic is dpas.<src1=B>.<src2=A> per sycl-tla), DTOK is the
// vISA .decl type of the accumulator (F or D), DVEC the matching vector type.
// The dpas inline vISA is only valid in the device compilation pass; the host
// pass sees a no-op (identical to sycl-tla's CUTE_ARCH_MMA_XE_ENABLED guard).
#ifdef __SYCL_DEVICE_ONLY__
#define DPAS_INSN(TBTOK, TATOK, DTOK, dd, aa, bb)                                          \
  asm("{\n"                                                                                \
      ".decl DST     v_type=G type=" DTOK " num_elts=128 alias=<%0,0>\n"                   \
      ".decl SRC1_UD v_type=G type=UD num_elts=128 alias=<%2,0>\n"                         \
      ".decl SRC2_UD v_type=G type=UD num_elts=64 alias=<%1,0>\n"                          \
      "dpas." TBTOK "." TATOK ".8.8 (M1, 16) DST.0 DST.0 SRC1_UD.0 SRC2_UD(0,0)\n"         \
      "}\n"                                                                                \
      : "+rw"(dd) : "rw"(aa), "rw"(bb))
#else
#define DPAS_INSN(TBTOK, TATOK, DTOK, dd, aa, bb) do { (void)(aa); (void)(bb); } while (0)
#endif

#define DECL_DPAS_PROBE(NAME, TBTOK, TATOK, DTOK, DVEC)                                    \
  struct probe_##NAME {                                                                    \
    DVEC*     out;                                                                         \
    uint32_t  apat, bpat;                                                                  \
    void operator()(sycl::nd_item<1> it) const {                                           \
      uintv4 a; uintv8 b;                                                                  \
      for (int i = 0; i < 4; i++) a[i] = apat;                                             \
      for (int i = 0; i < 8; i++) b[i] = bpat;                                             \
      DVEC d[NCH];                                                                         \
      for (int c = 0; c < NCH; c++) for (int i = 0; i < 8; i++) d[c][i] = 0;               \
      for (int t = 0; t < ITER; t++) {                                                     \
        _Pragma("unroll")                                                                  \
        for (int c = 0; c < NCH; c++) {                                                    \
          DPAS_INSN(TBTOK, TATOK, DTOK, d[c], a, b);                                       \
        }                                                                                  \
      }                                                                                    \
      /* publish so nothing is dead code; lane 0 of subgroup 0 writes chain sums */        \
      DVEC acc = d[0];                                                                     \
      for (int c = 1; c < NCH; c++) for (int i = 0; i < 8; i++) acc[i] += d[c][i];         \
      if (it.get_global_linear_id() == 0) *out = acc;                                      \
    }                                                                                      \
  };

// fp accumulators
DECL_DPAS_PROBE(hf,   "hf",   "hf",   "F", floatv8)
DECL_DPAS_PROBE(bf,   "bf",   "bf",   "F", floatv8)
DECL_DPAS_PROBE(tf32, "tf32", "tf32", "F", floatv8)
// int accumulators
DECL_DPAS_PROBE(s8s8, "s8", "s8", "D", intv8)
DECL_DPAS_PROBE(u8u8, "u8", "u8", "D", intv8)
DECL_DPAS_PROBE(s4s4, "s4", "s4", "D", intv8)
DECL_DPAS_PROBE(u4u4, "u4", "u4", "D", intv8)
DECL_DPAS_PROBE(s8s4, "s4", "s8", "D", intv8)  // B=s4, A=s8
DECL_DPAS_PROBE(s4s8, "s8", "s4", "D", intv8)  // B=s8, A=s4

struct dpas_result {
  bool   ok = false;
  double tops = 0.0;
  double expected = 0.0, got = 0.0;
  std::string err;
};

template <typename Probe, typename DVEC>
static dpas_result run_dpas(sycl::queue& q, int wgs, int nwg, int K,
                            uint32_t apat, uint32_t bpat, double per_mac_expected) {
  dpas_result r;
  DVEC* out = sycl::malloc_shared<DVEC>(1, q);
  try {
    // warmup + JIT
    q.parallel_for(sycl::nd_range<1>(wgs, wgs),
                   Probe{out, apat, bpat}).wait_and_throw();
    const int n_sg = (nwg * wgs) / SG;
    auto t0 = std::chrono::steady_clock::now();
    auto ev = q.parallel_for(sycl::nd_range<1>(nwg * wgs, wgs), Probe{out, apat, bpat});
    ev.wait_and_throw();
    auto t1 = std::chrono::steady_clock::now();
    double s = std::chrono::duration<double>(t1 - t0).count();
    double macs = double(n_sg) * ITER * NCH * M * SG * K;
    r.tops = 2.0 * macs / s / 1e12;
    // validation: every accumulator element should be ITER * K * per_mac
    r.expected = double(ITER) * K * per_mac_expected;
    r.got = double((*out)[0]) / NCH;   // out is the sum of NCH chains
    r.ok = (r.expected != 0.0) && (r.got == r.expected * 1.0) ;
    if (!r.ok && r.got == 0.0) r.err = "zero result (instruction may have been dropped)";
    else if (!r.ok) r.err = "value mismatch";
    r.ok = r.got == r.expected;
  } catch (sycl::exception const& e) {
    r.err = e.what();
    // trim JIT spew to one line
    auto nl = r.err.find('\n');
    if (nl != std::string::npos) r.err = r.err.substr(0, nl);
  }
  sycl::free(out, q);
  return r;
}

// ---------------------------------------------------------------- bandwidth
struct triad_k {
  const float* a; const float* b; float* c; size_t n;
  void operator()(sycl::nd_item<1> it) const {
    size_t i = it.get_global_linear_id();
    size_t stride = it.get_global_range(0);
    for (size_t j = i; j < n; j += stride) c[j] = a[j] + 2.0f * b[j];
  }
};
struct read_k {
  const sycl::float4* a; float* out; size_t n4;
  void operator()(sycl::nd_item<1> it) const {
    size_t i = it.get_global_linear_id();
    size_t stride = it.get_global_range(0);
    sycl::float4 acc = {0,0,0,0};
    for (size_t j = i; j < n4; j += stride) acc += a[j];
    float s = acc.x() + acc.y() + acc.z() + acc.w();
    if (s == 1234567.0f) out[0] = s;  // never true; defeats DCE
  }
};

static void run_bandwidth(sycl::queue& q) {
  const size_t bytes = size_t(2) << 30;  // 2 GiB working set
  const size_t n = bytes / sizeof(float) / 3;
  float *a = sycl::malloc_device<float>(n, q),
        *b = sycl::malloc_device<float>(n, q),
        *c = sycl::malloc_device<float>(n, q);
  float *sink = sycl::malloc_device<float>(1, q);
  q.fill(a, 1.0f, n); q.fill(b, 2.0f, n); q.wait();

  const size_t g = 1024 * 256, l = 256;
  auto time_it = [&](auto&& launch) {
    launch(); q.wait();  // warm
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 3; i++) launch();
    q.wait();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count() / 3;
  };
  double st = time_it([&]{ q.parallel_for(sycl::nd_range<1>(g, l), triad_k{a, b, c, n}); });
  printf("  bandwidth triad (2r+1w)      : %8.1f GB/s\n", 3.0 * n * 4 / st / 1e9);
  double rt = time_it([&]{ q.parallel_for(sycl::nd_range<1>(g, l),
                                          read_k{(const sycl::float4*)a, sink, n / 4}); });
  printf("  bandwidth pure read          : %8.1f GB/s\n", 1.0 * n * 4 / rt / 1e9);
  sycl::free(a, q); sycl::free(b, q); sycl::free(c, q); sycl::free(sink, q);
}

// ---------------------------------------------------------- launch latency
struct empty_k { int* p; void operator()() const { if (p) *p = 1; } };

static void run_latency(sycl::queue& q) {
  int* p = sycl::malloc_device<int>(1, q);
  for (int i = 0; i < 100; i++) q.single_task(empty_k{p});
  q.wait();
  const int N = 2000;
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < N; i++) q.single_task(empty_k{p});
  q.wait();
  auto t1 = std::chrono::steady_clock::now();
  printf("  launch latency async chain   : %8.2f us/kernel\n",
         std::chrono::duration<double>(t1 - t0).count() / N * 1e6);
  t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < 200; i++) { q.single_task(empty_k{p}); q.wait(); }
  t1 = std::chrono::steady_clock::now();
  printf("  launch latency sync each     : %8.2f us/kernel\n",
         std::chrono::duration<double>(t1 - t0).count() / 200 * 1e6);
  sycl::free(p, q);
}

// ------------------------------------------------------------------- p2p
static void run_p2p(sycl::queue& q0, sycl::queue& q1) {
  const size_t bytes = size_t(1) << 30;
  auto d0 = q0.get_device(); auto d1 = q1.get_device();
  char* b0 = sycl::malloc_device<char>(bytes, q0);
  char* b1 = sycl::malloc_device<char>(bytes, q1);
  q0.memset(b0, 1, bytes).wait();
  bool peer = false;
  try {
    if (d0.ext_oneapi_can_access_peer(d1, sycl::ext::oneapi::peer_access::access_supported)) {
      d0.ext_oneapi_enable_peer_access(d1);
      d1.ext_oneapi_enable_peer_access(d0);
      peer = true;
    }
  } catch (sycl::exception const&) {}
  printf("  peer access supported        : %s\n", peer ? "yes" : "no (staged copy)");
  auto time_copy = [&](sycl::queue& q, char* dst, char* src) {
    q.memcpy(dst, src, bytes).wait();
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 3; i++) q.memcpy(dst, src, bytes);
    q.wait();
    auto t1 = std::chrono::steady_clock::now();
    return bytes * 3.0 / std::chrono::duration<double>(t1 - t0).count() / 1e9;
  };
  printf("  dev0 -> dev1 copy            : %8.1f GB/s\n", time_copy(q0, b1, b0));
  printf("  dev1 -> dev0 copy            : %8.1f GB/s\n", time_copy(q1, b0, b1));
  sycl::free(b0, q0); sycl::free(b1, q1);
}

// ------------------------------------------------------------------ main
int main(int argc, char** argv) {
  std::vector<sycl::device> gpus;
  for (auto& d : sycl::device::get_devices(sycl::info::device_type::gpu))
    if (d.get_backend() == sycl::backend::ext_oneapi_level_zero) gpus.push_back(d);
  if (gpus.empty()) { fprintf(stderr, "no level-zero GPUs found\n"); return 1; }

  printf("== knivesysl-xe phase 0 probe ==\n");
  for (size_t gi = 0; gi < gpus.size(); gi++) {
    auto& dev = gpus[gi];
    sycl::queue q(dev, sycl::property::queue::in_order{});
    auto name = dev.get_info<sycl::info::device::name>();
    int cu   = dev.get_info<sycl::info::device::max_compute_units>();
    long mem = (long)(dev.get_info<sycl::info::device::global_mem_size>() >> 20);
    int freq = dev.get_info<sycl::info::device::max_clock_frequency>();
    printf("\n-- device %zu: %s | CUs %d | %ld MiB | %d MHz --\n",
           gi, name.c_str(), cu, mem, freq);

    // grid: enough subgroups to saturate; wg 128 (8 subgroups), 8 wgs per CU
    const int wgs = 128;
    const int nwg = cu * 8;

    struct row { const char* name; int K; dpas_result r; };
    std::vector<row> rows;
    // all-ones patterns: hf 1.0=0x3C00, bf 1.0=0x3F80, tf32 1.0=0x3F800000,
    // s8 0x01 bytes, s4 0x11 nibbles
    rows.push_back({"hf   x hf   (K=16)", 16,
      run_dpas<probe_hf,   floatv8>(q, wgs, nwg, 16, 0x3C003C00u, 0x3C003C00u, 1.0)});
    rows.push_back({"bf   x bf   (K=16)", 16,
      run_dpas<probe_bf,   floatv8>(q, wgs, nwg, 16, 0x3F803F80u, 0x3F803F80u, 1.0)});
    rows.push_back({"tf32 x tf32 (K=8) ", 8,
      run_dpas<probe_tf32, floatv8>(q, wgs, nwg, 8, 0x3F800000u, 0x3F800000u, 1.0)});
    rows.push_back({"s8   x s8   (K=32)", 32,
      run_dpas<probe_s8s8, intv8>(q, wgs, nwg, 32, 0x01010101u, 0x01010101u, 1.0)});
    rows.push_back({"u8   x u8   (K=32)", 32,
      run_dpas<probe_u8u8, intv8>(q, wgs, nwg, 32, 0x01010101u, 0x01010101u, 1.0)});
    rows.push_back({"s4   x s4   (K=64)", 64,
      run_dpas<probe_s4s4, intv8>(q, wgs, nwg, 64, 0x11111111u, 0x11111111u, 1.0)});
    rows.push_back({"u4   x u4   (K=64)", 64,
      run_dpas<probe_u4u4, intv8>(q, wgs, nwg, 64, 0x11111111u, 0x11111111u, 1.0)});
    rows.push_back({"a:s8 x b:s4 (K=32)", 32,
      run_dpas<probe_s8s4, intv8>(q, wgs, nwg, 32, 0x01010101u, 0x11111111u, 1.0)});
    rows.push_back({"a:s4 x b:s8 (K=32)", 32,
      run_dpas<probe_s4s8, intv8>(q, wgs, nwg, 32, 0x11111111u, 0x01010101u, 1.0)});

    printf("  %-22s %10s %8s   %s\n", "dpas config", "TOPS", "check", "note");
    for (auto& x : rows) {
      if (x.r.err.empty())
        printf("  %-22s %10.1f %8s   %s\n", x.name, x.r.tops,
               x.r.ok ? "PASS" : "FAIL",
               x.r.ok ? "" : ("got " + std::to_string(x.r.got) + " want " +
                              std::to_string(x.r.expected)).c_str());
      else
        printf("  %-22s %10s %8s   %s\n", x.name, "-", "UNSUP", x.r.err.c_str());
    }

    run_bandwidth(q);
    run_latency(q);
  }

  if (gpus.size() >= 2) {
    printf("\n-- cross-device --\n");
    sycl::queue q0(gpus[0], sycl::property::queue::in_order{});
    sycl::queue q1(gpus[1], sycl::property::queue::in_order{});
    run_p2p(q0, q1);
  }
  printf("\ndone\n");
  return 0;
}
