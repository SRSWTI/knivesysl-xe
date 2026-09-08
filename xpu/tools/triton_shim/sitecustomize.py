"""Force Triton onto the Intel XPU driver in EVERY spawned process.

This box has an NVIDIA 5090 alongside two Intel B70s, so Triton's driver
auto-selection finds two candidates and refuses:

    RuntimeError: 2 active drivers ([XPUDriver, CudaDriver]).
                  There should only be one.

The consequence is silent and expensive. Triton never initializes, so in
vLLM-XPU:
  - `--attention-backend TRITON_ATTN` cannot actually engage,
  - the Triton rejection sampler is unavailable, which disables MTP
    speculative decoding,
  - and FULL XPU-graph capture is blocked (flash-attn FULL is separately
    blocked by SYCL-Graph work_group_scratch_memory, so TRITON_ATTN is the
    only route to FULL).

vLLM spawns its engine core as a separate process, so fixing this in the
launching shell is not enough - the driver choice has to be forced on every
interpreter start. That is what sitecustomize.py on PYTHONPATH buys, and it is
the same trick b70_ai_things calls TRITONSHIM (bin/30_serve_w4a8_graph.sh):
"inject a sitecustomize.py that warms torch.xpu.device_count() in EVERY spawned
process, so triton's is_active() returns True in the engine worker".

Deliberately silent on failure: if a future stack resolves the ambiguity by
itself, this must not break the launch.
"""


def _force_xpu_triton() -> None:
    try:
        import torch

        # Warm the XPU runtime first. Triton's is_active() for the Intel
        # backend consults torch.xpu availability, which is lazily initialized.
        if hasattr(torch, "xpu"):
            torch.xpu.device_count()
    except Exception:
        pass

    try:
        from triton.backends.intel.driver import XPUDriver
        from triton.runtime import driver

        driver.set_active(XPUDriver())
    except Exception:
        pass

    # vLLM 0.27's importing.py does not consult the ACTIVE driver - it counts
    # every backend whose is_active() returns True and disables Triton
    # entirely when it sees two (XPU + CUDA on this dual-vendor box). The
    # 5090 only drives the display; make its Triton backend report inactive
    # so the count is one and TRITON_ATTN/XPU-graph stay available.
    try:
        from triton.backends.nvidia.driver import CudaDriver

        CudaDriver.is_active = staticmethod(lambda: False)
    except Exception:
        pass


_force_xpu_triton()
