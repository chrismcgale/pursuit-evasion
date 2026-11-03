"""pursuit_evasion: 2v2 3D pursuit-evasion with BT-gated self-play policies."""
import os as _os


def _pin_openblas_kernel() -> None:
    """Make numpy's 3-vector dot products bit-reproducible across CPUs.

    ``a @ b`` and ``np.linalg.norm`` on float64 vectors go through OpenBLAS
    ``ddot``. On AVX-512 hosts OpenBLAS selects its SkylakeX kernel, which uses
    fused multiply-adds and gives a different last bit on about a third of
    3-vector dots (measured: 6663/20000; ``norm`` 2145/20000). The C++ runtime
    computes ``(x0*y0 + x1*y1) + x2*y2`` and so does the Haswell/Sandybridge
    kernel, which is what the laptop and CI run. Without this pin, tag bt_safe
    parity fails on an AVX-512 machine and every gated tag number moves.

    OpenBLAS reads the variable once, when numpy is first imported, so this only
    works if ``pursuit_evasion`` is imported before numpy (true for every
    ``pe-*`` entry point; ``tests/conftest.py`` does it for pytest).
    ``test_small_dot_is_sequential`` fails if it did not take effect. Only set
    on AVX-512 hosts: forcing the Haswell kernel on a CPU without AVX2 would
    crash.
    """
    if "OPENBLAS_CORETYPE" in _os.environ:
        return
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    if " avx512f" in line:
                        _os.environ["OPENBLAS_CORETYPE"] = "Haswell"
                    return
    except OSError:
        pass


_pin_openblas_kernel()

__version__ = "0.1.0"
