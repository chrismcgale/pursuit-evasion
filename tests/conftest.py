# Import the package before any test module imports numpy, so its OpenBLAS
# kernel pin (see pursuit_evasion._pin_openblas_kernel) takes effect.
import pursuit_evasion  # noqa: F401
