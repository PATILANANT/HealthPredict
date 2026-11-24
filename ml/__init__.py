import os

# joblib/loky otherwise shells out to `wmic` on Windows to count physical cores and prints a
# noisy (harmless) traceback when that tool is missing.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
