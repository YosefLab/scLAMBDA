"""Initialize native thread controls before launching annotation workers.

threadpoolctl <=3.6 on Linux discovers DSOs with a Python callback under
ld.so's dl_iterate_phdr lock. Concurrent NumPy/BLAS work can invert that
lock and the GIL. Use the upstream PR #243 /proc approach on those versions.
This is a process-local compatibility fix; the installed package is untouched.
"""
import logging
import os
from pathlib import Path
import sys
from threading import Lock

_lock = Lock()
_ready = False


def _libraries_from_maps(self):
    # Do not fall back to dl_iterate_phdr if procfs becomes inaccessible.
    # Failing initialization is preferable to silently restoring the deadlock.
    paths = set()
    with open('/proc/self/maps', 'rb') as stream:
        for line in stream:
            fields = line.rstrip(b'\n').split(None, 5)
            if len(fields) != 6:
                continue
            path = os.fsdecode(fields[5])
            if path.startswith('/') and '.so' in path:
                paths.add(path)
    for path in sorted(paths):
        if os.path.exists(path):
            self._make_controller_from_path(path)


def prepare_native_threads():
    """Apply compatibility protection and eagerly initialize sklearn once."""
    global _ready
    with _lock:
        if _ready:
            return
        import threadpoolctl
        from packaging.version import Version

        if sys.platform == 'linux' and Version(threadpoolctl.__version__) < Version('3.7'):
            if not Path('/proc/self/maps').is_file():
                raise RuntimeError('Safe native-library discovery requires /proc/self/maps')
            threadpoolctl.ThreadpoolController._find_libraries_with_dl_iterate_phdr = _libraries_from_maps
            logging.getLogger(__name__).info('Enabled threadpoolctl Linux deadlock compatibility fix')
        # Trigger discovery on the caller before the executor starts workers.
        import numpy
        import scipy.linalg
        from sklearn.utils.parallel import _get_threadpool_controller
        _get_threadpool_controller()
        _ready = True
