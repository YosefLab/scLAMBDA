"""Regression coverage for native library discovery under concurrent work."""
import subprocess
import sys
import pytest


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux loader deadlock')
def test_concurrent_discovery_and_native_backtraces_finish():
    code = "import ctypes,threading,sys\nimport numpy as np\nfrom threadpoolctl import ThreadpoolController\nif '--fixed' in sys.argv:\n    from sclambda.core.native_threads import prepare_native_threads\n    prepare_native_threads()\nbarrier=threading.Barrier(8)\ngil=ctypes.PyDLL(None).backtrace;gil.argtypes=[ctypes.c_void_p,ctypes.c_int]\nnogil=ctypes.CDLL(None).backtrace;nogil.argtypes=gil.argtypes\ndef work():\n    buf=(ctypes.c_void_p*20)();barrier.wait()\n    for _ in range(10):\n        ThreadpoolController();gil(buf,20);nogil(buf,20)\nthreads=[threading.Thread(target=work) for _ in range(8)]\nfor t in threads:t.start()\nfor t in threads:t.join()\nprint('completed',flush=True)\n"
    result = subprocess.run([sys.executable, '-c', code, '--fixed'],
        capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    assert 'completed' in result.stdout


def test_native_library_detection_preserved():
    code = """
import numpy, scipy.linalg
from threadpoolctl import ThreadpoolController
from sclambda.core.native_threads import prepare_native_threads
before = {x['filepath'] for x in ThreadpoolController().info()}
prepare_native_threads()
after = {x['filepath'] for x in ThreadpoolController().info()}
assert before, 'No BLAS libraries detected'
assert before <= after, (before, after)
"""
    subprocess.run([sys.executable, '-c', code], check=True, timeout=30)
