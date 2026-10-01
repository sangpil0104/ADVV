"""Minimal torch stand-in for the fake SAM 3 worker tests. Not a tensor library."""

import contextlib
import os

__version__ = "0.0-fixture"
bfloat16 = "bfloat16"


class _Flags:
    allow_tf32 = False


class backends:
    class cuda:
        matmul = _Flags()

    cudnn = _Flags()


class cuda:
    @staticmethod
    def is_available():
        return os.environ.get("ADVV_FAKE_SAM3") != "no_cuda"

    @staticmethod
    def reset_peak_memory_stats():
        pass

    @staticmethod
    def max_memory_allocated():
        return 0


def inference_mode():
    return contextlib.nullcontext()


def autocast(device, dtype=None):
    return contextlib.nullcontext()
