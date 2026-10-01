Test doubles for the SAM 3 proposal worker (CPU tests only).

`torch` and `sam3` here are tiny stand-ins placed on `PYTHONPATH` of the worker subprocess so the real
adapter (`advv.backends.sam3`) and worker protocol run without CUDA or weights. `sam3.__fixture__` makes the
adapter label its output `backend=fixture`, which only fake-generation runs accept. `ADVV_FAKE_SAM3`
selects an error path: `missing_keys`, `no_cuda`, `text_error`, `hang`.
