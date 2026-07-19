"""Barebones-container stub of nemo_rl's vllm_backend (no ray dependency).

Only what nvfp4_pertoken_vllm needs at import time. The real worker-extension
behavior is NOT exercised through this stub — engine-level smokes only.
"""


class VllmInternalWorkerExtension:  # noqa: D101
    pass


WeightUpdateTransport = str
WeightUpdateFinalizer = object
