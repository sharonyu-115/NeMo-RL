"""Probe check B2 against the GRADUATED production module (Step 10 of the
implementation plan): pre-quantized ModelOpt weights + per-token activations
through nemo_rl's nvfp4_pertoken_vllm, not the research overlay.

Run with the shim tree on PYTHONPATH so every process — including vLLM's
EngineCore subprocess, which re-imports the pickled quantization config by
package name — resolves nemo_rl.* without the heavy runtime deps:

    PYTHONPATH=research/vllm-nvfp4-pertoken/shim python3 smoke_graduated_module.py

(In the real nemo-rl venv the full package imports fine and no shim is used.)
"""

import os

MODEL_NVFP4 = os.environ.get(
    "MODEL_NVFP4", "/lustre/fsw/general_sa/shuangy/models/nvidia/Qwen3-30B-A3B-NVFP4"
)

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


def main() -> None:
    from nemo_rl.models.generation.vllm.quantization import nvfp4_pertoken_vllm as vmod

    vmod.register_nvfp4_pertoken()
    print("registered:", vmod.NVFP4_PER_TOKEN_METHOD)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL_NVFP4,
        quantization=vmod.NVFP4_PER_TOKEN_METHOD,
        tensor_parallel_size=1,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
    )

    def check(model):
        names = []
        for module in model.modules():
            m = getattr(module, "_quant_method", None) or getattr(
                module, "quant_method", None
            )
            if m is not None and "MoE" in type(m).__name__:
                names.append(type(m).__name__)
        return names

    names = [n for worker in llm.apply_model(check) for n in worker]
    assert names and all(n == "ModelOptNvFp4PerTokenFusedMoE" for n in names), names
    print(f"quant method active on {len(names)} MoE layers")

    outs = llm.generate(
        ["The capital of France is", "2 + 2 ="],
        SamplingParams(temperature=0.0, max_tokens=24, logprobs=0),
    )
    for o in outs:
        text = o.outputs[0].text
        lps = [next(iter(d.values())).logprob for d in (o.outputs[0].logprobs or [])]
        assert o.outputs[0].token_ids and all(x == x for x in lps)
        print("gen:", repr(text[:60]))

    print("SMOKE_GRADUATED_OK")


if __name__ == "__main__":
    main()
