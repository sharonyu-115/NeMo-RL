# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Standalone validation of vLLM per-token NVFP4 activation scaling.

Checks (select with pytest -k):
  check_a  smoke + feature-active proof + TP negative test
  check_b  pre-quantized weights + per-token activations (the go/no-go)
  check_c  weight-reload determinism (RL-critical)
  check_d  logprob fidelity vs BF16 / static-scale / hybrid legs

Tests run in file order; check_d legs pass state via RESULTS_DIR files so a
single 4h sbatch (or selective re-runs) work the same way. One engine lives
at a time — every leg frees its engine before the next starts.
"""

import gc
import json
import os
import pathlib

import pytest
import torch

MODEL_BF16 = os.environ.get(
    "MODEL_BF16", "/lustre/fsw/general_sa/shuangy/models/Qwen/Qwen3-30B-A3B"
)
MODEL_NVFP4 = os.environ.get(
    "MODEL_NVFP4", "/lustre/fsw/general_sa/shuangy/models/nvidia/Qwen3-30B-A3B-NVFP4"
)
MODEL_SMALL = os.environ.get(
    "MODEL_SMALL",
    "/lustre/fsw/general_sa/shuangy/models/ibm-granite/granite-3.0-1b-a400m-base",
)
RESULTS_DIR = pathlib.Path(os.environ.get("RESULTS_DIR", "results/local"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# apply_model ships closures to the engine-core process.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

GEN_SEED = 1234
GEN_TOKENS = 256
SMOKE_PROMPTS = [
    "The capital of France is",
    "2 + 2 =",
    "def fibonacci(n):",
    "Water boils at a temperature of",
]
FIDELITY_PROMPTS = [
    "Explain why the sky is blue in two sentences.",
    "Write a Python function that reverses a linked list.",
    "What is 137 * 24? Show your work.",
    "Summarize the plot of Romeo and Juliet.",
    "Translate 'good morning' into French, Spanish, and German.",
    "List three causes of the French Revolution.",
    "A train travels 60 km in 45 minutes. What is its speed in km/h?",
    "Describe the process of photosynthesis.",
    "Write a haiku about mountains.",
    "What are the prime factors of 360?",
    "Explain the difference between TCP and UDP.",
    "If x + 3 = 11, what is 2x?",
    "Name the planets of the solar system in order.",
    "Write a SQL query to find duplicate emails in a table.",
    "What is the boiling point of water at high altitude, and why?",
    "Give a one-paragraph history of the printing press.",
    "Solve: a farmer has 17 sheep, all but 9 run away. How many are left?",
    "Explain recursion to a five-year-old.",
    "What is the derivative of x^3 + 2x?",
    "Describe three uses of machine learning in healthcare.",
    "Write a limerick about a cat and a laser pointer.",
    "How many seconds are in a leap year?",
    "Explain what an API is to a non-programmer.",
    "What is the chemical formula of table salt, and how does it dissolve?",
    "Compare and contrast lists and tuples in Python.",
    "A rectangle has area 48 and one side 6. What is its perimeter?",
    "Why do we have seasons on Earth?",
    "Write the first sentence of a mystery novel.",
    "What is the greatest common divisor of 84 and 126?",
    "Explain how a refrigerator keeps food cold.",
    "Name three famous theorems in mathematics and state one.",
    "What would happen if the Moon disappeared?",
]


# ---------------------------------------------------------------- helpers


def _make_llm(model: str, quantization: str | None = None, tp: int = 1, **kw):
    from vllm import LLM

    if not os.path.isdir(model) and "/" not in model:
        pytest.skip(f"model path {model} not available")
    return LLM(
        model=model,
        quantization=quantization,
        tensor_parallel_size=tp,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        **kw,
    )


def _free(llm) -> None:
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def _skip_if_missing(path: str) -> None:
    if not os.path.isdir(path):
        pytest.skip(f"model not downloaded yet: {path}")


def _moe_quant_method_names(llm) -> list[str]:
    """Type names of every FusedMoE/RoutedExperts quant method in the model."""

    def collect(model):
        names = []
        for module in model.modules():
            method = getattr(module, "_quant_method", None) or getattr(
                module, "quant_method", None
            )
            if method is not None and "MoE" in type(method).__name__:
                names.append(type(method).__name__)
        return names

    return [n for worker in llm.apply_model(collect) for n in worker]


def _static_input_scale_param_count(llm) -> int:
    """Count leftover checkpoint-shaped (E, 2) input-scale params on MoE layers."""

    def count(model):
        total = 0
        for name, param in model.named_parameters():
            if name.endswith(("w13_input_scale", "w2_input_scale")):
                if param.dim() >= 2:  # checkpoint layout, not neutral/converted
                    total += 1
        return total

    return sum(llm.apply_model(count))


def _greedy(llm, prompts: list[str], max_tokens: int = 32):
    from vllm import SamplingParams

    outs = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=0),
    )
    results = []
    for o in outs:
        comp = o.outputs[0]
        lps = [next(iter(d.values())).logprob for d in (comp.logprobs or [])]
        results.append(
            {
                "text": comp.text,
                "token_ids": list(comp.token_ids),
                "logprobs": lps,
            }
        )
    return results


def _sample(llm, prompts: list[str], max_tokens: int):
    from vllm import SamplingParams

    outs = llm.generate(
        prompts,
        SamplingParams(
            temperature=1.0, seed=GEN_SEED, max_tokens=max_tokens, logprobs=0
        ),
    )
    return [
        {
            "prompt_token_ids": list(o.prompt_token_ids),
            "gen_token_ids": list(o.outputs[0].token_ids),
            "gen_logprobs": [
                next(iter(d.values())).logprob for d in (o.outputs[0].logprobs or [])
            ],
        }
        for o in outs
    ]


def _score_sequences(llm, sequences: list[dict]) -> list[list[float]]:
    """Exact fprop logprobs of each sequence's generated tokens via prompt_logprobs."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    prompts = [
        TokensPrompt(prompt_token_ids=s["prompt_token_ids"] + s["gen_token_ids"])
        for s in sequences
    ]
    outs = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0),
    )
    scored = []
    for seq, out in zip(sequences, outs):
        n_gen = len(seq["gen_token_ids"])
        entries = out.prompt_logprobs[-n_gen:]
        token_ids = seq["gen_token_ids"]
        lps = []
        for tid, entry in zip(token_ids, entries):
            assert entry is not None and tid in entry, "scored token missing"
            lps.append(entry[tid].logprob)
        scored.append(lps)
    return scored


def _save(name: str, obj) -> None:
    torch.save(obj, RESULTS_DIR / f"{name}.pt")


def _load(name: str):
    path = RESULTS_DIR / f"{name}.pt"
    if not path.exists():
        pytest.skip(f"missing prerequisite artifact {path} (run earlier legs first)")
    return torch.load(path)


def _prob_mult_error(ref: list[list[float]], other: list[list[float]]) -> dict:
    diffs = []
    for r, o in zip(ref, other):
        n = min(len(r), len(o))
        diffs.extend(abs(a - b) for a, b in zip(r[:n], o[:n]))
    t = torch.tensor(diffs, dtype=torch.float64)
    return {
        "avg_prob_mult_error": torch.exp(t).mean().item(),
        "mean_abs_logprob_diff": t.mean().item(),
        "max_abs_logprob_diff": t.max().item(),
        "num_positions": t.numel(),
    }


# ---------------------------------------------------------------- check A


def test_check_a_smoke_small():
    """nvfp4_per_token loads a small MoE, generates, per-token method active."""
    _skip_if_missing(MODEL_SMALL)
    llm = _make_llm(MODEL_SMALL, quantization="nvfp4_per_token")
    names = _moe_quant_method_names(llm)
    assert names, "no MoE quant methods found"
    assert all(n == "Nvfp4OnlineMoEMethod" for n in names), names
    assert _static_input_scale_param_count(llm) == 0

    results = _greedy(llm, SMOKE_PROMPTS)
    for r in results:
        assert r["token_ids"], "empty generation"
        assert all(lp == lp for lp in r["logprobs"]), "NaN logprob"
    _save("check_a_small_outputs", results)
    _free(llm)


def test_check_a_smoke_qwen30b():
    """Same smoke on the RL-relevant Qwen3-30B-A3B."""
    _skip_if_missing(MODEL_BF16)
    llm = _make_llm(MODEL_BF16, quantization="nvfp4_per_token")
    names = _moe_quant_method_names(llm)
    assert names and all(n == "Nvfp4OnlineMoEMethod" for n in names), names
    results = _greedy(llm, SMOKE_PROMPTS)
    assert all(r["token_ids"] for r in results)
    _save("check_a_qwen30b_outputs", results)
    _free(llm)


def test_check_a_tp2_raises():
    """TP>1 is documented-unsupported; must fail loudly, not silently degrade."""
    _skip_if_missing(MODEL_SMALL)
    if torch.cuda.device_count() < 2:
        pytest.skip("needs 2 GPUs")
    with pytest.raises(Exception) as excinfo:
        llm = _make_llm(MODEL_SMALL, quantization="nvfp4_per_token", tp=2)
        _free(llm)
    msg = str(excinfo.value) + str(getattr(excinfo.value, "__cause__", ""))
    assert "NotImplemented" in msg or "not implemented" in msg.lower() or "TP" in msg, (
        f"TP=2 failed for an unexpected reason: {msg[:500]}"
    )


# ---------------------------------------------------------------- check B


def test_check_b1_external_quant_layout_and_determinism():
    """The refit export contract: upstream's quantizer output matches the
    ModelOpt checkpoint tensor layout and is bitwise deterministic."""
    from vllm.model_executor.layers.quantization.online.nvfp4 import (
        _quantize_moe_weight_to_nvfp4,
    )

    torch.manual_seed(0)
    w = torch.randn(8, 256, 512, dtype=torch.bfloat16, device="cuda")

    q1, bs1, gs1 = _quantize_moe_weight_to_nvfp4(w)
    q2, bs2, gs2 = _quantize_moe_weight_to_nvfp4(w.clone())

    # ModelOpt NVFP4 checkpoint layout: packed uint8 (E,N,K/2),
    # fp8-e4m3 block scales (E,N,K/16), fp32 per-expert global scale (E,)
    assert q1.dtype == torch.uint8 and q1.shape == (8, 256, 256)
    assert bs1.dtype == torch.float8_e4m3fn and bs1.shape == (8, 256, 32)
    assert gs1.dtype == torch.float32 and gs1.shape == (8,)

    assert torch.equal(q1, q2), "weight quantization is not deterministic"
    assert torch.equal(bs1.view(torch.uint8), bs2.view(torch.uint8))
    assert torch.equal(gs1, gs2)


def test_check_b2_hybrid_overlay():
    """GO/NO-GO: pre-quantized ModelOpt weights + per-token activations."""
    _skip_if_missing(MODEL_NVFP4)
    from pertoken_overlay import register_modelopt_fp4_pertoken

    register_modelopt_fp4_pertoken()
    llm = _make_llm(MODEL_NVFP4, quantization="modelopt_fp4_pertoken")
    names = _moe_quant_method_names(llm)
    assert names, "no MoE quant methods found"
    assert all(n == "ModelOptNvFp4PerTokenFusedMoE" for n in names), names

    results = _greedy(llm, SMOKE_PROMPTS)
    for r in results:
        assert r["token_ids"], "empty generation"
        assert all(lp == lp for lp in r["logprobs"]), "NaN logprob"
    _save("check_b2_hybrid_outputs", results)
    _free(llm)


# ---------------------------------------------------------------- check C


def _reload_weights(llm) -> None:
    try:
        llm.collective_rpc("reload_weights")
    except Exception as e:  # noqa: BLE001 - reported as a blocking finding
        pytest.fail(
            "BLOCKING FINDING: collective_rpc('reload_weights') unavailable or "
            f"failed — NeMo-RL refit contract needs an equivalent path: {e!r}"
        )


def _perturb_layer0_scale(llm) -> None:
    def perturb(model):
        for module in model.modules():
            if hasattr(module, "w13_weight_scale_2"):
                module.w13_weight_scale_2.data.mul_(2.0)
                method = getattr(module, "_quant_method", None) or getattr(
                    module, "quant_method", None
                )
                if hasattr(method, "moe_quant_config"):
                    # rebuild kernel view of the scales
                    method.process_weights_after_loading(module)
                return True
        return False

    assert any(llm.apply_model(perturb)), "no MoE layer found to perturb"


def _run_reload_check(model: str, quantization: str) -> dict:
    llm = _make_llm(model, quantization=quantization)
    out0 = _greedy(llm, SMOKE_PROMPTS, max_tokens=64)

    # 1. identity reload: outputs must be reproduced exactly
    _reload_weights(llm)
    out1 = _greedy(llm, SMOKE_PROMPTS, max_tokens=64)
    ids_equal = all(a["token_ids"] == b["token_ids"] for a, b in zip(out0, out1))
    lp_max_diff = max(
        (
            abs(x - y)
            for a, b in zip(out0, out1)
            for x, y in zip(a["logprobs"], b["logprobs"])
        ),
        default=0.0,
    )

    # 2. corrupt: output must change (guards vacuous pass) ...
    _perturb_layer0_scale(llm)
    out2 = _greedy(llm, SMOKE_PROMPTS, max_tokens=64)
    corrupted_changed = any(
        a["token_ids"] != b["token_ids"]
        or any(abs(x - y) > 1e-3 for x, y in zip(a["logprobs"], b["logprobs"]))
        for a, b in zip(out0, out2)
    )

    # 3. ... and reload must restore the baseline exactly
    _reload_weights(llm)
    out3 = _greedy(llm, SMOKE_PROMPTS, max_tokens=64)
    restored = all(a["token_ids"] == b["token_ids"] for a, b in zip(out0, out3))

    _free(llm)
    return {
        "identity_ids_equal": ids_equal,
        "identity_logprob_max_diff": lp_max_diff,
        "corrupt_changed_output": corrupted_changed,
        "reload_restored_output": restored,
    }


def test_check_c_reload_pertoken_small():
    _skip_if_missing(MODEL_SMALL)
    r = _run_reload_check(MODEL_SMALL, "nvfp4_per_token")
    _save("check_c_pertoken_small", r)
    assert r["identity_ids_equal"], f"identity reload changed outputs: {r}"
    assert r["identity_logprob_max_diff"] < 1e-5, r
    assert r["corrupt_changed_output"], (
        "perturbing scales did not change output — reload check is vacuous"
    )
    assert r["reload_restored_output"], f"reload did not restore baseline: {r}"


def test_check_c_reload_hybrid():
    """Reload path through the overlay (mirrors NeMo-RL per-step refit)."""
    _skip_if_missing(MODEL_NVFP4)
    from pertoken_overlay import register_modelopt_fp4_pertoken

    register_modelopt_fp4_pertoken()
    r = _run_reload_check(MODEL_NVFP4, "modelopt_fp4_pertoken")
    _save("check_c_hybrid", r)
    assert r["identity_ids_equal"], f"identity reload changed outputs: {r}"
    assert r["corrupt_changed_output"], "reload check is vacuous"
    assert r["reload_restored_output"], f"reload did not restore baseline: {r}"


# ---------------------------------------------------------------- check D


def test_check_d1_bf16_reference():
    _skip_if_missing(MODEL_BF16)
    llm = _make_llm(MODEL_BF16)
    seqs = _sample(llm, FIDELITY_PROMPTS, GEN_TOKENS)
    # exact fprop logprobs from the same engine (sampling-path logprobs can
    # differ from prefill scoring; use the scored ones as the reference)
    ref_scores = _score_sequences(llm, seqs)
    _save("check_d_sequences", seqs)
    _save("check_d_scores_bf16", ref_scores)
    _free(llm)
    assert sum(len(s) for s in ref_scores) > 0


def test_check_d2_pertoken_scores():
    _skip_if_missing(MODEL_BF16)
    seqs = _load("check_d_sequences")
    llm = _make_llm(MODEL_BF16, quantization="nvfp4_per_token")
    _save("check_d_scores_pertoken", _score_sequences(llm, seqs))
    _free(llm)


def test_check_d3_static_scores():
    _skip_if_missing(MODEL_NVFP4)
    seqs = _load("check_d_sequences")
    llm = _make_llm(MODEL_NVFP4)  # stock modelopt_fp4, static input_scale
    _save("check_d_scores_static", _score_sequences(llm, seqs))
    _free(llm)


def test_check_d4_hybrid_scores():
    _skip_if_missing(MODEL_NVFP4)
    from pertoken_overlay import register_modelopt_fp4_pertoken

    register_modelopt_fp4_pertoken()
    seqs = _load("check_d_sequences")
    llm = _make_llm(MODEL_NVFP4, quantization="modelopt_fp4_pertoken")
    _save("check_d_scores_hybrid", _score_sequences(llm, seqs))
    _free(llm)


def test_check_d5_report():
    ref = _load("check_d_scores_bf16")
    report = {}
    for leg in ("pertoken", "static", "hybrid"):
        path = RESULTS_DIR / f"check_d_scores_{leg}.pt"
        if path.exists():
            report[leg] = _prob_mult_error(ref, torch.load(path))
    (RESULTS_DIR / "check_d_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    assert "pertoken" in report, "per-token leg missing"
    # provisional hard gate
    assert report["pertoken"]["avg_prob_mult_error"] <= 1.20, report
    # the thesis: same quantized weights, dynamic per-token activation scales
    # must not be worse than the calibrated static ones
    if "static" in report and "hybrid" in report:
        assert (
            report["hybrid"]["avg_prob_mult_error"]
            <= report["static"]["avg_prob_mult_error"] * 1.05
        ), report
