"""Producer + refit-filter tests for the NVFP4 per-token rollout path.

The bitwise test against vLLM's online-quant kernel is the P0 hard gate of
research/vllm-nvfp4-pertoken/IMPLEMENTATION-PLAN-te-nvfp4-pertoken.md.
"""

import importlib.util
import pathlib

import pytest
import torch


def _load_nvfp4_pertoken():
    """Import the module under test without triggering heavy package inits.

    The producer must be importable in environments without ray/transformers
    (training workers, barebones test containers), so the test loads it the
    same way: straight from the file when the package import fails.
    """
    try:
        from nemo_rl.models.generation.vllm.quantization import (  # noqa: PLC0415
            nvfp4_pertoken,
        )

        return nvfp4_pertoken
    except ImportError:
        path = (
            pathlib.Path(__file__).resolve().parents[4]
            / "nemo_rl/models/generation/vllm/quantization/nvfp4_pertoken.py"
        )
        spec = importlib.util.spec_from_file_location("nvfp4_pertoken", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


M = _load_nvfp4_pertoken()

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _vllm_reference():
    try:
        from vllm.model_executor.layers.quantization.online.nvfp4 import (  # noqa: PLC0415
            _quantize_moe_weight_to_nvfp4,
        )

        return _quantize_moe_weight_to_nvfp4
    except ImportError:
        return None


# ------------------------------------------------------------- producer


@cuda_only
def test_bitwise_vs_vllm_online_quant():
    """THE P0 gate: producer output ≡ vLLM's kernel output, bit for bit."""
    ref = _vllm_reference()
    if ref is None:
        pytest.skip("vLLM online-quant module unavailable")

    torch.manual_seed(0)
    for e, n, k in [(8, 256, 512), (4, 64, 128), (2, 96, 2048)]:
        w = torch.randn(e, n, k, dtype=torch.bfloat16, device="cuda") * 3.0
        q_ref, bs_ref, gs_ref = ref(w)
        q, bs, gs = M.quantize_nvfp4_weight(w)

        assert torch.equal(gs, gs_ref), f"global scales differ ({e},{n},{k})"
        assert torch.equal(
            bs.view(torch.uint8), bs_ref.view(torch.uint8)
        ), f"block scales differ ({e},{n},{k})"
        mismatch = (q != q_ref).sum().item()
        assert mismatch == 0, (
            f"packed weights differ at {mismatch}/{q.numel()} bytes ({e},{n},{k})"
        )


@cuda_only
def test_2d_matches_3d_per_expert():
    torch.manual_seed(1)
    w = torch.randn(4, 128, 256, dtype=torch.bfloat16, device="cuda")
    q3, bs3, gs3 = M.quantize_nvfp4_weight(w)
    for e in range(4):
        q2, bs2, gs2 = M.quantize_nvfp4_weight(w[e])
        assert torch.equal(q2, q3[e])
        assert torch.equal(bs2.view(torch.uint8), bs3[e].view(torch.uint8))
        assert torch.equal(gs2, gs3[e])


@cuda_only
def test_deterministic_and_layout():
    torch.manual_seed(2)
    w = torch.randn(2, 64, 160, dtype=torch.bfloat16, device="cuda")
    q1, bs1, gs1 = M.quantize_nvfp4_weight(w)
    q2, bs2, gs2 = M.quantize_nvfp4_weight(w.clone())
    assert torch.equal(q1, q2)
    assert torch.equal(bs1.view(torch.uint8), bs2.view(torch.uint8))
    assert torch.equal(gs1, gs2)

    assert q1.dtype == torch.uint8 and q1.shape == (2, 64, 80)
    assert bs1.dtype == torch.float8_e4m3fn and bs1.shape == (2, 64, 10)
    assert gs1.dtype == torch.float32 and gs1.shape == (2,)


@cuda_only
def test_roundtrip_dequant_close():
    torch.manual_seed(3)
    w = torch.randn(64, 256, dtype=torch.bfloat16, device="cuda")
    q, bs, gs = M.quantize_nvfp4_weight(w)

    lut = torch.tensor(M._E2M1_VALUES, device="cuda")
    lo, hi = (q & 0x7).long(), ((q >> 4) & 0x7).long()
    slo = torch.where((q >> 3) & 1 == 1, -1.0, 1.0)
    shi = torch.where((q >> 7) & 1 == 1, -1.0, 1.0)
    deq = torch.empty(64, 256, device="cuda")
    deq[:, 0::2] = lut[lo] * slo
    deq[:, 1::2] = lut[hi] * shi
    deq = deq.reshape(64, 16, 16) * bs.float().unsqueeze(-1)
    deq = deq.reshape(64, 256) * gs

    rel = (deq - w.float()).abs().mean() / w.float().abs().mean()
    assert rel < 0.10, f"round-trip relative error too high: {rel:.4f}"


def test_rejects_bad_shapes():
    with pytest.raises(ValueError):
        M.quantize_nvfp4_weight(torch.randn(16))
    with pytest.raises(AssertionError):
        M.quantize_nvfp4_weight(torch.randn(4, 20))  # K % 16 != 0


@cuda_only
def test_zero_block_yields_zero_codes():
    w = torch.zeros(16, 32, dtype=torch.bfloat16, device="cuda")
    w[0, 16:] = 1.0  # non-zero amax so global scale is finite
    q, bs, _ = M.quantize_nvfp4_weight(w)
    assert (q[0, :8] == 0).all()  # the all-zero block packs to zero codes


# ------------------------------------------------------------- refit filter


def _expert_stream(num_experts=4, n=32, k=64, layers=("model.layers.0",)):
    stream = []
    for layer in layers:
        for e in range(num_experts):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                shape = (n, k) if proj != "down_proj" else (k, n * 2)
                stream.append((f"{layer}.mlp.experts.{e}.{proj}.weight", torch.randn(*shape)))
    return stream


def test_filter_emits_fused_tensors_and_passes_rest():
    stream = _expert_stream(num_experts=4, n=32, k=64)
    stream.insert(0, ("model.layers.0.self_attn.q_proj.weight", torch.randn(8, 8)))
    stream.append(("model.layers.0.self_attn.attn.k_scale", torch.tensor(1.0)))
    out = dict(
        M.iter_nvfp4_pertoken_weights(iter(stream), quant_patterns=["*.experts.*"])
    )
    p = "model.layers.0.mlp.experts"
    assert out[f"{p}.w13_weight"].shape == (4, 64, 32)          # (E, 2N, K/2)
    assert out[f"{p}.w13_weight"].dtype == torch.uint8
    assert out[f"{p}.w13_weight_scale"].shape == (4, 64, 4)     # (E, 2N, K/16)
    assert out[f"{p}.w13_weight_scale_2"].shape == (4, 2)
    assert out[f"{p}.w2_weight"].shape == (4, 64, 32)           # (E, K, 2N/2)
    assert out[f"{p}.w2_weight_scale_2"].shape == (4,)
    assert torch.equal(out["model.layers.0.self_attn.q_proj.weight"], stream[0][1])
    assert "model.layers.0.mlp.experts.0.gate_proj.weight" not in out


def test_filter_w13_shares_one_global_scale_per_expert():
    """Gate+up must be quantized under ONE per-expert global scale.

    vLLM's ModelOptNvFp4FusedMoE.process_weights_after_loading keeps only
    w13_weight_scale_2[:, 0] for the whole fused tensor — per-projection
    scales silently decode the up half with the gate scale.
    """
    stream = _expert_stream(num_experts=2, n=16, k=32)
    tensors = {n: t for n, t in stream}
    out = dict(M.iter_nvfp4_pertoken_weights(iter(stream), ["*.experts.*"]))
    p = "model.layers.0.mlp.experts"
    s2 = out[f"{p}.w13_weight_scale_2"]
    assert torch.equal(s2[:, 0], s2[:, 1])
    for e in range(2):
        fused = torch.cat(
            [
                tensors[f"{p}.{e}.gate_proj.weight"],
                tensors[f"{p}.{e}.up_proj.weight"],
            ],
            dim=0,
        )
        fq, _, fs2 = M.quantize_nvfp4_weight(fused)
        assert torch.equal(out[f"{p}.w13_weight"][e], fq)
        assert torch.equal(s2[e, 0], fs2)


def test_filter_flushes_multiple_layers_in_order():
    stream = _expert_stream(
        num_experts=2, n=16, k=32, layers=("model.layers.0", "model.layers.1")
    )
    names = [n for n, _ in M.iter_nvfp4_pertoken_weights(iter(stream), ["*.experts.*"])]
    assert names.index("model.layers.0.mlp.experts.w13_weight") < names.index(
        "model.layers.1.mlp.experts.w13_weight"
    )
    assert len(names) == 12


def test_filter_respects_ignore_patterns():
    stream = _expert_stream(num_experts=2, n=16, k=32)
    stream.append(("m.shared_expert.gate_proj.weight", torch.randn(16, 32)))
    out = dict(
        M.iter_nvfp4_pertoken_weights(
            iter(stream),
            quant_patterns=["*expert*"],
            ignore_patterns=["*shared_expert*"],
        )
    )
    assert "model.layers.0.mlp.experts.w13_weight" in out
    assert out["m.shared_expert.gate_proj.weight"].dtype != torch.uint8


@cuda_only
def test_filter_keeps_device():
    stream = [
        (n, t.to("cuda")) for n, t in _expert_stream(num_experts=2, n=16, k=32)
    ]
    out = dict(M.iter_nvfp4_pertoken_weights(iter(stream), ["*.experts.*"]))
    assert all(t.device.type == "cuda" for t in out.values())


def test_filter_raises_when_nothing_quantized():
    stream = [("m.self_attn.q_proj.weight", torch.randn(8, 16))]
    with pytest.raises(RuntimeError, match="quantized 0 params"):
        list(M.iter_nvfp4_pertoken_weights(iter(stream), ["*.experts.*"]))


def test_expand_fused_roundtrips_to_per_expert_checkpoint_names():
    """Fused transport tensors must expand back to exactly the per-expert
    ModelOpt names RoutedExperts' expert mapping matches (defect #8: the raw
    w13_/w2_ names are silently dropped by vLLM's loader)."""
    stream = _expert_stream(num_experts=2, n=16, k=32)
    stream.insert(0, ("m.self_attn.q_proj.weight", torch.randn(8, 8)))
    fused = list(M.iter_nvfp4_pertoken_weights(iter(stream), ["*.experts.*"]))
    expanded = dict(M.expand_fused_expert_weights(iter(fused)))
    fused = dict(fused)

    # Passthrough tensor is untouched.
    assert torch.equal(expanded["m.self_attn.q_proj.weight"], stream[0][1])
    # No fused transport names survive expansion.
    assert not any(".experts.w13_" in n or ".experts.w2_" in n for n in expanded)

    p = "model.layers.0.mlp.experts"
    # 1 passthrough + 2 experts x 3 projections x 3 tensors
    assert len(expanded) == 1 + 2 * 3 * 3
    for e in range(2):
        assert torch.equal(
            expanded[f"{p}.{e}.gate_proj.weight"], fused[f"{p}.w13_weight"][e, :16]
        )
        assert torch.equal(
            expanded[f"{p}.{e}.up_proj.weight"], fused[f"{p}.w13_weight"][e, 16:]
        )
        assert torch.equal(
            expanded[f"{p}.{e}.down_proj.weight"], fused[f"{p}.w2_weight"][e]
        )
        assert torch.equal(
            expanded[f"{p}.{e}.gate_proj.weight_scale"].contiguous().view(torch.uint8),
            fused[f"{p}.w13_weight_scale"][e, :16].contiguous().view(torch.uint8),
        )
        assert torch.equal(
            expanded[f"{p}.{e}.down_proj.weight_scale_2"],
            fused[f"{p}.w2_weight_scale_2"][e],
        )
        # Shared gate/up global scale lands on both per-expert names as scalars.
        assert expanded[f"{p}.{e}.gate_proj.weight_scale_2"].dim() == 0
        assert torch.equal(
            expanded[f"{p}.{e}.gate_proj.weight_scale_2"],
            expanded[f"{p}.{e}.up_proj.weight_scale_2"],
        )


def test_rollout_config_defaults():
    cfg = M.NvFp4PerTokenRolloutConfig()
    assert cfg.enabled is False
    assert cfg.quant_patterns == ["*.experts.*"]
    assert cfg.resolved_ignore() == M.DEFAULT_NVFP4_IGNORE

    cfg2 = M.NvFp4PerTokenRolloutConfig.model_validate(
        {"enabled": True, "ignore": ["*foo*"], "unknown_key": 1}
    )
    assert cfg2.enabled and cfg2.resolved_ignore() == ["*foo*"]


# --------------------------------------------------------- worker resolution


def test_resolver_dispatch_and_mutual_exclusion():
    try:
        from nemo_rl.models.generation.vllm.utils import (  # noqa: PLC0415
            resolve_generation_worker_cls,
        )
    except ImportError:
        pytest.skip("nemo_rl full deps unavailable")

    base = "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker"
    assert resolve_generation_worker_cls(base, {}) == base
    assert (
        "nvfp4_pertoken_worker.NvFp4PerTokenGenerationWorker"
        in resolve_generation_worker_cls(
            base, {"nvfp4_pertoken_rollout": {"enabled": True}}
        )
    )
    assert "VllmQuantGenerationWorker" in resolve_generation_worker_cls(
        base, {"quant_cfg": "some.yaml"}
    )
    with pytest.raises(ValueError, match="mutually exclusive|pick one"):
        resolve_generation_worker_cls(
            base,
            {"quant_cfg": "some.yaml", "nvfp4_pertoken_rollout": {"enabled": True}},
        )
