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


def test_filter_quantizes_matching_and_passes_rest():
    stream = [
        ("model.layers.0.mlp.experts.3.gate_proj.weight", torch.randn(32, 64)),
        ("model.layers.0.self_attn.q_proj.weight", torch.randn(8, 8)),
        ("model.layers.0.mlp.experts.3.gate_proj.bias", torch.randn(32)),
        ("model.layers.0.self_attn.attn.k_scale", torch.tensor(1.0)),
    ]
    out = list(
        M.iter_nvfp4_pertoken_weights(iter(stream), quant_patterns=["*.experts.*"])
    )
    names = [n for n, _ in out]
    assert names == [
        "model.layers.0.mlp.experts.3.gate_proj.weight",
        "model.layers.0.mlp.experts.3.gate_proj.weight_scale",
        "model.layers.0.mlp.experts.3.gate_proj.weight_scale_2",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.experts.3.gate_proj.bias",
        "model.layers.0.self_attn.attn.k_scale",
    ]
    tensors = dict(out)
    assert tensors["model.layers.0.mlp.experts.3.gate_proj.weight"].dtype == torch.uint8
    assert torch.equal(tensors["model.layers.0.self_attn.q_proj.weight"], stream[1][1])


@cuda_only
def test_filter_keeps_device():
    stream = [("m.experts.0.up_proj.weight", torch.randn(16, 32, device="cuda"))]
    out = dict(M.iter_nvfp4_pertoken_weights(iter(stream), ["*.experts.*"]))
    assert all(t.device.type == "cuda" for t in out.values())


def test_hf_quant_config_shape():
    cfg = M.build_nvfp4_pertoken_hf_quant_config(["*lm_head*"])
    assert cfg["quant_algo"] == "NVFP4"
    assert cfg["exclude_modules"] == ["*lm_head*"]
    acts = cfg["config_groups"]["group_0"]["input_activations"]
    assert acts["dynamic"] is True
