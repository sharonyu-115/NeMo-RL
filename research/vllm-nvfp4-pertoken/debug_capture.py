# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Faithful 1-GPU isolation reproducer for the NVFP4 routed-experts capture bug.

The capture runs in vLLM WORKER subprocesses, so instrumentation must be
installed there via ``collective_rpc`` (a driver-side monkeypatch never fires).
This probe wraps ``RoutedExpertsCapturer.capture`` in-worker and records, from
the RAW ``topk_ids`` the router hands the hook, any all-zero / duplicate top-k
rows -- the decisive fork:

  raw already has zero/dup rows  -> router/kernel produces bad routing
  raw clean but returned routes zero -> slice / D2H / slot-mapping drops them

Run (bf16 vLLM venv has everything; nvfp4 adds the overlay):
    MODEL=$MODEL_NVFP4 QUANT=modelopt_fp4_pertoken python debug_capture.py
    MODEL=$MODEL_BF16  QUANT=                       python debug_capture.py   # control
"""

import os

# ---- module-level probes (picklable; run INSIDE each vLLM worker) ----------


def _install_capture_probe(worker):
    import torch
    from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
        RoutedExpertsCapturer,
    )

    if getattr(RoutedExpertsCapturer, "_probe_installed", False):
        return "already"
    orig = RoutedExpertsCapturer.capture
    RoutedExpertsCapturer._probe_stats = []
    RoutedExpertsCapturer._probe_calls = 0

    def capture(self, layer_id, topk_ids):
        try:
            RoutedExpertsCapturer._probe_calls += 1
            srt = topk_ids.sort(dim=-1).values
            dup = (srt[..., 1:] == srt[..., :-1]).any(dim=-1)
            zero = (topk_ids == 0).all(dim=-1)
            nd, nz = int(dup.sum().item()), int(zero.sum().item())
            if (nd or nz) and layer_id <= 3:
                bad = torch.nonzero(dup).flatten()[:3].tolist()
                RoutedExpertsCapturer._probe_stats.append(
                    (
                        int(layer_id),
                        int(topk_ids.shape[0]),
                        nd,
                        nz,
                        bad,
                        topk_ids[bad[0]].tolist() if bad else None,
                    )
                )
        except Exception as e:  # noqa: BLE001
            RoutedExpertsCapturer._probe_stats.append(("err", repr(e)))
        return orig(self, layer_id, topk_ids)

    RoutedExpertsCapturer.capture = capture
    RoutedExpertsCapturer._probe_installed = True
    return "installed"


def _read_capture_probe(worker):
    from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
        RoutedExpertsCapturer,
    )

    return {
        "calls": getattr(RoutedExpertsCapturer, "_probe_calls", None),
        "bad_rows": getattr(RoutedExpertsCapturer, "_probe_stats", ["no-probe"])[:15],
    }


def _inspect_binding(worker):
    """Why is capture not binding? Report runner state + MoE/router module types
    and whether each router has a capture_fn set."""
    try:
        from vllm.model_executor.layers.fused_moe.layer import MoERunner
        from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    except Exception as e:  # noqa: BLE001
        return {"import_err": repr(e)}
    mr = getattr(worker, "model_runner", None)
    info = {
        "has_model_runner": mr is not None,
        "routed_experts_initialized": getattr(mr, "routed_experts_initialized", "?"),
        "enable_flag": getattr(
            getattr(mr, "model_config", None), "enable_return_routed_experts", "?"
        ),
    }
    moe_runners = 0
    router_types = {}
    capfn_set = 0
    router_is_baserouter = 0
    example = None
    if mr is not None and getattr(mr, "model", None) is not None:
        for m in mr.model.modules():
            if isinstance(m, MoERunner):
                moe_runners += 1
                r = getattr(m, "router", None)
                rt = type(r).__name__ if r is not None else "None"
                router_types[rt] = router_types.get(rt, 0) + 1
                if isinstance(r, BaseRouter):
                    router_is_baserouter += 1
                if getattr(r, "capture_fn", None) is not None:
                    capfn_set += 1
                if example is None:
                    example = f"MoERunner.router={rt} (BaseRouter={isinstance(r, BaseRouter)})"
    info.update(
        moe_runner_count=moe_runners,
        router_types=router_types,
        routers_that_are_BaseRouter=router_is_baserouter,
        routers_with_capture_fn=capfn_set,
        example=example,
    )
    return info


def _apply_monolithic_capture_patch(worker):
    """PROPOSED FIX: monolithic fused-MoE (forward_monolithic) skips
    router.select_experts, so the routing-capture hook never fires. Fire it
    explicitly when capture is active. (Faithfulness of these routes vs the
    kernel's internal top-k is exactly what this test checks.)"""
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    if getattr(MoERunner, "_mono_capture_patched", False):
        return "already"
    orig = MoERunner._apply_quant_method

    def _apply_quant_method(
        self, hidden_states, router_logits, shared_experts_input, input_ids=None
    ):
        r = getattr(self, "router", None)
        if (
            r is not None
            and getattr(r, "capture_fn", None) is not None
            and self.routed_experts.quant_method.is_monolithic
        ):
            r.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_indices_dtype=self._quant_method.topk_indices_dtype,
                input_ids=input_ids,
            )
        return orig(self, hidden_states, router_logits, shared_experts_input, input_ids)

    MoERunner._apply_quant_method = _apply_quant_method
    MoERunner._mono_capture_patched = True
    return "patched"


def main():
    model = os.environ["MODEL"]
    quant = os.environ.get("QUANT") or None
    if quant == "modelopt_fp4_pertoken":
        from pertoken_overlay import register_modelopt_fp4_pertoken

        register_modelopt_fp4_pertoken()

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        quantization=quant,
        tensor_parallel_size=1,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        enable_return_routed_experts=True,
    )

    print("[probe] install:", llm.collective_rpc(_install_capture_probe), flush=True)
    print("[bind] inspect:", llm.collective_rpc(_inspect_binding), flush=True)
    if os.environ.get("PATCH") == "1":
        print("[patch] apply:", llm.collective_rpc(_apply_monolithic_capture_patch), flush=True)
    prompts = [
        "Solve: what is 12 plus 30? Show your steps.",
        "Compute the area of a circle with radius 3.",
        "What is the 8th prime number? Explain briefly.",
        "Simplify (x^2 - 9)/(x - 3) and state the domain.",
    ]
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=24))
    stats = llm.collective_rpc(_read_capture_probe)
    print("\n==== IN-WORKER raw topk_ids probe (the decisive fork) ====", flush=True)
    for wi, st in enumerate(stats):
        print(f"worker{wi}: capture_calls={st['calls']}", flush=True)
        for row in st["bad_rows"]:
            print(f"   RAW bad-row: layer/ntok/ndup/nzero/first/sample = {row}", flush=True)
        if not st["bad_rows"]:
            print("   RAW: no zero/dup rows in captured topk_ids (clean)", flush=True)

    import torch

    print("\n==== RETURNED routes scan (driver side) ====", flush=True)
    for ri, o in enumerate(outs):
        cr = getattr(o.outputs[0], "routed_experts", None)
        if cr is None:
            print(f"req{ri}: routed_experts=None", flush=True)
            continue
        t = torch.as_tensor(cr)
        l0 = t[:, 0, :] if t.dim() == 3 else t
        srt = l0.sort(dim=-1).values
        dup = (srt[..., 1:] == srt[..., :-1]).any(dim=-1)
        bad = torch.nonzero(dup).flatten().tolist()
        print(
            f"req{ri}: shape={tuple(t.shape)} dup_rows_L0={len(bad)} first={bad[:5]} "
            f"sample={l0[bad[0]].tolist() if bad else None}",
            flush=True,
        )


if __name__ == "__main__":
    main()
