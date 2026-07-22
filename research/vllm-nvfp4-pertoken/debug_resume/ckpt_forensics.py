# Checkpoint tensor forensics for the bf16 resume reward-drop investigation.
# Compares per-tensor content across step_90/110/120/130 and the HF-import base
# to discriminate: H1 all-checkpoints-stale vs H2 step_120-only vs H3 load-side.
import json
import os
import sys

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader

RUN = "/lustre/fsw/general_sa/shuangy/src/NeMo-RL/nemo-rl/results/grpo-qwen3-30ba3b-base-8n4g-dapo512-20k-bf16-20260721"
BASE = "/lustre/fsw/general_sa/shuangy/hf/megatron_ckpt/model__lustre_fsw_general_sa_shuangy_models_Qwen_Qwen3-30B-A3B-Base/iter_0000000"
CKPTS = {
    "base": BASE,
    "step_90": f"{RUN}/step_90/policy/weights/iter_0000000",
    "step_110": f"{RUN}/step_110/policy/weights/iter_0000000",
    "step_120": f"{RUN}/step_120/policy/weights/iter_0000000",
    "step_130": f"{RUN}/step_130/policy/weights/iter_0000000",
}


def init_single_rank():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)


def pick_keys(meta):
    keys = sorted(meta.state_dict_metadata.keys())
    picked = []

    def grab(substrs, limit=1):
        got = 0
        for k in keys:
            if all(s in k for s in substrs) and k not in picked:
                picked.append(k)
                got += 1
                if got >= limit:
                    return

    grab(["embedding.word_embeddings.weight"])
    grab(["output_layer.weight"])
    grab(["final_layernorm.weight"])
    for layer in ["layers.0.", "layers.23.", "layers.47."]:
        grab([layer, "self_attention.linear_qkv.weight"])
        grab([layer, "self_attention.linear_proj.weight"])
        grab([layer, "router.weight"])
        grab([layer, "experts", "linear_fc1"], limit=2)
        grab([layer, "experts", "linear_fc2"], limit=1)
    return picked


def tensor_meta(meta, key):
    md = meta.state_dict_metadata[key]
    return md.size, md.properties.dtype


def load_one(path, key, size, dtype):
    sd = {key: torch.empty(size, dtype=dtype)}
    dcp.load(sd, storage_reader=FileSystemReader(path))
    return sd[key]


def rel_diff(a, b):
    a = a.float()
    b = b.float()
    denom = b.norm().item() + 1e-12
    return (a - b).norm().item() / denom


def main():
    init_single_rank()
    torch.set_grad_enabled(False)

    # --- common.pt / metadata sanity per checkpoint ---
    print("=" * 100)
    print("METADATA / common.pt SANITY")
    metas = {}
    for name, path in CKPTS.items():
        try:
            metas[name] = FileSystemReader(path).read_metadata()
            nkeys = len(metas[name].state_dict_metadata)
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] METADATA READ FAILED: {e}")
            continue
        line = f"[{name}] metadata keys={nkeys}"
        common = os.path.join(path, "common.pt")
        if os.path.exists(common):
            try:
                c = torch.load(common, map_location="cpu", weights_only=False)
                topk = list(c.keys()) if isinstance(c, dict) else type(c)
                line += f" common.pt top-level={topk}"
                if isinstance(c, dict):
                    if "iteration" in c:
                        line += f" iteration={c['iteration']}"
                    if "opt_param_scheduler" in c:
                        line += f" opt_param_scheduler={c['opt_param_scheduler']}"
                    if "optimizer" in c:
                        line += " has-optimizer=True"
            except Exception as e:  # noqa: BLE001
                line += f" common.pt LOAD FAILED: {e}"
        print(line)

    # Key-set differences (base is TP=1 import; run ckpts TP=2 — fqns should match)
    ref = set(metas["step_120"].state_dict_metadata)
    for name in CKPTS:
        if name in metas:
            d1 = ref - set(metas[name].state_dict_metadata)
            d2 = set(metas[name].state_dict_metadata) - ref
            if d1 or d2:
                print(f"[{name}] key diff vs step_120: missing={sorted(d1)[:5]} extra={sorted(d2)[:5]}")

    keys = pick_keys(metas["step_120"])
    print(f"\nSampled {len(keys)} tensor keys:")
    for k in keys:
        print(f"  {k}")

    # --- per-tensor cross-checkpoint comparison ---
    names = [n for n in ["base", "step_90", "step_110", "step_120", "step_130"] if n in metas]
    print("\n" + "=" * 100)
    header = f"{'tensor':68s} " + " ".join(f"{n + '-base':>13s}" for n in names[1:])
    print("RELATIVE DIFF ||x - base|| / ||base||   +   consecutive diffs")
    for key in keys:
        try:
            size, dtype = tensor_meta(metas["step_120"], key)
            tensors = {}
            for n in names:
                if key not in metas[n].state_dict_metadata:
                    print(f"{key}: MISSING in {n}")
                    continue
                tensors[n] = load_one(CKPTS[n], key, size, dtype)
            vs_base = " ".join(
                f"{rel_diff(tensors[n], tensors['base']):13.6f}" for n in names[1:] if n in tensors and "base" in tensors
            )
            consec = " ".join(
                f"{a}->{b}:{rel_diff(tensors[b], tensors[a]):.6f}"
                for a, b in zip(names[1:], names[2:])
                if a in tensors and b in tensors
            )
            print(f"{key}\n    vs base: {vs_base}\n    consec : {consec}")
            del tensors
        except Exception as e:  # noqa: BLE001
            print(f"{key}: FAILED: {e}")

    print("\nDecision guide: H1 stale-saves => step_90~=step_110~=step_120 (consec ~0, vs-base equal & small);")
    print("H2 step_120-only => 90/110 progressive, 110->120 anomalous (or 120 ~= base);")
    print("H3 load-side => all progressive sane deltas on disk.")


if __name__ == "__main__":
    sys.exit(main())
