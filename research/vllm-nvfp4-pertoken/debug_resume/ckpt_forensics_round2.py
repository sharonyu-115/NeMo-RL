# Round 2: discover real decoder-layer FQNs, sample transformer/expert tensors,
# and track optimizer main-param/exp_avg evolution across checkpoints.
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
    os.environ.setdefault("MASTER_PORT", "29512")
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)


def load_one(path, key, size, dtype):
    sd = {key: torch.empty(size, dtype=dtype)}
    dcp.load(sd, storage_reader=FileSystemReader(path))
    return sd[key]


def stats_vs(a, b):
    a = a.float()
    b = b.float()
    rd = (a - b).norm().item() / (b.norm().item() + 1e-12)
    frac_diff = (a != b).float().mean().item()
    maxabs = (a - b).abs().max().item()
    return rd, frac_diff, maxabs


def main():
    init_single_rank()
    torch.set_grad_enabled(False)

    metas = {n: FileSystemReader(p).read_metadata() for n, p in CKPTS.items()}
    keys120 = sorted(metas["step_120"].state_dict_metadata.keys())

    print("=" * 100)
    print("KEY NAMESPACE SAMPLE (step_120)")
    seen_prefix = set()
    for k in keys120:
        prefix = k.split(".")[0]
        if prefix not in seen_prefix:
            seen_prefix.add(prefix)
            print(f"  [prefix {prefix}] example: {k}")
    print("\n  keys containing 'decoder' (first 25):")
    dec = [k for k in keys120 if "decoder" in k]
    for k in dec[:25]:
        md = metas["step_120"].state_dict_metadata[k]
        size = getattr(md, "size", None)
        print(f"    {k} size={tuple(size) if size is not None else '?'}")
    print(f"  ... total decoder keys: {len(dec)}")
    opt = [k for k in keys120 if "optimizer" in k]
    print(f"\n  optimizer keys ({len(opt)} total, first 12):")
    for k in opt[:12]:
        md = metas["step_120"].state_dict_metadata[k]
        size = getattr(md, "size", None)
        print(f"    {k} size={tuple(size) if size is not None else '?'}")

    # --- sample model tensors: qkv/proj/router/experts across a few layers ---
    def pick(substrs, pool, limit=1):
        out = []
        for k in pool:
            if all(s in k for s in substrs):
                out.append(k)
                if len(out) >= limit:
                    break
        return out

    sample = []
    for pat in [
        ["linear_qkv.weight"],
        ["linear_proj.weight"],
        ["router.weight"],
        ["experts", "linear_fc1"],
        ["experts", "linear_fc2"],
        ["mlp.linear_fc1.weight"],
        ["layernorm"],
    ]:
        sample += pick(pat, dec, limit=3)
    sample = list(dict.fromkeys(sample))[:18]

    names = ["base", "step_90", "step_110", "step_120", "step_130"]
    print("\n" + "=" * 100)
    print("MODEL TENSORS: rel-diff vs base | frac elements differing vs base   (per checkpoint)")
    for key in sample:
        try:
            md = metas["step_120"].state_dict_metadata[key]
            size, dtype = md.size, md.properties.dtype
            tensors = {}
            for n in names:
                if key in metas[n].state_dict_metadata:
                    tensors[n] = load_one(CKPTS[n], key, size, dtype)
            if "base" not in tensors:
                print(f"{key}: not in base (skipped)")
                continue
            cols = []
            for n in names[1:]:
                if n in tensors:
                    rd, fd, mx = stats_vs(tensors[n], tensors["base"])
                    cols.append(f"{n}: rd={rd:.2e} frac={fd:.3f}")
            print(f"{key}\n    " + " | ".join(cols))
            c1 = stats_vs(tensors["step_110"], tensors["step_90"])
            c2 = stats_vs(tensors["step_120"], tensors["step_110"])
            c3 = stats_vs(tensors["step_130"], tensors["step_120"])
            print(
                f"    consec rd: 90->110={c1[0]:.2e} 110->120={c2[0]:.2e} 120->130={c3[0]:.2e}"
                f"  frac: {c1[1]:.3f} / {c2[1]:.3f} / {c3[1]:.3f}"
            )
            del tensors
        except Exception as e:  # noqa: BLE001
            print(f"{key}: FAILED: {e}")

    # --- optimizer main params + exp_avg evolution (not in base) ---
    print("\n" + "=" * 100)
    print("OPTIMIZER STATE EVOLUTION (main 'param' + 'exp_avg' buckets, run ckpts only)")
    opt_sample = [k for k in opt if k.endswith(".param")][:2] + [k for k in opt if k.endswith(".exp_avg")][:2]
    run_names = ["step_90", "step_110", "step_120", "step_130"]
    for key in opt_sample:
        try:
            md = metas["step_120"].state_dict_metadata[key]
            size, dtype = md.size, md.properties.dtype
            tensors = {n: load_one(CKPTS[n], key, size, dtype) for n in run_names if key in metas[n].state_dict_metadata}
            line = [f"{key} size={tuple(size)} dtype={dtype}"]
            for a, b in zip(run_names, run_names[1:]):
                if a in tensors and b in tensors:
                    rd, fd, mx = stats_vs(tensors[b], tensors[a])
                    line.append(f"    {a}->{b}: rd={rd:.3e} frac={fd:.3f} norm({b})={tensors[b].float().norm().item():.4e}")
            print("\n".join(line))
            del tensors
        except Exception as e:  # noqa: BLE001
            print(f"{key}: FAILED: {e}")


if __name__ == "__main__":
    sys.exit(main())
