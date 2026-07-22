# Tensor-level staleness check for the NVFP4 run's checkpoints (step_120/130/140).
import os
import sys

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader

RUN = "/lustre/fsw/general_sa/shuangy/src/NeMo-RL/nemo-rl/results/grpo-qwen3-30ba3b-base-8n4g-dapo512-20k-nvfp4-pertoken-20260721"
BASE = "/lustre/fsw/general_sa/shuangy/hf/megatron_ckpt/model__lustre_fsw_general_sa_shuangy_models_Qwen_Qwen3-30B-A3B-Base/iter_0000000"
CKPTS = {
    "base": BASE,
    "step_120": f"{RUN}/step_120/policy/weights/iter_0000000",
    "step_130": f"{RUN}/step_130/policy/weights/iter_0000000",
    "step_140": f"{RUN}/step_140/policy/weights/iter_0000000",
}
NAMES = ["base", "step_120", "step_130", "step_140"]


def init_single_rank():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29513")
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
    return rd, frac_diff


def main():
    init_single_rank()
    torch.set_grad_enabled(False)

    metas = {}
    for n, p in CKPTS.items():
        try:
            metas[n] = FileSystemReader(p).read_metadata()
        except Exception as e:  # noqa: BLE001
            print(f"[{n}] METADATA READ FAILED: {e}")
    ref = "step_140" if "step_140" in metas else NAMES[-1]
    keys = sorted(metas[ref].state_dict_metadata.keys())

    def pick(substrs, limit=1):
        out = []
        for k in keys:
            md = metas[ref].state_dict_metadata[k]
            if all(s in k for s in substrs) and hasattr(md, "size"):
                out.append(k)
                if len(out) >= limit:
                    break
        return out

    sample = (
        pick(["embedding.word_embeddings.weight"])
        + pick(["output_layer.weight"])
        + pick(["linear_qkv.weight"])
        + pick(["linear_proj.weight"])
        + pick(["experts", "linear_fc1.weight"])
        + pick(["experts", "linear_fc2.weight"])
    )
    opt_keys = [k for k in keys if "optimizer" in k and k.endswith(".param")][:2] + [
        k for k in keys if "optimizer" in k and k.endswith(".exp_avg")
    ][:2]

    print(f"sampled model keys: {sample}")
    print(f"sampled optimizer keys: {opt_keys}")

    print("\nMODEL TENSORS")
    for key in sample:
        try:
            md = metas[ref].state_dict_metadata[key]
            tensors = {
                n: load_one(CKPTS[n], key, md.size, md.properties.dtype)
                for n in NAMES
                if n in metas and key in metas[n].state_dict_metadata
            }
            cols = []
            if "base" in tensors:
                for n in NAMES[1:]:
                    if n in tensors:
                        rd, fd = stats_vs(tensors[n], tensors["base"])
                        cols.append(f"{n} vs base: rd={rd:.2e} frac={fd:.3f}")
            consec = []
            run_names = [n for n in NAMES[1:] if n in tensors]
            for a, b in zip(run_names, run_names[1:]):
                rd, fd = stats_vs(tensors[b], tensors[a])
                consec.append(f"{a}->{b}: rd={rd:.2e} frac={fd:.3f}")
            print(f"{key}\n    " + " | ".join(cols) + "\n    consec: " + " | ".join(consec))
        except Exception as e:  # noqa: BLE001
            print(f"{key}: FAILED: {e}")

    print("\nOPTIMIZER STATE")
    for key in opt_keys:
        try:
            md = metas[ref].state_dict_metadata[key]
            tensors = {
                n: load_one(CKPTS[n], key, md.size, md.properties.dtype)
                for n in NAMES[1:]
                if n in metas and key in metas[n].state_dict_metadata
            }
            run_names = [n for n in NAMES[1:] if n in tensors]
            consec = []
            for a, b in zip(run_names, run_names[1:]):
                rd, fd = stats_vs(tensors[b], tensors[a])
                consec.append(f"{a}->{b}: rd={rd:.3e} frac={fd:.3f}")
            print(f"{key}\n    " + " | ".join(consec))
        except Exception as e:  # noqa: BLE001
            print(f"{key}: FAILED: {e}")


if __name__ == "__main__":
    sys.exit(main())
