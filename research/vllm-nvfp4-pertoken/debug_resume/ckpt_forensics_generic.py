# Generic cross-checkpoint tensor staleness check.
# Usage: python ckpt_forensics_generic.py <run_results_dir> <step> <step> [<step> ...]
# PASS iff every consecutive checkpoint pair differs in model weights AND exp_avg.
import os
import sys

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader

MODEL_KEY_PATTERNS = [
    ["embedding.word_embeddings.weight"],
    ["output_layer.weight"],
    ["linear_qkv.weight"],
    ["linear_proj.weight"],
    ["experts", "linear_fc1.weight"],
]


def init_single_rank():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29514")
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)


def load_one(path, key, size, dtype):
    sd = {key: torch.empty(size, dtype=dtype)}
    dcp.load(sd, storage_reader=FileSystemReader(path))
    return sd[key]


def main():
    run_dir = sys.argv[1]
    steps = [int(s) for s in sys.argv[2:]]
    names = [f"step_{s}" for s in steps]
    paths = {n: os.path.join(run_dir, n, "policy", "weights", "iter_0000000") for n in names}

    init_single_rank()
    torch.set_grad_enabled(False)

    metas = {n: FileSystemReader(p).read_metadata() for n, p in paths.items()}
    ref = names[-1]
    keys = sorted(metas[ref].state_dict_metadata.keys())

    sample = []
    for pats in MODEL_KEY_PATTERNS:
        for k in keys:
            md = metas[ref].state_dict_metadata[k]
            if all(p in k for p in pats) and hasattr(md, "size"):
                sample.append(k)
                break
    sample += [k for k in keys if "optimizer" in k and k.endswith(".exp_avg")][:2]

    n_frozen_pairs = 0
    for key in sample:
        md = metas[ref].state_dict_metadata[key]
        tensors = {}
        for n in names:
            if key in metas[n].state_dict_metadata:
                tensors[n] = load_one(paths[n], key, md.size, md.properties.dtype).float()
        cols = []
        for a, b in zip(names, names[1:]):
            if a in tensors and b in tensors:
                frac = (tensors[a] != tensors[b]).float().mean().item()
                rd = (tensors[b] - tensors[a]).norm().item() / (tensors[a].norm().item() + 1e-12)
                frozen = frac == 0.0
                n_frozen_pairs += int(frozen)
                cols.append(f"{a}->{b}: rd={rd:.2e} frac={frac:.4f}{' FROZEN' if frozen else ''}")
        print(f"{key}\n    " + " | ".join(cols))

    if n_frozen_pairs:
        print(f"\nRESULT: FAIL — {n_frozen_pairs} frozen consecutive tensor pair(s); stale-save bug still present")
        return 1
    print("\nRESULT: PASS — all sampled tensors (weights + exp_avg) evolve between consecutive checkpoints")
    return 0


if __name__ == "__main__":
    sys.exit(main())
