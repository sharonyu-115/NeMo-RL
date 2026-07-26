import math
import wandb

PROJECT = "nv-welcome/qwen3-30b-nvfp4"
RUNS = {
    "r2 (row-scaled fwd, old TE)": "grpo-qwen3-30ba3b-base-8n4g-dapo512-20k-nvfp4-pertoken-r2-20260721",
    "fp4fwd (full per-token, PR#3045)": "grpo-qwen3-30ba3b-base-8n4g-dapo512-20k-nvfp4-pertoken-fp4fwd",
}
# Metrics that characterise forward numerical fidelity + convergence/stability.
METRICS = [
    "train/gen_kl_error", "train/policy_kl_error", "train/js_divergence_error",
    "train/token_mult_prob_error", "train/max_seq_mult_prob_error",
    "train/num_masked_seqs_by_logprob_error",
    "train/probs_ratio", "train/probs_ratio_max", "train/probs_ratio_min",
    "train/sampling_importance_ratio",
    "train/reward", "train/mean_total_reward", "train/loss",
    "train/approx_entropy", "train/grad_norm", "train/lr",
]


def is_nan_like(v):
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return str(v) in ("NaN", "nan")


def clean(series):
    out = []
    for v in series:
        if is_nan_like(v):
            continue
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            pass
    return out


api = wandb.Api(timeout=60)
data = {}
for label, name in RUNS.items():
    runs = list(api.runs(PROJECT, filters={"display_name": name}))
    if not runs:
        print(f"!! no run found for {name}")
        continue
    run = sorted(runs, key=lambda r: r.created_at)[-1]  # newest if dupes
    h = run.history(samples=100000, pandas=True)
    cols = [c for c in METRICS if c in h.columns]
    data[label] = {"run": run, "h": h, "cols": cols}
    print(f"== {label}\n   id={run.id} state={run.state} rows={len(h)} name={name}")

# Restrict every run to the COMMON step range so summaries are apples-to-apples.
def maxstep(h):
    return int(h["_step"].max()) if "_step" in h else int(h.index.max())
overlap_max = min(maxstep(d["h"]) for d in data.values())
for d in data.values():
    h = d["h"]
    d["ho"] = h[h["_step"] <= overlap_max] if "_step" in h else h.loc[h.index <= overlap_max]
print(f"\n================ PER-RUN SUMMARY over COMMON steps 0..{overlap_max} "
      "(metric: n | mean | std | min | max | last) ================")
allcols = sorted({c for d in data.values() for c in d["cols"]})
for m in allcols:
    print(f"\n--- {m} ---")
    for label, d in data.items():
        if m not in d["cols"]:
            print(f"   {label:38s}  (absent)"); continue
        vals = clean(d["ho"][m].tolist())
        if not vals:
            print(f"   {label:38s}  (no numeric)"); continue
        n = len(vals); mean = sum(vals)/n
        std = (sum((x-mean)**2 for x in vals)/n) ** 0.5
        print(f"   {label:38s}  n={n:4d}  mean={mean:+.5g}  std={std:.4g}  min={min(vals):+.5g}  max={max(vals):+.5g}  last={vals[-1]:+.5g}")

# Step-aligned overlap comparison on the key fidelity metrics.
print("\n================ STEP-ALIGNED (overlap) — forward-fidelity metrics ================")
key = [m for m in ["train/gen_kl_error", "train/token_mult_prob_error",
                   "train/js_divergence_error", "train/reward",
                   "train/grad_norm"] if any(m in d["cols"] for d in data.values())]
labels = list(data.keys())
if len(labels) == 2:
    a, b = labels
    ha, hb = data[a]["h"], data[b]["h"]
    steps_a = set(int(s) for s in ha["_step"].tolist()) if "_step" in ha else set(ha.index)
    steps_b = set(int(s) for s in hb["_step"].tolist()) if "_step" in hb else set(hb.index)
    common = sorted(steps_a & steps_b)
    print(f"overlap steps: {len(common)} (0..{max(common) if common else 0})")
    def val_at(h, m, step):
        try:
            row = h[h["_step"] == step] if "_step" in h else h.loc[[step]]
            if row.empty or m not in h.columns:
                return None
            v = row[m].iloc[0]
            return None if is_nan_like(v) else float(v)
        except Exception:
            return None
    for m in key:
        print(f"\n--- {m}  ({a}  vs  {b}) ---")
        pts = [s for s in common if s % 5 == 0][:40]
        for s in pts:
            va, vb = val_at(ha, m, s), val_at(hb, m, s)
            if va is None and vb is None:
                continue
            fa = f"{va:+.5g}" if va is not None else "  --  "
            fb = f"{vb:+.5g}" if vb is not None else "  --  "
            d = f"{vb-va:+.4g}" if (va is not None and vb is not None) else ""
            print(f"   step {s:4d}:  {fa:>12s}   {fb:>12s}   Δ={d}")
print("\nDONE")
