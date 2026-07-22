# Wandb metric sweep across the step 120 -> 121 resume boundary for the bf16 run.
import math
import os

import wandb

RUN_ID = "e6548363a470d7ec6f14918df57b83b9"
PROJECT = os.environ.get("WANDB_PROJECT", "qwen3-30b-nvfp4")
ENTITY = os.environ.get("WANDB_ENTITY", "")

METRICS = [
    "train/reward",
    "train/mean_total_reward",
    "train/loss",
    "train/grad_norm",
    "train/lr",
    "train/approx_entropy",
    "train/probs_ratio",
    "train/probs_ratio_max",
    "train/probs_ratio_min",
    "train/token_mult_prob_error",
    "train/gen_kl_error",
    "train/sampling_importance_ratio",
    "train/num_masked_seqs_by_logprob_error",
    "train/advantages/mean",
    "validation/accuracy",
]


def is_nan_like(val):
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    return str(val) in ("NaN", "nan")


def fmt(val):
    if is_nan_like(val):
        return "-"
    try:
        return f"{float(val):.5g}"
    except (TypeError, ValueError):
        return "?"


def main():
    api = wandb.Api()
    path = f"{ENTITY}/{PROJECT}/{RUN_ID}" if ENTITY else f"{PROJECT}/{RUN_ID}"
    run = api.run(path)
    print(f"run: {run.name} state={run.state}")
    hist = run.history(samples=10000, pandas=True)
    step_col = "_step" if "_step" in hist.columns else "step"
    hist = hist.sort_values(step_col)

    avail = [m for m in METRICS if m in hist.columns]
    missing = [m for m in METRICS if m not in hist.columns]
    if missing:
        print(f"missing columns: {missing}")

    print("\nBoundary window (steps 100-140):")
    window = hist[(hist[step_col] >= 100) & (hist[step_col] <= 140)]
    print(f"{'step':>5s} " + " ".join(f"{m.split('/')[-1][:16]:>16s}" for m in avail))
    for _, row in window.iterrows():
        if all(is_nan_like(row.get(m)) for m in avail):
            continue
        print(f"{int(row[step_col]):5d} " + " ".join(f"{fmt(row.get(m)):>16s}" for m in avail))

    print("\nReference stats per segment:")
    for lo, hi, label in [(1, 120, "exp_001 (1-120)"), (121, 200, "exp_002 (121+)")]:
        seg = hist[(hist[step_col] >= lo) & (hist[step_col] <= hi)]
        print(f"-- {label}, {len(seg)} rows")
        for m in avail:
            vals = [float(v) for v in seg[m] if not is_nan_like(v)]
            if vals:
                mean = sum(vals) / len(vals)
                print(f"   {m:45s} n={len(vals):4d} mean={mean:.5g} min={min(vals):.5g} max={max(vals):.5g} last={vals[-1]:.5g}")

    nan_hits = []
    for m in avail:
        for _, row in hist.iterrows():
            v = row.get(m)
            if row[step_col] > 0 and v is not None and isinstance(v, float) and math.isnan(v):
                nan_hits.append((int(row[step_col]), m))
    print(f"\nNaN cells (step>0): {nan_hits[:20] if nan_hits else 'none'}")


if __name__ == "__main__":
    main()
