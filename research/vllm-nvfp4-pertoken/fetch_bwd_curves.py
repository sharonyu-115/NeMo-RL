"""Fetch reward / gen_kl curves for the NVFP4 backward campaign legs.

Run via srun inside a container (wandb is not installed on the login node):
  srun ... bash -c "export WANDB_API_KEY=... && python research/vllm-nvfp4-pertoken/fetch_bwd_curves.py"
"""

import math

import wandb

ENTITY = "nv-welcome"
PROJECT = "qwen3-30b-nvfp4"

RUNS = {
    "fp4bwd (replay OFF)": "78e43bb48a7353767342a7bc02ca7976",
    "fp4bwd-r3 (replay ON)": "212ddfeea37498222932f9ada325009d",
    "fp4bwd-w2d (replay OFF)": "c3948fa7b703dba4333aaa10c3f95933",
    "fp4fwd (replay OFF)": "84115c24b1546f369462e60c898ce77f",
}

# Reward key varies by config; take the first that is present.
REWARD_KEYS = ("train/mean_total_reward", "train/reward")
KL_KEYS = ("train/gen_kl_error", "train/policy_kl_error")


def is_nan_like(val):
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    return str(val) in ("NaN", "nan")


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


api = wandb.Api(timeout=120)

for label, run_id in RUNS.items():
    print("=" * 78)
    print(label, run_id)
    try:
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    except Exception as e:  # noqa: BLE001 - diagnostic script
        print(f"  FETCH FAILED: {type(e).__name__}: {e}")
        continue

    hist = run.history(samples=10000, pandas=False)
    rows = [r for r in hist if r.get("_step") is not None]
    if not rows:
        print("  no history rows")
        continue

    rk = next((k for k in REWARD_KEYS if any(k in r for r in rows)), None)
    kk = next((k for k in KL_KEYS if any(k in r for r in rows)), None)
    print(f"  state={run.state} rows={len(rows)} reward_key={rk} kl_key={kk}")

    pts = [
        (r["_step"], r.get(rk), r.get(kk))
        for r in rows
        if rk and not is_nan_like(r.get(rk))
    ]
    if not pts:
        print("  no reward points")
        continue

    print(f"  step range: {pts[0][0]} .. {pts[-1][0]}  ({len(pts)} reward points)")
    # Windowed means: coarse trajectory without dumping every step.
    span = pts[-1][0] - pts[0][0] + 1
    nbins = min(12, max(2, span // 25))
    edges = [pts[0][0] + i * span // nbins for i in range(nbins + 1)]
    print(f"  {'step window':>16} {'reward':>10} {'kl':>12} {'n':>5}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        w = [p for p in pts if lo <= p[0] < hi]
        if not w:
            continue
        rs = [p[1] for p in w if not is_nan_like(p[1])]
        ks = [p[2] for p in w if not is_nan_like(p[2])]
        print(
            f"  {str(lo) + '-' + str(hi - 1):>16} {mean(rs):>10.4f} "
            f"{(mean(ks) if ks else float('nan')):>12.6f} {len(w):>5}"
        )

    first = [p[1] for p in pts[:20] if not is_nan_like(p[1])]
    last = [p[1] for p in pts[-20:] if not is_nan_like(p[1])]
    print(f"  first20 reward={mean(first):.4f}  last20 reward={mean(last):.4f}")
