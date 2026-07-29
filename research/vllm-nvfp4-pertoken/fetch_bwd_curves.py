"""Fetch reward / gen_kl curves for the NVFP4 backward campaign legs.

Covers the full {weight geometry} x {router replay} factorial plus the fp4fwd
(dequantized-backward) reference. Prints per-leg windowed means, a step-aligned
cross-leg table, and a peak/divergence summary.

Run via srun inside a container (wandb is not installed on the login node):
  srun ... bash -c "export WANDB_API_KEY=... && python research/vllm-nvfp4-pertoken/fetch_bwd_curves.py"
"""

import math

import wandb

ENTITY = "nv-welcome"
PROJECT = "qwen3-30b-nvfp4"

# label -> (run id, weight geometry, router replay)
RUNS = {
    "fp4bwd": ("78e43bb48a7353767342a7bc02ca7976", "vector/1x16", "off"),
    "fp4bwd-r3": ("212ddfeea37498222932f9ada325009d", "vector/1x16", "ON"),
    "fp4bwd-w2d": ("c3948fa7b703dba4333aaa10c3f95933", "scalar/16x16", "off"),
    "fp4bwd-w2d-r3": ("bcfe8a6a9029470a9b8779db65ef559f", "scalar/16x16", "ON"),
    "fp4bwd-w1d": ("eda77a36db894eaea44060e0b7318819", "scalar/1x16", "off"),
    "fp4bwd-w1d-r3": ("47ea0ed67c45233a2f6afa83146c8318", "scalar/1x16", "ON"),
    # Reference: per-token forward, DEQUANTIZED backward (no FP4 bwd at all).
    "fp4fwd (ref)": ("84115c24b1546f369462e60c898ce77f", "n/a (bf16 bwd)", "off"),
}

REWARD_KEYS = ("train/reward", "train/mean_total_reward")
KL_KEYS = ("train/gen_kl_error",)
LEN_KEYS = ("train/mean_gen_tokens_per_sample",)  # NOT max_gen_tokens_per_sample
# Validation is the task metric; train/reward is shaped (overlong-buffer penalty +
# rescaling to [-1,1]) so it is only a proxy. Logged every grpo.val_period steps, so
# these rows are sparse relative to the training rows — never filter history by a key
# list that mixes the two, or every training-only row is dropped.
VAL_KEYS = ("validation/accuracy", "validation/avg_accuracy", "val/accuracy")
VALLEN_KEYS = ("validation/avg_length", "validation/mean_gen_tokens")

# Step checkpoints for the aligned cross-leg comparison.
MARKS = (50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700)


def is_nan_like(v):
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return str(v) in ("NaN", "nan")


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def fmt(v, w=8, p=3):
    return f"{v:>{w}.{p}f}" if isinstance(v, float) and not math.isnan(v) else f"{'--':>{w}}"


api = wandb.Api(timeout=120)
series = {}  # label -> {step: (reward, kl, genlen)}
val = {}  # label -> {step: (accuracy, avg_length)}
meta = {}

for label, (rid, geom, replay) in RUNS.items():
    try:
        run = api.run(f"{ENTITY}/{PROJECT}/{rid}")
        rows = [r for r in run.history(samples=20000, pandas=False) if r.get("_step") is not None]
    except Exception as e:  # noqa: BLE001 - diagnostic script
        print(f"!! {label}: {type(e).__name__}: {str(e)[:80]}")
        continue
    rk = next((k for k in REWARD_KEYS if any(k in r for r in rows)), None)
    kk = next((k for k in KL_KEYS if any(k in r for r in rows)), None)
    lk = next((k for k in LEN_KEYS if any(k in r for r in rows)), None)
    vk = next((k for k in VAL_KEYS if any(k in r for r in rows)), None)
    vlk = next((k for k in VALLEN_KEYS if any(k in r for r in rows)), None)
    d = {}
    for r in rows:
        v = r.get(rk)
        if not is_nan_like(v):
            d[int(r["_step"])] = (v, r.get(kk), r.get(lk))
    vd = {}
    for r in rows:
        a = r.get(vk) if vk else None
        if not is_nan_like(a):
            vd[int(r["_step"])] = (a, r.get(vlk) if vlk else None)
    series[label] = d
    val[label] = vd
    meta[label] = (run.state, geom, replay, max(d) if d else 0, vk)

print("=" * 100)
print("PER-LEG TRAJECTORY (windowed means of train/reward, train/gen_kl_error)")
print("=" * 100)
for label, d in series.items():
    st, geom, replay, last, vk = meta[label]
    print(f"\n### {label}   [{geom}, replay={replay}]   state={st}  steps=1..{last}  n={len(d)}")
    if not d:
        print("   (no reward points)")
        continue
    lo, hi = min(d), max(d)
    nbins = min(12, max(2, (hi - lo + 1) // 25))
    edges = [lo + i * (hi - lo + 1) // nbins for i in range(nbins + 1)]
    print(f"   {'window':>14} {'reward':>9} {'gen_kl':>10} {'genlen':>9} {'n':>4}")
    for a, b in zip(edges[:-1], edges[1:]):
        w = [d[s] for s in d if a <= s < b]
        if not w:
            continue
        print(
            f"   {f'{a}-{b - 1}':>14}"
            f" {mean([x[0] for x in w if not is_nan_like(x[0])]):>9.4f}"
            f" {mean([x[1] for x in w if not is_nan_like(x[1])]):>10.5f}"
            f" {mean([x[2] for x in w if not is_nan_like(x[2])]):>9.0f}"
            f" {len(w):>4}"
        )
    rs = [(s, d[s][0]) for s in sorted(d)]
    pk = max(rs, key=lambda t: t[1])
    print(f"   peak reward {pk[1]:+.4f} @ step {pk[0]};  last {rs[-1][1]:+.4f} @ step {rs[-1][0]}")

print("\n" + "=" * 100)
print("STEP-ALIGNED COMPARISON — train/reward (mean over +/-10 steps around each mark)")
print("=" * 100)
hdr = f"{'leg':<16}{'geom':<15}{'rep':<5}" + "".join(f"{m:>9}" for m in MARKS)
print(hdr)
print("-" * len(hdr))
for label, d in series.items():
    _, geom, replay, _, _ = meta[label]
    row = f"{label:<16}{geom:<15}{replay:<5}"
    for m in MARKS:
        w = [d[s][0] for s in d if abs(s - m) <= 10 and not is_nan_like(d[s][0])]
        row += fmt(mean(w) if w else float("nan"), 9, 3)
    print(row)

print("\n" + "=" * 100)
print("STEP-ALIGNED COMPARISON — train/gen_kl_error")
print("=" * 100)
print(hdr)
print("-" * len(hdr))
for label, d in series.items():
    _, geom, replay, _, _ = meta[label]
    row = f"{label:<16}{geom:<15}{replay:<5}"
    for m in MARKS:
        w = [d[s][1] for s in d if abs(s - m) <= 10 and not is_nan_like(d[s][1])]
        row += fmt(mean(w) if w else float("nan"), 9, 5)
    print(row)

print("\n" + "=" * 100)
print("DIVERGENCE SCAN — on 25-step rolling means (single steps are far too noisy)")
print("=" * 100)
# An earlier version ran this over raw single-step rewards against a running max.
# Step-to-step reward swings by ~0.3, so it flagged step ~55 for nearly every leg —
# during the initial climb out of -0.85, i.e. the opposite of divergence. Smooth first.
WIN = 25
for label, d in series.items():
    if not d:
        continue
    steps = sorted(d)
    sm = []  # (step, rolling-mean reward, rolling-mean gen_kl)
    for i, s in enumerate(steps):
        w = [d[t] for t in steps[max(0, i - WIN + 1) : i + 1]]
        r = [x[0] for x in w if not is_nan_like(x[0])]
        k = [x[1] for x in w if not is_nan_like(x[1])]
        if r:
            sm.append((s, mean(r), mean(k) if k else float("nan")))
    if len(sm) < WIN:
        print(f"{label:<16} too few points to smooth ({len(sm)})")
        continue
    pk = max(sm, key=lambda t: t[1])
    div = next((s for s, r, _ in sm if s > pk[0] and pk[1] - r > 0.25), None)
    kl_pk = next((k for s, _, k in sm if s == pk[0]), float("nan"))
    kl_end = sm[-1][2]
    print(
        f"{label:<16} smoothed peak {pk[1]:+.4f} @ {pk[0]:<5}"
        f" decay>0.25 @ {str(div) if div else 'not yet':<8}"
        f" gen_kl peak->last {kl_pk:.5f} -> {kl_end:.5f}"
    )


print("\n" + "=" * 100)
print("VALIDATION ACCURACY (task metric; train/reward is shaped and only a proxy)")
print("=" * 100)
for label, vd in val.items():
    _, geom, replay, _, vk = meta[label]
    if not vd:
        print(f"\n### {label}: no validation points (key searched: {VAL_KEYS})")
        continue
    steps = sorted(vd)
    best = max(steps, key=lambda s: vd[s][0])
    print(f"\n### {label}   [{geom}, replay={replay}]   key={vk}  n={len(vd)}")
    print("   " + "  ".join(f"{s}:{vd[s][0]:.3f}" for s in steps))
    print(f"   best {vd[best][0]:.4f} @ step {best};  last {vd[steps[-1]][0]:.4f} @ step {steps[-1]}")

print("\n" + "=" * 100)
print("STEP-ALIGNED VALIDATION ACCURACY (nearest val point within +/-15 steps)")
print("=" * 100)
vhdr = f"{'leg':<16}{'geom':<15}{'rep':<5}" + "".join(f"{m:>9}" for m in MARKS) + f"{'best':>9}"
print(vhdr); print("-" * len(vhdr))
for label, vd in val.items():
    _, geom, replay, _, _ = meta[label]
    row = f"{label:<16}{geom:<15}{replay:<5}"
    for m in MARKS:
        cand = [s for s in vd if abs(s - m) <= 15]
        row += fmt(vd[min(cand, key=lambda s: abs(s - m))][0] if cand else float("nan"), 9, 3)
    row += fmt(max((v[0] for v in vd.values()), default=float("nan")), 9, 3)
    print(row)
