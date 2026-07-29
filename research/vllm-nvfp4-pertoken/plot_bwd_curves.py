"""Render the NVFP4 backward factorial as a 5-panel small-multiple figure.

Encoding is composite so that identity is never colour-alone and the palette stays
inside the three categorical slots that validate on the all-pairs pairlist (small
multiples are the all-pairs case):

    hue       = weight geometry   (3 slots: blue / orange / aqua)
    linestyle = router replay     (solid = ON, dashed = off)
    gray      = fp4fwd reference  (not a factorial cell — a baseline)

Run via srun inside a container (wandb + matplotlib are not on the login node):
  srun ... bash -c "export WANDB_API_KEY=... && python research/vllm-nvfp4-pertoken/plot_bwd_curves.py"
"""

import math
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import wandb  # noqa: E402

ENTITY = "nv-welcome"
PROJECT = "qwen3-30b-nvfp4"
OUT = "research/vllm-nvfp4-pertoken/nvfp4-bwd-factorial.png"

# Categorical slots 1-3 (light mode) from the reference palette; documented as
# passing the all-pairs CVD and normal-vision floors in both modes.
GEOM_COLOR = {
    "vector/1x16": "#2a78d6",  # slot 1 blue
    "scalar/1x16": "#eb6834",  # slot 2 orange
    "scalar/16x16": "#1baf7a",  # slot 3 aqua
}
REF_COLOR = "#8a8985"  # neutral ink, not a categorical slot
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#dedddaa0"

# label -> (run id, geometry, replay ON?)
RUNS = {
    "fp4bwd": ("78e43bb48a7353767342a7bc02ca7976", "vector/1x16", False),
    "fp4bwd-r3": ("212ddfeea37498222932f9ada325009d", "vector/1x16", True),
    "fp4bwd-w1d": ("eda77a36db894eaea44060e0b7318819", "scalar/1x16", False),
    "fp4bwd-w1d-r3": ("47ea0ed67c45233a2f6afa83146c8318", "scalar/1x16", True),
    "fp4bwd-w2d": ("c3948fa7b703dba4333aaa10c3f95933", "scalar/16x16", False),
    "fp4bwd-w2d-r3": ("bcfe8a6a9029470a9b8779db65ef559f", "scalar/16x16", True),
    "fp4fwd (ref)": ("84115c24b1546f369462e60c898ce77f", None, False),
}

# (panel title, y label, candidate keys, smooth?)
PANELS = [
    ("Validation accuracy  (task metric)", "accuracy", ["validation/accuracy"], False),
    ("Train reward  (shaped)", "reward", ["train/reward", "train/mean_total_reward"], True),
    ("Generation KL error", "gen_kl_error", ["train/gen_kl_error"], True),
    ("Approx. entropy", "approx_entropy", ["train/approx_entropy"], True),
    ("Mean generated tokens / sample", "tokens", ["__GENLEN__"], True),
]
# Discovered at runtime — key naming for generation length varies by version. Must be
# the MEAN, not the max: the run also logs train/max_gen_tokens_per_sample, which sorts
# first alphabetically and would silently plot the wrong quantity.
GENLEN_RE = re.compile(r"^train/mean_gen_tokens", re.I)

WIN = 15  # rolling-mean window for the per-step metrics


def is_nan_like(v):
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return str(v) in ("NaN", "nan")


def smooth(xs, ys, win):
    out = []
    for i in range(len(ys)):
        w = [y for y in ys[max(0, i - win + 1) : i + 1] if not is_nan_like(y)]
        out.append(sum(w) / len(w) if w else float("nan"))
    return xs, out


api = wandb.Api(timeout=120)
data = {}  # label -> {key: (steps, values)}
genlen_key = None

for label, (rid, geom, replay) in RUNS.items():
    run = api.run(f"{ENTITY}/{PROJECT}/{rid}")
    rows = [r for r in run.history(samples=20000, pandas=False) if r.get("_step") is not None]
    if genlen_key is None:
        cands = sorted({k for r in rows for k in r if GENLEN_RE.match(k)})
        if cands:
            genlen_key = cands[0]
            print(f"[keys] generation-length key -> {genlen_key}   (candidates: {cands})")
    per_key = {}
    for _, _, keys, _ in PANELS:
        key = genlen_key if keys == ["__GENLEN__"] else next(
            (k for k in keys if any(k in r for r in rows)), None
        )
        if not key:
            continue
        pts = sorted(
            (int(r["_step"]), r[key]) for r in rows if key in r and not is_nan_like(r.get(key))
        )
        if pts:
            per_key[keys[0]] = ([p[0] for p in pts], [p[1] for p in pts])
    data[label] = per_key
    print(f"[data] {label:16s} " + " ".join(f"{k.split('/')[-1]}:{len(v[0])}" for k, v in per_key.items()))

plt.rcParams.update(
    {
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelcolor": INK2,
        "text.color": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
    }
)
fig, axes = plt.subplots(3, 2, figsize=(12.5, 11.5))
axes = axes.ravel()

for ax, (title, ylab, keys, do_smooth) in zip(axes, PANELS):
    for label, (_, geom, replay) in RUNS.items():
        series = data.get(label, {}).get(keys[0])
        if not series:
            continue
        xs, ys = series
        if do_smooth:
            xs, ys = smooth(xs, ys, WIN)
        color = REF_COLOR if geom is None else GEOM_COLOR[geom]
        ax.plot(
            xs,
            ys,
            lw=2.0,
            color=color,
            linestyle="-" if (replay or geom is None) else (0, (5, 2.5)),
            alpha=0.95 if geom is not None else 0.8,
            marker="o" if not do_smooth else None,
            markersize=3.2 if not do_smooth else 0,
            markevery=3,
            label=label,
        )
    ax.set_title(title, loc="left", color=INK, pad=8)
    ax.set_xlabel("training step")
    ax.set_ylabel(ylab)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c4")

# Legend occupies the spare sixth cell — hue = geometry, dash = replay.
lg = axes[5]
lg.axis("off")
handles = []
for geom, color in GEOM_COLOR.items():
    for replay in (True, False):
        lbl = next(
            (l for l, (_, g, r) in RUNS.items() if g == geom and r == replay), None
        )
        if lbl:
            handles.append(
                plt.Line2D([], [], color=color, lw=2.0,
                           linestyle="-" if replay else (0, (5, 2.5)), label=lbl)
            )
handles.append(plt.Line2D([], [], color=REF_COLOR, lw=2.0, label="fp4fwd (ref, BF16 bwd)"))
lg.legend(handles=handles, loc="upper left", frameon=False, fontsize=9.5,
          title="hue = weight geometry   ·   dashed = replay off",
          title_fontsize=9.5, labelspacing=0.9)
lg.text(0.0, 0.10,
        f"Per-step metrics smoothed with a {WIN}-step rolling mean.\n"
        "Validation accuracy is plotted raw (logged every 10 steps).",
        transform=lg.transAxes, color=INK2, fontsize=8.5, va="top")

fig.suptitle(
    "NVFP4 per-token backward — weight quantization x router replay",
    x=0.008, ha="left", fontsize=13, color=INK, y=0.995,
)
fig.tight_layout(rect=(0, 0, 1, 0.975))
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"wrote {OUT}")
