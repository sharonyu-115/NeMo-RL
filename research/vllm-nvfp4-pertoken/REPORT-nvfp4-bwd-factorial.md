# NVFP4 per-token backward: weight quantization x router replay

Readout of six training legs plus one reference. Data pulled 2026-07-28 from W&B
`nv-welcome/qwen3-30b-nvfp4` via `research/vllm-nvfp4-pertoken/fetch_bwd_curves.py`
(run ids in that script).

**Primary metric is `validation/accuracy`** — DAPOMathAIME2024, 256 samples, every 10
steps. `train/reward` is shown as a secondary signal but is *shaped* (overlong-buffer
penalty, rescaling to [-1,1]) and compresses the real differences between legs; where
the two disagree, accuracy is the one to believe.

Several legs are still running and none has plateaued, so this is an interim readout —
findings and hypotheses only, no conclusions.

---

## 1. Experiment design

All legs: Qwen3-30B-A3B-Base, DAPO-512 / 20k response, 8 nodes x 4 GB200, TP2/EP8,
identical data and seed. Real NVFP4 per-token **forward and backward** on the MoE expert
GEMMs (attention and the first-2/last-4 layers stay BF16). Activations and gradients are
quantized identically in every leg: per-token outer scale + 1x16 e4m3 inner blocks. Only
the two dials below vary.

**Dial A — weight quantization.** An NVFP4 value is `4-bit code x inner block scale x
outer scale`. For weights, both levels are set independently:

| geometry | outer scale | inner block |
|---|---|---|
| `vector/1x16` | per-row `(M,)` + per-col `(K,)` vectors | 16 elements |
| `scalar/1x16` | one scalar for the whole tensor | 16 elements |
| `scalar/16x16` | one scalar for the whole tensor | 16x16 tile |

They form a chain in which each step changes exactly one thing:

```
vector/1x16  --[outer: vector -> scalar]-->  scalar/1x16  --[inner: 1x16 -> 16x16]-->  scalar/16x16
```

**Dial B — router replay.** Off, or on (vLLM returns its generation-time MoE routing and
the training forward replays it, removing routing nondeterminism between generation and
training).

3 geometries x 2 replay settings = six legs. A seventh run, `fp4fwd`, sits **outside the
factorial as a reference**: same per-token forward, but a dequantized (BF16) backward —
no FP4 backward at all.

---

## 2. The curves

![NVFP4 backward factorial — five metrics across seven legs](nvfp4-bwd-factorial.png)

*Hue = weight geometry; dashed = router replay off; gray = the BF16-backward reference.
Per-step metrics use a 15-step rolling mean; validation accuracy is raw. Regenerate with
`plot_bwd_curves.py`.*

### 2.1 Validation accuracy (primary)

Nearest validation point within +/-15 steps of each mark.

| leg | geometry | replay | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | best |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `fp4bwd` | vector/1x16 | off | 0.012 | 0.020 | 0.012 | 0.023 | 0.012 | 0.027 | 0.031 | — | 0.047 |
| `fp4bwd-r3` | vector/1x16 | ON | 0.016 | 0.031 | 0.125 | 0.266 | 0.273 | 0.348 | 0.328 | 0.355 | 0.402 |
| `fp4bwd-w1d` | scalar/1x16 | off | 0.066 | 0.137 | 0.160 | 0.152 | 0.195 | 0.195 | — | — | 0.230 |
| `fp4bwd-w1d-r3` | scalar/1x16 | ON | 0.113 | 0.133 | 0.156 | 0.172 | 0.188 | 0.211 | — | — | **0.316** |
| `fp4bwd-w2d` | scalar/16x16 | off | 0.113 | 0.137 | 0.270 | 0.301 | 0.352 | **0.367** | 0.359 | — | 0.402 |
| `fp4bwd-w2d-r3` | scalar/16x16 | ON | 0.082 | 0.148 | 0.219 | 0.305 | 0.324 | — | — | — | 0.340 |
| `fp4fwd` *(ref)* | BF16 bwd | off | 0.094 | 0.223 | 0.266 | 0.336 | 0.355 | 0.359 | 0.371 | 0.367 | 0.402 |

Read the step-300 column as a snapshot of legs mid-climb, not as their level:
`w1d-r3` went 0.211 @300 -> 0.316 @330 and is still rising.

### 2.2 Mean generated tokens per sample

`train/mean_gen_tokens_per_sample`, windowed.

| leg | ~100 | ~200 | ~300 | ~340 |
|---|---|---|---|---|
| `fp4bwd` | 863 | 824 | 816 | 850 |
| `fp4bwd-r3` | 744 | 3793 | 4268 | 5642 |
| `fp4bwd-w1d` | 1050 | 1100 | 1232 | 1405 |
| `fp4bwd-w1d-r3` | 1037 | 1145 | 1871 | **3510** |
| `fp4bwd-w2d` | 1211 | 4567 | 6098 | 6764 |
| `fp4bwd-w2d-r3` | 1060 | 4524 | 5297 | — |
| `fp4fwd` *(ref)* | 2320 | 4413 | 6320 | 7452 |

### 2.3 Generation KL error

| leg | replay | 50 | 100 | 150 | 200 | 250 | 300 | 350 |
|---|---|---|---|---|---|---|---|---|
| `fp4bwd` | off | 0.0102 | 0.0104 | 0.0100 | 0.0097 | 0.0097 | 0.0084 | 0.0075 |
| `fp4bwd-r3` | ON | 0.0084 | 0.0077 | 0.0039 | 0.0044 | 0.0047 | 0.0049 | 0.0050 |
| `fp4bwd-w1d` | off | 0.0046 | 0.0032 | 0.0035 | 0.0042 | 0.0053 | **0.0181** | — |
| `fp4bwd-w1d-r3` | ON | 0.0031 | 0.0027 | 0.0029 | 0.0031 | 0.0037 | 0.0043 | — |
| `fp4bwd-w2d` | off | 0.0083 | 0.0066 | 0.0074 | 0.0088 | 0.0095 | 0.0103 | **0.0140** |
| `fp4bwd-w2d-r3` | ON | 0.0065 | 0.0058 | 0.0061 | 0.0079 | 0.0081 | — | — |
| `fp4fwd` *(ref)* | off | 0.0056 | 0.0050 | 0.0053 | 0.0059 | 0.0065 | 0.0071 | 0.0080 |

### 2.4 Train reward (secondary, shaped)

| leg | 50 | 100 | 150 | 200 | 250 | 300 |
|---|---|---|---|---|---|---|
| `fp4bwd` | -0.896 | -0.856 | -0.847 | -0.838 | -0.818 | -0.753 |
| `fp4bwd-r3` | -0.863 | -0.749 | -0.307 | -0.059 | 0.044 | 0.127 |
| `fp4bwd-w1d` | -0.528 | -0.188 | -0.158 | -0.100 | -0.065 | 0.084 |
| `fp4bwd-w1d-r3` | -0.465 | -0.179 | -0.147 | -0.098 | -0.059 | 0.082 |
| `fp4bwd-w2d` | -0.484 | -0.202 | -0.033 | 0.050 | 0.061 | 0.148 |
| `fp4bwd-w2d-r3` | -0.474 | -0.202 | -0.128 | 0.067 | 0.061 | — |
| `fp4fwd` *(ref)* | -0.490 | -0.111 | 0.052 | 0.060 | 0.081 | 0.133 |

Note the compression: `w1d` and `w2d` differ by 0.064 reward at step 300 (0.084 vs
0.148) where their accuracies differ by 0.172 (0.195 vs 0.367). And `fp4bwd`'s reward
climbs steadily, -0.90 -> -0.75, while its accuracy never leaves the noise floor. Reward
credits format and length compliance that accuracy does not.

### 2.5 Where each leg stands

| leg | best val acc | at step | last step | state |
|---|---|---|---|---|
| `fp4bwd` | 0.047 | 310 | 346 | cancelled |
| `fp4bwd-r3` | 0.402 | 630 | 763 | cancelled |
| `fp4bwd-w1d` | 0.230 | 310 | 320 | cancelled |
| `fp4bwd-w1d-r3` | 0.316 | 330 | 339 | **running** |
| `fp4bwd-w2d` | 0.402 | 330 | 367 | cancelled |
| `fp4bwd-w2d-r3` | 0.340 | 190 | 267 | **running** |
| `fp4fwd` *(ref)* | 0.402 | 410 | 467 | stalled |

---

## 3. Findings and hypotheses

### F1 — The weight OUTER scale decides whether the run learns; the INNER block decides how fast

Walking the single-dial chain. Both comparisons isolate one thing: `fp4bwd` vs `w1d`
share 1x16 inner blocks and differ only in the outer scale; `w1d` vs `w2d` share the
scalar outer and differ only in the inner block.

| geometry | val acc @300 | best so far | reads as |
|---|---|---|---|
| `vector/1x16` (`fp4bwd`) | 0.027 | 0.047 @310 | never learns |
| `scalar/1x16` (`w1d-r3`) | 0.211 | 0.316 @330, still rising | learns, **late** |
| `scalar/16x16` (`w2d`) | 0.367 | 0.402 @330 | learns, on time |

The two steps differ in kind:

- **Outer, vector -> scalar: qualitative.** `vector/1x16` sits at 0.012-0.031 accuracy
  for its entire 346-step run against an untrained baseline of 0.004. It does not learn
  the task, and its reward trend (-0.90 -> -0.75) reads as slow learning only because
  reward is shaped.
- **Inner, 1x16 -> 16x16: a delay, not a demonstrated ceiling.** At step 300 the gap
  looked like ~2x (0.195 vs 0.367), but `w1d-r3` then went 0.211 -> 0.316 in 30 steps and
  has not plateaued. Whether the asymptotes differ is unresolved — no `scalar/1x16` leg
  has been run to a plateau.

### F2 — Accuracy is downstream of generation length; the geometries differ in when length takes off

On AIME the policy has to learn to reason at length, and that is where the geometries
separate (2.2). `fp4bwd` never grows (~850 tokens, flat). The `scalar/1x16` legs sit at
~1000-1200 for 300 steps, then break upward. The `scalar/16x16` legs and the reference
grow from ~step 150.

Accuracy tracks this almost one-for-one: `w1d-r3`'s accuracy jump (0.211 -> 0.316 over
steps 300-330) coincides exactly with its length jump (1871 -> 3510). The legs that grow
length by step 150 are the legs at ~0.35 accuracy by step 300.

`approx_entropy` shows the same ordering from the other side — every leg except `fp4bwd`
collapses from ~1.0 to ~0.1 within ~150 steps; `fp4bwd` is still at ~0.5 when cancelled,
never committing to a policy.

So the weight geometry's effect on accuracy is mediated: it changes *when* the policy
starts producing long reasoning chains.

### H1 — The damage scales with how much the forward and backward weights disagree

*(explains F1, F2)*

TE quantizes each weight twice from the same BF16 source: rowwise (consumed by the
forward, `Y = X·W`) and columnwise (consumed by dgrad, `dX = dY·Wᵀ`). If the two
disagree, the backward computes the gradient of a different function than the forward
evaluated. That disagreement is a fixed function of `W`, so it is a systematic bias, not
noise.

How much they disagree depends on which levels are direction-dependent:

| geometry | outer, fwd vs bwd | inner, fwd vs bwd | outcome |
|---|---|---|---|
| `vector/1x16` | **differs** — `(M,)` vs `(K,)` | **differs** | never learns |
| `scalar/1x16` | identical | **differs** | learns, late |
| `scalar/16x16` | identical | identical — a 16x16 tile is the same 256 numbers either way | learns, on time |

Monotonic in the number of direction-dependent levels, and the outer level costs far
more than the inner.

**Test.** Measure the *magnitude* of disagreement rather than the fraction of differing
elements: `||W_row - W_colᵀ|| / ||W||` per geometry on real checkpoint weights. H1
predicts `vector/1x16 > scalar/1x16 > scalar/16x16 = 0`, with a large first gap and a
small second. `verify_w1d_weight_legs.py` already performs the casts and reports
element-mismatch fraction (0.00% for `scalar/16x16`, ~76-77% for `scalar/1x16`); it needs
a relative-norm metric and a `vector/1x16` case. One node, no training.

### F3 — Router replay rescues a bad quantizer, does nothing for a good one, and suppresses late drift

Replay's effect on accuracy scales inversely with how good the weight geometry is:

| geometry | replay off | replay ON | delta |
|---|---|---|---|
| `vector/1x16` | 0.027 | 0.348 | **+0.321** |
| `scalar/1x16` | 0.195 | 0.211 | +0.016 |
| `scalar/16x16` | 0.352 *(@250)* | 0.324 *(@250)* | ~0 |

So replay is a **compensating** mechanism, not an independent gain.

Separately, it holds gen_kl flat where the replay-off legs drift:

| leg | 150 | 250 | 300 | 350 |
|---|---|---|---|---|
| `w1d` (off) | 0.0035 | 0.0053 | **0.0181** | — |
| `w1d-r3` (ON) | 0.0029 | 0.0037 | **0.0043** | — |
| `w2d` (off) | 0.0074 | 0.0095 | 0.0103 | **0.0140** |
| `w2d-r3` (ON) | 0.0061 | 0.0081 | — | — |

~4x apart at step 300 with replay the only dial; both replay-off legs were cancelled
during that rise. The `w2d`/`w2d-r3` accuracy comparison is the weakest cell here —
0.352 vs 0.324 at step 250, within the noise of this metric, with only 27 validation
points on `w2d-r3`.

### H2 — Replay removes a routing-flip channel that a bad weight quantizer amplifies

*(explains F3)*

Replay makes the training forward reuse generation's MoE routing. Its benefit tracks how
bad the quantizer is. MoE routing is a top-k over logits, so small numerical
perturbations can reorder it, and a weight quantizer whose forward and backward disagree
perturbs exactly those logits. Replay makes the training pass agree with generation by
construction, however large the perturbation.

**Test.** Log the per-step fraction of tokens whose top-k expert set differs between
generation and training. H2 predicts this rises with the H1 disagreement metric across
geometries, and tracks gen_kl within a leg.

### F4 — gen_kl measures agreement with the ROLLOUT's weight cast, and ranks the legs backwards

The vLLM rollout quantizes expert weights at refit with a **per-tensor scalar outer scale
plus 1x16 e4m3 inner blocks** — `weight_scale_2 = amax / (6*448)` per expert, and
`_quantize_blocks` reshaping the last dim into `(k//16, 16)` with
`block_scale = block_amax / 6`
(`nemo_rl/models/generation/vllm/quantization/nvfp4_pertoken.py`). TE computes its
per-tensor global scale by the identical formula,
`global_encode_scale = fp8_max * fp4_max / global_amax` = `448*6/amax`
(`common/cast/nvfp4/core_nvfp4.cuh:87`).

That is exactly the `scalar/1x16` geometry, so one leg quantizes its forward weight the
way the rollout does and the others do not:

| leg | training forward (rowwise) cast | vs rollout `scalar/1x16` | gen_kl @250 |
|---|---|---|---|
| `w1d` | scalar outer + 1x16 | **matches at both levels** | **0.0053** |
| `w2d` | scalar outer + 16x16 | outer matches, inner differs | 0.0095 |
| `fp4bwd` | per-row `(M,)` outer + 1x16 | inner matches, **outer differs** | 0.0097 |

gen_kl orders exactly by that alignment, with outer-level mismatch costing more than
inner-level. The consequence is that **gen_kl mis-ranks the legs**: `w1d` has the best
train/generation agreement of any FP4 leg and the slowest learning of the two scalar
geometries. Minimising gen_kl selects the wrong configuration.

*Basis: geometry and scale-formula agreement read from source, not a measured
bit-comparison of the two casts.*

### H3 — Two alignment axes; a leg needs the outer scale right on both

*(unifies F1 and F4)*

Two independent things can agree or disagree, and no available geometry wins both:

| leg | **A: train-fwd ↔ train-bwd** (gradient) | **B: train-fwd ↔ rollout** (importance ratio) | learns | gen_kl |
|---|---|---|---|---|
| `scalar/16x16` | **full** — tile invariant under transpose | partial — outer matches, inner differs | fastest | 0.0074-0.0095 |
| `scalar/1x16` | partial — outer matches, inner differs | **full** — the rollout *is* scalar + 1x16 | slower, yes | **0.0031-0.0053** |
| `vector/1x16` | **broken at the outer level** | **broken at the outer level** | never | 0.0097-0.0104 |

Each geometry keeps one channel clean, except `vector/1x16`, which loses the outer scale
on **both** — its gradients are biased *and* its importance ratios are computed against a
policy it does not match. That accounts for the discontinuity in F1: `vector/1x16` does
not fail by degree, it fails outright, while `scalar/1x16` is merely late.

The outer level dominates both axes, and the data shows this independently on each: on B,
`fp4bwd` (outer mismatch, inner match) has higher gen_kl than `w2d` (outer match, inner
mismatch), 0.0104 vs 0.0074; on A, `w1d`'s inner-only disagreement costs a delay while
`fp4bwd`'s outer disagreement costs the run.

**Test.** A `vector/16x16` weight — direction-dependent outer, invariant inner — should
still fail to learn if axis A gates learning, despite its invariant inner blocks. That
geometry is not implemented in the TE fork (the reachability matrix lists it as needing a
new cast kernel), so it is not free, but it is the clean discriminator. Failing that, the
H1 norm measurement plus H2's routing-flip counter test the two axes separately.
