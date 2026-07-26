# Analysis: row-scaled vs full per-token NVFP4 forward — numerics (wandb)

Two **forward-only** per-token NVFP4 runs (both dequant backward, no router replay,
8n4g DAPO-512/20k Qwen3-30B-A3B, experts-only NVFP4 / f2l4), differing ONLY in the
forward quantization implementation:

| Run | Forward cast | TE |
|---|---|---|
| `…-nvfp4-pertoken-r2-20260721` (`054c75d9`) | **row-scaled activation** (`NVTE_NVFP4_ROW_SCALED_ACTIVATION=1`) | old release line |
| `…-nvfp4-pertoken-fp4fwd` (`84115c24`) | **full per-token cast** (`NVTE_NVFP4_PER_TOKEN=1`) | **PR #3045** (fork @690ffea) |

wandb project `nv-welcome/qwen3-30b-nvfp4`. Concept background:
`ROW-SCALED-vs-FULL-PER-TOKEN.md`. Script:
`wandb_cmp_r2_vs_fp4fwd.py` (this dir).

## Methodology note (important)

Per-run summaries MUST be over the **common step range** — r2 ran 501 steps,
fp4fwd 213, so full-history means are NOT comparable (r2's 288 extra steps contain
large spikes that skew mean/max). All numbers below are over **steps 0–213**.
(An earlier full-history read wrongly concluded fp4fwd had a *smaller* outlier tail
and lower variance — both were artifacts of the unmatched ranges; the matched data
reverses them.)

## Results (overlap-matched, steps 0–213)

| Metric (mean unless noted) | r2 row-scaled | fp4fwd full per-token | Verdict |
|---|---|---|---|
| `gen_kl_error` | **0.00444** (std 0.0015) | 0.00606 (std 0.0017) | r2 lower (~37%) |
| `js_divergence_error` | **0.00115** | 0.00157 | r2 lower (~37%) |
| `token_mult_prob_error` (max) | **369** (mean 3.85) | **2.6e7** (mean 1.3e5) | r2 far tamer tail |
| `max_seq_mult_prob_error` (max) | **2.0e5** | **2.6e10** | r2 far tamer |
| `grad_norm` | 0.127 | 0.124 | comparable |
| `approx_entropy` | 0.237 | 0.248 | comparable |
| `sampling_importance_ratio` | 0.99939 | 0.99896 | both ≈1 |
| `reward` (mean / last) | −0.239 / −0.154 | −0.227 / +0.227 | comparable |
| `num_masked_seqs_by_logprob_error` | 0 | 0 | both clean, no NaN |

Typical per-step values are close (both `token_mult_prob` ~1.02–1.04, `gen_kl`
~0.004–0.006); the differences are in the AVERAGE divergence and the rare-spike TAIL.

## Findings

1. **The full per-token cast (PR #3045) is numerically ROUGHER than row-scaled** in
   this experts-only/f2l4 setup: **~37% higher average `gen_kl` and `js`**, and a
   **much larger rare-spike tail** (`token_mult_prob` up to 2.6e7 vs r2's 369;
   `max_seq` 2.6e10 vs 2e5). The worst-case per-token routing-flip / quant spikes are
   far more extreme with the new cast.
2. **Convergence not obviously hurt (yet)** on this early window: `grad_norm`,
   `approx_entropy`, `reward`, `loss` are comparable (fp4fwd even edges ahead on
   last-step reward, though noisy). So worse *fidelity* has not translated into worse
   *reward* over 0–213 steps.
3. Both keep `probs_ratio` ≈ 1 and 0 logprob-masked sequences — no NaN, no
   catastrophic masking.

## Caveats

- Different runs / seeds / dates (r2 `state=crashed` at ~504 steps); only 214
  overlapping steps. Fidelity deltas (gen_kl/js/tail) are consistent across steps and
  robust; the reward/convergence read is early and noisy.
- Neither has router replay, so both carry the NVFP4 routing-flip tail
  (cf. `nvfp4-r2-divergence-rootcause` memory); the tail is *larger* for full
  per-token here.
- Not yet controlled: a clean forward-cast A/B would run row-scaled vs full-per-token
  from the same seed on the same TE build. This compares the two *implementations as
  shipped*, which is what was asked.

## Takeaway

Enabling the PR #3045 full per-token forward, in exchange for its being the path that
*also* unlocks FP4 per-token backward, costs ~37% higher train/gen divergence and a
heavier outlier tail vs the row-scaled forward — without a clear reward penalty over
the first ~200 steps. Re-check the tail and reward over a longer matched horizon
before drawing convergence conclusions.
