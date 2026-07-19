# Implementation plan: TE NVFP4 per-token W4A4 (train → refit → rollout)

Executes `DESIGN-te-nvfp4-pertoken.md`. Steps are ordered so every step is
independently testable and lands green; each PR follows conventional-commit
titles + sign-off (`-s`) and triggers CI with `/ok to test <full-sha>`.

## PR breakdown (rev. 2026-07-19: nemo-rl-embedded first, Bridge later/optional)

**Revision:** the original plan routed refit quantization through a new
Megatron-Bridge exporter (PR-A). That is deferred: the base worker already
receives TP-gathered, HF-named, on-GPU tensors from
`megatron_bridge.export_hf_weights`, so quantization is a pure **iterator
filter inside nemo-rl** — same compute point (post-gather, pre-IPC, on GPU;
4x transfer saving preserved), no cross-repo PR, no submodule bump. Bonus:
quantizing the per-expert HF names the base export emits produces exactly the
ModelOpt NVFP4 HF checkpoint layout vLLM's loaders already consume — no
fused-family batching in the transport at all. The Bridge exporter becomes an
optional later streamlining, justified only if refit profiling shows
per-expert kernel-launch overhead (then: batch into stacked (E,N,K) calls
bridge-side) or a non-nemo-rl consumer appears.

| PR | Repo | Title | Depends on |
|---|---|---|---|
| PR-1 | NeMo-RL | `feat(vllm): nvfp4_pertoken quantized rollout (producer, refit filter, registered config)` | vLLM repin; #2983 only if transport-factoring included (see Step 4) |
| PR-2 | NeMo-RL | `feat(megatron): fp4_cfg TE NVFP4 training config` | TE pin decision |
| PR-3 | NeMo-RL | `test: nvfp4_pertoken functional + nightly recipe` | PR-1, PR-2, GB200 env |
| PR-4 (optional) | Megatron-Bridge + NeMo-RL | `refactor(conversion): move NVFP4 refit quantization into bridge exporter` | e2e validated; profiling shows need |

---

## Phase P0 — producer + refit filter (no vLLM, no TE, no Bridge deps)

### Step 1: vendored NVFP4 producer (nemo-rl)
**File:** `nemo_rl/models/generation/vllm/quantization/nvfp4_pertoken.py` (new)

```python
def quantize_nvfp4_weight(weight: torch.Tensor)  # (N,K) or (E,N,K) bf16, CUDA
    -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # packed uint8 (...,K/2), block-16 e4m3 (...,K/16), fp32 global scalar/(E,)
```

The 2D per-tensor variant is the primary interface: the base export stream
yields per-expert HF names (`...experts.{e}.gate_proj.weight`), each a 2D
tensor, and per-expert amax ≡ per-tensor amax. The 3D stacked variant is kept
for the bitwise test against vLLM's `_quantize_moe_weight_to_nvfp4` and for
the optional future Bridge batching (PR-4).

- **Pure torch** (amax → global scale fold → block-16 e4m3 scales → E2M1
  round + nibble-pack). CRITICAL: no vLLM import — this runs on **training
  workers** (mcore venv, no vLLM installed). Guard all vLLM imports in this
  module inside functions.
- Match vLLM's `_quantize_moe_weight_to_nvfp4` semantics exactly:
  `global_scale = (6*448)/amax`, `scaled_fp4_quant(w*g, 1)` equivalence,
  non-swizzled scale layout.

**Test:** `tests/unit/models/generation/test_nvfp4_pertoken_producer.py`
- GPU + `@pytest.mark.vllm`: bitwise vs `_quantize_moe_weight_to_nvfp4`
  (skips when vLLM lacks `quantization/online`; runs in the nightly container
  and later in CI post-repin). THE hard gate of P0.
- GPU-only: determinism, dtypes/shapes, K%16 validation, round-trip
  dequant≈input sanity.
**Verify:** run in the probe's vLLM-nightly sqsh on a GB200 node (same srun
pattern as `stage0_gates.sh`).

### Step 2: refit filter (nemo-rl, replaces the Bridge exporter for now)
**File:** `nvfp4_pertoken.py` (same module)

```python
def iter_nvfp4_pertoken_weights(
    base_iter: Iterator[tuple[str, torch.Tensor]],
    quant_patterns: list[str],          # explicit allowlist, e.g. ["*.experts.*"]
) -> Iterator[tuple[str, torch.Tensor]]:
    # matching *.weight → yield .weight/.weight_scale/.weight_scale_2
    # (quantized on-GPU before IPC — 4x transfer saving); rest pass through
```

- Pure filter over the stream the base worker already produces
  (`megatron_bridge.export_hf_weights` output: TP-gathered, HF-named,
  on-GPU). No Bridge changes, no submodule bump.
- Output names = per-expert ModelOpt NVFP4 HF checkpoint layout — exactly
  what vLLM's NVFP4 loaders consume from disk; no fused-family batching in
  the transport.
- (Deferred PR-4: move this loop bridge-side with stacked-(E,N,K) batching if
  refit profiling shows per-expert launch overhead matters.)

**Test:** pure-python unit with a fake iterator: pattern selection, 3-tensor
yield order + naming, pass-through of non-matching weights, GPU tensors stay
on device.
**Verify:** covered by the producer's GPU test (Step 1) + Step 10's
injected-weights probe leg.

---

## Phase P1 — vLLM rollout path + worker hook + config (nemo-rl PR-1)

### Step 3: graduate the probe overlay (neutral module)
**File:** `nvfp4_pertoken.py` (same module as Step 1)

- Move/rename probe classes: `NvFp4PerTokenConfig` / `NvFp4PerTokenFusedMoE`,
  `register_nvfp4_pertoken()` registering `"nvfp4_pertoken"`. Keep the
  `.contiguous()` reload fix + FLASHINFER_TRTLLM assert. Classes at module
  scope (pickling to engine procs — probe lesson), vLLM imports inside the
  register function (training-venv importability).
- `build_nvfp4_pertoken_hf_quant_config(ignore: list[str]) -> dict` — literal
  dict (quant_algo NVFP4, group_size 16, dynamic input activations,
  exclude_modules), no ModelOpt helper.
- Re-home `DEFAULT_NVFP4_IGNORE` here; `nemo_rl/modelopt/utils.py` re-imports
  (one-way dependency).

### Step 4: transport factoring + worker extension
**Files:** `nemo_rl/models/generation/vllm/quantization/nvfp4_transport.py`
(new), `nemo_rl/modelopt/models/generation/vllm_quant_backend.py` (refactor)

- Extract format-generic pieces (parameterized where #2983 already did:
  `require_input_scales` flag exists): fused-MoE suffix map (input-scale
  entries optional), shard/manifest validation, weight batching/unbind,
  layerwise-reload lifecycle helpers, ZMQ timeout, EP-rejection check.
- `NvFp4PerTokenWorkerExtension(VllmInternalWorkerExtension)`: lifecycle =
  initialize/finalize layerwise reload over MoE roots, manifest without input
  scales, fatal errors, accelerator sync. ~100 lines over the shared helpers.
- ModelOpt backend switches to importing the factored helpers — **behavior
  frozen**: `tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py`
  must pass unchanged (this is the refactor's regression gate).

### Step 5: engine configuration + generation worker selection
**Files:** `nemo_rl/models/generation/vllm/config.py`,
worker-selection site (where `VllmQuantGenerationWorker` is chosen today),
plus a small `configure_nvfp4_pertoken_engine_kwargs()` in the neutral module

- `VllmConfig` (existing TypedDict — extend, don't create new TypedDicts):
  `nvfp4_pertoken_rollout: NotRequired[NvFp4PerTokenRolloutConfig]` where the
  sub-config is a new v2 `BaseModel(extra="allow")`:
  `enabled: bool = False`, `ignore: list[str] | None = None` (None →
  `DEFAULT_NVFP4_IGNORE`). Defaults live on the BaseModel only.
- Engine kwargs: `quantization="nvfp4_pertoken"`,
  `hf_overrides["quantization_config"]=build_...(ignore)`,
  `worker_extension_cls=<NvFp4PerTokenWorkerExtension path>`; register before
  engine build (mirror `_configure_quant_engine_kwargs`'s shape, no
  VLLM_MODELOPT_REAL_QUANT / VLLM_QUANT_CFG).
- **Guards (fail loudly at startup):** mutual exclusion with
  `generation.quant_cfg`/`real_quant`; warn when `fp4_cfg` f2l4 layers
  diverge from the rollout ignore list.
- Exemplar YAML documentation + `tests/unit/reference_configs/` sync
  (`test_config_v2.py` gate).

### Step 6: training-side refit hook
**File:** `nemo_rl/models/policy/workers/megatron_policy_worker.py`

- `_iter_nvfp4_pertoken_refit_params()`: wraps
  `_iter_params_with_optional_kv_scales()` (or the base export iterator plus
  the shared kv-scale tail) with `iter_nvfp4_pertoken_weights(...,
  quant_patterns)` from Step 2 — kv scales and draft weights pass through the
  filter untouched.
- Selection: where refit iterators are chosen, gate on
  `generation.nvfp4_pertoken_rollout.enabled`.
- `prepare_refit_info`: state_dict_info must advertise the **quantized**
  names/shapes/dtypes (packed w13/w2 + scales, stacked expert layout) so the
  manifest validates — add `build_nvfp4_refit_state_dict_info()` next to the
  producer (single source of truth for shapes); mirror how #2983's
  `prepare_refit_info` handles fused-MoE families, minus input scales.

### Step 7: P1 unit tests (CPU-mocked, run in today's CI)
`tests/unit/models/generation/test_nvfp4_pertoken_rollout.py` +
`tests/unit/models/policy/test_megatron_worker.py` additions:
- config guards (mutual exclusion, ignore defaulting, f2l4 consistency warn)
- engine-kwargs configuration (fake vLLM in sys.modules — reuse #2983 test fakes)
- manifest accepts no-input-scale families / rejects incomplete ones
- refit iterator yields quantized names (fake bridge, stub quantize_fn)
- transport-refactor regression: modelopt tests unchanged

---

## Phase P2 — environment + integration + e2e (PR-2, PR-3)

### Step 8: environment
- vLLM repin ≥ #48538 via `/env-profile` → `/env-build` sqsh (shared
  prerequisite with the ModelOpt per-token mode; coordinate once).
- TE pin with per-token NVFP4: decide upstream-TE vs study-fork pin;
  `/env-refresh` the worker venvs. Gates PR-2 only.

### Step 9: `fp4_cfg` upstreaming (PR-2)
- Port the rl-fp4 study's TE NVFP4 wiring next to `fp8_cfg`
  (`megatron_policy_worker.py:373` pattern): `Fp4Config` BaseModel
  (`enabled=False`, `recipe="nvfp4_pertoken"`, f2l4 knobs), exemplar YAML doc,
  reference-config sync. Training-only — no refit coupling in this PR.

### Step 10: integration via the probe (no new code)
- Point probe checks B2/C/D at `"nvfp4_pertoken"` (one-line rename in
  `research/vllm-nvfp4-pertoken/`), run in the repinned container on GB200.
- Add a check-B3 leg: weights produced by Step-1 producer injected via
  reload → outputs identical to checkpoint-loaded weights (closes the loop
  refit-payload → kernel without Megatron).

### Step 11: functional + recipe + nightly (PR-3)
- `tests/functional/grpo_vllm_nvfp4_pertoken_rollout_gb200.sh` modeled on
  `grpo_vllm_mxfp8_rollout_gb200.sh`; assert_grep markers from the probe
  README (`Using 'FLASHINFER_TRTLLM' NvFp4 MoE backend`,
  `quantization=nvfp4_pertoken`; forbid `VllmQuantInternalWorkerExtension`,
  `FakeQuantWorker`); `check_metrics.py`: step-1 `gen_kl_error < 0.03`,
  `token_mult_prob_error < 1.15` (w4a4-static's values; tighten after data).
- Recipe `examples/configs/recipes/llm/grpo-qwen3-30ba3b-4n4g-megatron-te-nvfp4-pertoken.yaml`
  (+`.sh`), `defaults:` inheritance, minimized via `tools/config_cli.py`;
  register in `tests/test_suites/nightly_gb200.txt` (`/register-nightly`).
  Initial thresholds from the w4a4-static recipe (reward ≥ 0.25,
  validation accuracy ≥ 0.4, gen_kl < 0.03, js_div < 0.007).

## Phase P3 — alignment study (rl-fp4 harness, no nemo-rl code)
- TE↔FlashInfer recipe parity legs on the m-inf study harness; decide whether
  to configure TE weight quant to 1D block-16 / 4over6 parity based on
  measured train/gen gap.

## Sizes & sequencing at a glance

```
P0  Step 1 producer+test        ~250 loc   ─┐ parallelizable
    Step 2 refit filter         ~80 loc    ─┘ (same module)
P1  Step 3 graduate overlay     ~200 loc (mostly moves)
    Step 4 transport factor     ~400 loc moved + ~100 new   ← riskiest (refactor gate)
    Step 5 config+engine        ~150 loc
    Step 6 refit hook           ~200 loc
    Step 7 unit tests           ~400 loc
P2  Step 8 env (ops)            sqsh bakes
    Step 9 fp4_cfg port         study-patch sized
    Step 10 probe rerun         ~0 loc
    Step 11 functional+nightly  ~200 loc scripts/yaml
```

Critical path: Step 1 → 2 → 6 (payload correctness, all in one nemo-rl
module), with Step 4 the merge risk (coordinate with #2983's fate before
starting it). The optional Bridge move (PR-4) happens only after e2e is
validated and profiled.
