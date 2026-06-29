# Gemma4 Model-Owned Context Parallel Flow

This note explains the logical function invocation flow for context parallelism
(CP) in NeMo RL when using Automodel, with focus on why Gemma4 needs a
model-owned CP path instead of the generic SDPA or Transformer Engine (TE) CP
path.

## Mental Model

Context parallelism splits the sequence dimension across CP ranks. The training
loop still reasons about the full logical sequence, but each rank only runs a
local sequence shard. The attention backend is responsible for making remote
key/value information visible to the local query shard.

At a high level, every CP implementation must satisfy this contract:

```text
full logical [B, S] input
  -> sequence-sharded model forward
  -> local shard logits or logprobs
  -> loss/logprob code maps results back to the full logical sequence
```

Generic SDPA CP, generic TE CP, and Gemma4 model-owned CP all follow this
logical contract. They differ in who shards the batch, who wires the attention
communication, and what layout is used for sequence shards.

## Generic CP Flow

The generic path is used for models whose attention can run through the normal
SDPA or TE CP machinery. NeMo RL does not need model-specific CP code here.

```mermaid
flowchart TD
    A["DTensorPolicyWorkerV2Impl.train()"] --> B["use_model_owned_cp = False"]
    B --> C["effective_cp_size = real cp_size"]
    C --> D["get_microbatch_iterator(..., cp_size=real cp_size)"]
    D --> E["process_microbatch()"]
    E --> F["build input_ids, position_ids, attention_mask"]
    F --> G{"cp_size > 1?"}
    G -->|yes| H["attention_mask = None\nseq_index = arange(seq_len)\ncp_buffers = [input_ids, position_ids, seq_index]"]
    G -->|no| I["normal non-CP inputs"]
    H --> J["automodel_forward_backward(..., train_context_fn)"]
    I --> J
    J --> K["train_context_fn(processed_inputs)"]
    K --> L["get_train_context()"]
    L --> M["create_context_parallel_ctx(cp_mesh, cp_buffers, cp_seq_dims, cp_no_restore_buffers)"]
    M --> N["with CP context: forward_with_post_processing_fn()"]
    N --> O["model_forward()"]
    O --> P["model(**processed_inputs)"]
    P --> Q["extract_logits()"]
    Q --> R["LossPostProcessor / LogprobsPostProcessor"]
    R --> S["prepare_data_for_cp()"]
    S --> T["redistribute_logits_for_cp()"]
    T --> U["get_logprobs_from_vocab_parallel_logits() or loss_fn"]
```

Function responsibilities:

| Layer | Functions | Responsibility |
| --- | --- | --- |
| NeMo RL worker | `DTensorPolicyWorkerV2Impl.train`, `get_logprobs` | Select the generic path, build iterators, create post-processors, call forward/backward. |
| NeMo RL data | `process_microbatch` | Build `input_ids`, `position_ids`, and `cp_buffers` for generic CP. |
| NeMo RL context | `get_train_context` | Build and enter `create_context_parallel_ctx` around the model forward. |
| Automodel / torch CP | `create_context_parallel_ctx` | Shard/restore CP buffers and make SDPA see CP-sharded query/key/value tensors. |
| NeMo RL post-processing | `prepare_data_for_cp`, `redistribute_logits_for_cp`, `get_logprobs_from_vocab_parallel_logits` | Convert data and logits into DTensor placements and compute loss/logprobs. |

For SDPA CP, the generic CP context and SDPA hooks handle remote key/value
visibility. For TE CP, Automodel configures TE attention modules with the CP
group, for example through `DotProductAttention.set_context_parallel_group(...)`.
TE then owns the CP communication inside `DotProductAttention`.

The common interface works because NeMo RL only needs a logical sequence-sharded
forward and a way to post-process local outputs. The attention kernel details
are behind the SDPA or TE backend.

## Gemma4 Model-Owned CP Flow

Gemma4 cannot use the generic torch CP SDPA path for the long-context dense E2B
setup. Its CP behavior is owned by the model implementation in Automodel:
Gemma4 prepares full-sequence embeddings, keeps contiguous sequence shards, and
installs its own ring/FlexAttention path for key/value exchange and masking.

NeMo RL therefore bypasses the generic CP context and asks Gemma4/Automodel to
produce CP-sharded logits. NeMo RL then computes only the current-policy
logprobs needed by RL loss.

```mermaid
flowchart TD
    A["DTensorPolicyWorkerV2Impl.train()"] --> B["use_model_owned_cp = cp_size > 1 and model_has_model_owned_cp(model)"]
    B --> C["effective_cp_size = 1"]
    C --> D["define cp_curr_logprobs_fn(processed_inputs)"]
    D --> E["LossPostProcessor(..., cp_size=1)"]
    E --> F["loss_fn.use_linear_ce_fusion = True"]
    F --> G["get_microbatch_iterator(..., cp_size=1)"]
    G --> H["automodel_forward_backward(train_context_fn=None, cp_curr_logprobs_fn, cp_size=real cp_size)"]
    H --> I["forward_with_post_processing_fn()"]
    I --> J["cp_curr_logprobs_fn(processed_inputs)"]
    J --> K["model_owned_cp_curr_logprobs()"]
    K --> L["_model_owned_cp_pred_logprobs()"]
    L --> M["_model_owned_cp_shard_logits()"]
    M --> N["model.prepare_model_inputs_for_cp(...)"]
    N --> O["Gemma4 builds inputs_embeds, mm_token_type_ids, vision group ids, per-layer inputs, _cp_make_batch_fn"]
    O --> P["make_cp_batch_and_ctx(device_mesh, batch)"]
    P --> Q{"batch has _cp_make_batch_fn?"}
    Q -->|yes| R["Gemma4._cp_shard_batch(cp_mesh, tp_mesh, batch)"]
    R --> S["Gemma4.setup_cp_attention(cp_mesh)"]
    S --> T["install Gemma4 ring/FlexAttention hooks on attention modules"]
    R --> U["make_contiguous_shard_cp_batch_and_ctx()"]
    U --> V["rank r keeps x[:, r*L:(r+1)*L]"]
    V --> W["model(**sharded)"]
    W --> X["Gemma4 attention ring exchanges K/V and metadata"]
    X --> Y["local shard logits [B, S/cp, V]"]
    Y --> Z["local log_softmax and target gather"]
    Z --> AA["_cp_contiguous_allgather([B, S/cp])"]
    AA --> AB["precomputed current logprobs [B, S-1]"]
    AB --> AC["LossPostProcessor treats input as LOGPROB"]
    AC --> AD["loss.backward() on result * dp_size * real cp_size"]
```

Function responsibilities:

| Layer | Functions | Responsibility |
| --- | --- | --- |
| NeMo RL worker | `DTensorPolicyWorkerV2Impl.train` | Detect model-owned CP, set `effective_cp_size=1`, skip generic CP context, pass `cp_curr_logprobs_fn` into forward/backward. |
| NeMo RL bridge | `model_owned_cp_curr_logprobs`, `_model_owned_cp_pred_logprobs`, `_model_owned_cp_shard_logits` | Call the model-owned CP prep/forward path, compute next-token logprobs from local shard logits, gather only cheap logprob tensors. |
| Automodel generic dispatch | `make_cp_batch_and_ctx` | Detect `_cp_make_batch_fn` on the batch and delegate CP batch handling back to the model. |
| Gemma4 model | `prepare_model_inputs_for_cp` | Embed the full sequence before CP sharding and attach Gemma4-specific metadata plus `_cp_make_batch_fn`. |
| Gemma4 model | `_cp_shard_batch`, `setup_cp_attention` | Install model-owned CP attention and call the contiguous batch sharder. |
| Gemma4 CP batch | `make_contiguous_shard_cp_batch_and_ctx` | Pad sequence tensors and keep a rank-ordered contiguous sequence slice per CP rank. |
| Gemma4 CP attention | `attach_gemma4_cp_ring_attention`, ring attention helpers | Implement Gemma4-specific K/V ring exchange, mask handling, packed sequence metadata, and vision metadata handling. |

## Logprob Path

The model-owned path has a separate no-grad logprob flow used by policy logprob
queries. It shares the same CP forward core but returns token logprobs directly.

```mermaid
flowchart TD
    A["DTensorPolicyWorkerV2Impl.get_logprobs()"] --> B["use_model_owned_cp?"]
    B -->|yes| C["effective_cp_size = 1"]
    C --> D["get_microbatch_iterator(..., cp_size=1)"]
    D --> E["model_owned_cp_token_logprobs()"]
    E --> F["_model_owned_cp_pred_logprobs()"]
    F --> G["_model_owned_cp_shard_logits()"]
    G --> H["Gemma4/Automodel model-owned CP forward"]
    H --> I["local log_softmax on [B, S/cp, V]"]
    I --> J["_cp_contiguous_allgather([B, S/cp])"]
    J --> K["prepend zero for token 0"]
    K --> L["mask padding with input_lengths"]
    L --> M["token_logprobs [B, S]"]
```

This avoids materializing full logits of shape `[B, S, V]` on every rank. For
long context, that is the main reason this path exists.

## Why NeMo RL Needs Extra Code

The model-owned CP support in NeMo RL is not implementing Gemma4 attention.
Automodel/Gemma4 does that. The extra NeMo RL code is needed to adapt Gemma4's
model-owned CP forward to NeMo RL's RL loss and logprob interfaces.

Key reasons:

- Generic NeMo RL CP assumes torch CP's sharding and restore layout. Gemma4 uses
  rank-ordered contiguous sequence shards, so NeMo RL needs
  `_ContiguousCPAllGather` instead of `allgather_cp_sharded_tensor`.
- RL loss needs next-token current-policy logprobs. Gathering full logits
  `[B, S, V]` would erase the memory benefit of CP at 32k/64k context, so NeMo
  RL computes logprobs locally on `[B, S/cp, V]` and gathers only `[B, S]`.
- Gemma4 pre-embeds the full sequence through `prepare_model_inputs_for_cp`.
  That direct embedding call happens outside the normal top-level FSDP forward
  hooks, so NeMo RL temporarily unshards FSDP modules for the prep step and
  reshards them before the real model forward.
- Gemma4 dense E2B/E4B uses shared KV state. NeMo RL injects Automodel's
  FSDP-safe shared KV store for Gemma4 so FSDP input casting does not copy a
  plain dict per layer.
- Existing loss plumbing expects either logits or a specific logprob path. The
  current integration marks `loss_fn.use_linear_ce_fusion = True` so
  `LossInputType.LOGPROB` treats the tensor as precomputed logprobs.
- Backward scaling still needs the real CP size. Even though the iterator and
  postprocessor run with `effective_cp_size=1`, `automodel_forward_backward`
  receives the real `cp_size` so `loss = result * dp_size * cp_size` cancels the
  FSDP averaging over the flattened DP/CP group.

## bf16 Autocast Workaround for Native MoE Gemma4 (26B-A4B)

Both the model-owned CP forward (`_model_owned_cp_shard_logits`, `cp_size > 1`)
and the generic non-CP forward (`model_forward`, `cp_size == 1`) wrap the Gemma4
model call in `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`. This is
a workaround, not a design feature. This section records the analysis so the wrap
can be removed once the underlying precision contract is fixed upstream.

### Symptom

Training or `get_logprobs` on the native (non-HF) MoE Gemma4 backend — 26B-A4B —
fails at the `lm_head` projection with:

```text
RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != BFloat16
```

Dense Gemma4 (E2B / E4B / 31B) is unaffected.

### Root cause

Two independent facts combine:

1. NeMo RL disables autocast for native MoE models. In `setup.py`:

   ```python
   is_moe_model = any("expert" in k for k in model_state_dict_keys)   # True for 26B-A4B
   is_hf_model  = arch not in Automodel's custom ModelRegistry        # False (custom-registered)
   autocast_enabled = not (is_moe_model and not is_hf_model)          # -> False for 26B-A4B
   ```

   The intent: native MoE backends run TE/DeepEP expert kernels that manage their
   own precision, and blanket autocast was found to cause numerical issues.

2. The native MoE Gemma4 forward delivers an **fp32** activation into a **bf16**
   `lm_head`. The model's final `Gemma4RMSNorm` output is fp32, and Automodel's
   `gemma4_moe` forward calls `self.lm_head(hidden_states[...])` directly with no
   explicit downcast (it relies on an ambient autocast region to reconcile the
   boundary).

With autocast disabled (fact 1) and an fp32 → bf16 matmul (fact 2) and no explicit
cast, the `lm_head` matmul raises. Dense Gemma4 keeps `autocast_enabled=True`, so
the worker already runs its forward under autocast and never hits this.

### Why this is effectively an Automodel-bump regression

The `autocast_enabled` rule is identical on `gemma4-support` and `gemma4-cp`, so
the disabled-autocast condition is *not* the new factor. The new factor is
Automodel's precision contract:

| Branch | Automodel pin | 26B-A4B at `lm_head` |
| --- | --- | --- |
| `gemma4-support` | `~6de0c361` | delivered bf16 -> no error, no wrap needed |
| `gemma4-cp` | `1ff3afc9` | delivers fp32 -> crashes without the wrap |

Four precision/dtype-contract commits landed in the native `gemma4_moe` path
between those pins (present in the newer pin, not the older):

```text
#1896  fp32 master weights for custom MoE models under FSDP2
#2419  dtype contract bug fixes for FSDP2 mixed-dtype loads
#2549  keep RoPE frequency buffers fp32 under bf16 model cast
#2359  cast dense params without casting buffers
```

These tightened the contract so the native MoE forward now carries fp32 hidden
states into the bf16 `lm_head`, where the older Automodel handed it bf16. So the
crash is the combination of a long-standing NeMo RL rule (autocast off for native
MoE) and a *new* Automodel fp32 contract introduced by the dependency bump — not a
NeMo RL design change.

### Where the fix lives

| Path | `cp_size` | Location | Landed in |
| --- | --- | --- | --- |
| Model-owned CP forward | `> 1` | `_model_owned_cp_shard_logits` (`with train_ctx(), autocast_ctx: model(**sharded)`) | model-owned CP commit |
| Generic forward | `== 1` | `model_forward` (`if model_type == "gemma4": with torch.autocast(bf16): model(...)`) | `21c5e3c7b` |

The earlier model-owned-CP fix only covered the `cp_size > 1` path; `21c5e3c7b`
extends the identical remedy to the generic `cp_size == 1` path so 26B-A4B trains
without CP too. The wrap is a no-op nesting for dense Gemma4, which already runs
under autocast.

### Evaluation and recommended upstream fix

- **Consistent and low risk.** The generic-path wrap mirrors the already-validated
  model-owned-CP wrap. Verified: 26B-A4B `cp=1` baseline trains; dense 31B
  unaffected.
- **Caveat.** Autocast was disabled for native MoE precisely because TE/DeepEP
  kernels self-manage precision; re-enabling autocast over the whole forward
  works slightly against that intent. It is fine empirically (autocast is a no-op
  for the TE-internal ops and the boundary it fixes is the lm_head), but it is the
  reason this is a workaround rather than the desired end state.
- **Recommended upstream fix.** The clean fix belongs in Automodel: the
  `gemma4_moe` forward should cast `hidden_states` to `lm_head.weight.dtype` before
  the matmul, or route the projection through `compute_lm_head_logits` with explicit
  dtype handling. Then the precision contract is self-consistent regardless of
  caller, and the NeMo RL autocast wraps in both `_model_owned_cp_shard_logits` and
  `model_forward` can be removed.

## Difference From Existing Generic CP Support

| Aspect | Generic SDPA/TE CP | Gemma4 model-owned CP |
| --- | --- | --- |
| CP selection | NeMo RL uses normal CP when `cp_size > 1`. | NeMo RL detects `prepare_model_inputs_for_cp` and switches to model-owned path. |
| Microbatch CP size | Real `cp_size`. | `effective_cp_size=1` to avoid generic CP buffer/context handling. |
| Batch sharding | `create_context_parallel_ctx` or TE batch conversion handles sharding. | Gemma4's `_cp_make_batch_fn` handles contiguous sharding. |
| Attention communication | SDPA CP hooks or TE `DotProductAttention`. | Gemma4 ring/FlexAttention implementation. |
| Sequence layout | Generic torch CP layout, including load-balanced/zigzag assumptions. | Rank-ordered contiguous slices. |
| Logits handling | Postprocessor can redistribute logits through DTensor CP/TP placements. | NeMo RL avoids full logits and gathers only per-token logprobs. |
| Model-specific code in NeMo RL | Minimal. | Bridge code for model-owned forward, logprob computation, contiguous gather, FSDP prep, and loss adaptation. |

## Future Model-Specific CP Support

The current design is a useful foundation for future model-specific CP support,
but it is only partially generic.

The Automodel handoff point is generic:

```text
model.prepare_model_inputs_for_cp()
  -> returns batch additions
  -> may attach _cp_make_batch_fn

make_cp_batch_and_ctx()
  -> sees _cp_make_batch_fn
  -> delegates CP sharding back to the model
```

This means a future model can own its CP batch sharding and attention algorithm
inside Automodel without forcing NeMo RL to know the internal attention details.
That is the right abstraction boundary for model-specific CP.

The current NeMo RL bridge, however, is still mostly a Gemma4-shaped adapter. It
assumes:

```text
local logits shape: [B, S/cp, V]
sequence layout: rank-ordered contiguous shard
rank r owns: input_ids[:, r*L : (r+1)*L]
gather: plain rank-order all_gather + cat
targets: full input_ids[:, seq_start+1 : seq_start+1+L]
```

Those assumptions appear in the logprob path:

```text
local_seq_len = shard_logits.shape[1]
seq_start = cp_rank * local_seq_len
targets = full_ids[:, seq_start + 1 : seq_start + 1 + local_seq_len]
_cp_contiguous_allgather(...)
```

So the current path naturally adapts to future models only when they share the
same basic layout and semantics:

```text
causal LM
text-only RL input
local logits as [B, local_seq, vocab]
rank-ordered contiguous sequence shards
same next-token target semantics
same differentiable contiguous gather behavior
```

Models with contiguous model-owned CP, such as a DeepSeek-V4-style
implementation, may be close to this shape. Models with packed or THD-specific
layouts, such as GLM DSA-style CP, would need a different logprob/gather
adapter. Multimodal models such as Qwen, Nemotron-Omni, or Step3p7 are another
important distinction: they may expose `prepare_model_inputs_for_cp` only to
perform full-sequence multimodal pre-embedding before generic CP. That does not
necessarily mean they own CP attention and sharding in the Gemma4 sense.

The weakest part of the current design is therefore the capability detector:

```python
return hasattr(model, "prepare_model_inputs_for_cp")
```

That conflates two different capabilities:

```text
prepare inputs before generic CP
```

and:

```text
the model owns CP sharding and attention
```

For future support, NeMo RL should use an explicit model-owned CP contract
instead of inferring it from `prepare_model_inputs_for_cp`. For example:

```python
model.supports_nemorl_model_owned_cp = True
model.nemorl_model_owned_cp_layout = "contiguous_bsh"
```

or a small adapter interface:

```python
adapter = get_model_owned_cp_adapter(model)

adapter.prepare_batch(...)
adapter.shard_logits(...)
adapter.local_target_ids(...)
adapter.gather_logprobs(...)
adapter.extra_forward_kwargs(...)
```

That would let NeMo RL support multiple layouts explicitly:

```text
contiguous_bsh -> current Gemma4-style path
packed_thd     -> packed/THD model-owned CP path
custom         -> model adapter computes or gathers logprobs itself
```

The summary is:

```text
Automodel dispatch design:
  reasonably generic

NeMo RL Gemma4 bridge:
  generic only for Gemma4-like contiguous causal LM CP

Recommended future direction:
  add explicit capability detection and layout-specific adapters before using
  this path broadly for arbitrary model-specific CP implementations
```

The division of labor is therefore:

```text
Automodel/Gemma4 owns:
  - CP-aware model input preparation
  - CP batch sharding layout
  - Gemma4-specific attention communication and masks
  - K/V ring behavior and model metadata

NeMo RL owns:
  - choosing generic CP vs model-owned CP
  - integrating the model-owned forward into RL train/logprob APIs
  - avoiding full [B, S, V] logits for long context
  - preserving gradient behavior and loss normalization
```
