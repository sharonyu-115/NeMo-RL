# Upgrading vLLM to v0.16 (main) in NeMo RL: Troubleshooting Guide

This document tracks all issues encountered when upgrading vLLM to v0.16 (from `main` branch) in NeMo RL, the root causes, resolutions, and useful debugging commands.

## Prerequisites

Run the `build-custom-vllm.sh` script with a specific vLLM commit and its matching precompiled wheel:

```bash
bash tools/build-custom-vllm.sh \
  https://github.com/vllm-project/vllm.git \
  111d86906 \
  https://wheels.vllm.ai/111d8690699927af686fa6750cfbbc692a1f8740/vllm-0.16.1rc1.dev12%2Bg111d86906-cp38-abi3-manylinux_2_31_x86_64.whl
```

This clones vLLM into `3rdparty/vllm/`, builds it using the precompiled wheel, and patches `pyproject.toml` to use the local source as an editable install.

> **Important:** Always use a specific commit hash rather than `main` to ensure the precompiled wheel matches the source code. Precompiled wheels are available at `https://wheels.vllm.ai/<full-commit-hash>/` — verify the wheel URL corresponds to the commit you're checking out.

## Issues Encountered

### Issue 1: `uv lock` fails — flashinfer-python conflict between vllm and mcore

**Error:**

```
× No solution found when resolving dependencies:
  Because vllm depends on flashinfer-python==0.6.4 and megatron-core depends on
  flashinfer-python>=0.5.0,<0.6.dev0, we can conclude that megatron-core and
  nemo-rl[vllm] are incompatible.
```

**Root cause:** vLLM v0.16 requires `flashinfer-python==0.6.4`, but `megatron-core` (a workspace member) constrains it to `<0.6.dev0`. The `vllm` and `mcore` extras were not declared as conflicting, so `uv lock` tried to resolve them together and failed.

**Resolution:** Add a conflict declaration between the `vllm` and `mcore` extras in `pyproject.toml`:

```toml
conflicts = [
  # ... existing conflicts ...
  [
    { extra = "vllm" },
    { extra = "mcore" },
  ],
]
```

**Why this works:** The `conflicts` section tells uv that these extras will never be installed in the same venv, so it can produce separate resolutions for each. This is correct because NeMo RL always installs vllm and mcore in separate worker venvs.

**Important nuance:** Adding the conflict resolved the *extra-level* dependency split, but `megatron-core` as a **workspace member** still participates in resolution globally (not just through the `mcore` extra). This required further overrides (see Issue 2).

**Debug commands:**

```bash
uv lock 2>&1 | head -20          # See which split fails and what's included/excluded
grep "conflicts" pyproject.toml   # Check current conflict declarations
```

---

### Issue 2: `uv lock` fails — flashinfer-python still conflicts via workspace member

**Error:**

```
× No solution found (included: nemo-rl[automodel], nemo-rl[fsdp], nemo-rl[vllm];
  excluded: nemo-rl[mcore], nemo-rl[sglang]):
  megatron-core depends on flashinfer-python>=0.5.0,<0.6.dev0
```

**Root cause:** Even with `mcore` extra excluded, `megatron-core` is a **workspace member** (listed in `[tool.uv.workspace].members`). All workspace members' dependencies participate in resolution regardless of which extras are included. The `conflicts` mechanism only controls NeMo RL's extras — it cannot exclude a workspace member's own constraints.

**Resolution:** Add an override to relax the flashinfer-python constraint:

```toml
override-dependencies = [
  # ... existing overrides ...
  "flashinfer-python>=0.5.0,<0.7.0",
]
```

**Why `>=0.5.0` and not `==0.6.4`:** Pinning `==0.6.4` applies globally to all splits, including the sglang split where it caused a cascade conflict with `nvidia-cutlass-dsl`. Using `>=0.5.0` lets each split pick its own compatible version.

**Debug commands:**

```bash
uv lock 2>&1 | grep "included:"   # See which extras are in the failing split
uv lock 2>&1 | grep "workspace"   # Identify workspace member constraints
```

---

### Issue 3: `uv lock` fails — sglang split breaks with `flashinfer-python==0.6.4` override

**Error:**

```
× No solution found (included: nemo-rl[sglang]; excluded: others):
  sglang==0.5.7 depends on nvidia-cutlass-dsl==4.2.1
  flashinfer-python==0.6.4 depends on nvidia-cutlass-dsl>=4.3.4
  → sglang==0.5.7 cannot be used
```

**Root cause:** An initial attempt used `"flashinfer-python==0.6.4"` as the override, which forced 0.6.4 globally — including in the sglang split. flashinfer 0.6.4 requires `nvidia-cutlass-dsl>=4.3.4`, but sglang pins `nvidia-cutlass-dsl==4.2.1`, creating an internal conflict within the sglang split.

**Resolution:** Use `"flashinfer-python>=0.5.0,<0.7.0"` instead of `==0.6.4`. This allows each split to resolve independently.

---

### Issue 4: `uv lock` fails — opentelemetry-api / protobuf conflict

**Error:**

```
× No solution found:
  megatron-core depends on opentelemetry-api>=1.33.1,<1.34.dev0
  opentelemetry 1.33.x depends on protobuf>=5.0,<6.0
  ray==2.49.2 depends on protobuf>=6.33.5
```

**Root cause:** `megatron-core` (workspace member) pins `opentelemetry-api` to 1.33.x, which depends on `protobuf<6.0`. But `ray==2.49.2` requires `protobuf>=6.33.5`. These are mutually exclusive, and since megatron-core is a workspace member, its constraint applies globally.

**Resolution:** Override the opentelemetry-api constraint:

```toml
override-dependencies = [
  # ... existing overrides ...
  "opentelemetry-api>=1.33.1",
]
```

**Debug commands:**

```bash
uv lock 2>&1 | grep "protobuf"           # Trace protobuf constraint chain
uv lock 2>&1 | grep "opentelemetry"       # Trace opentelemetry constraint chain
```

---

### Issue 5: `import vllm._C` fails — precompiled wheel ABI mismatch

**Error:**

```python
ImportError: .../vllm/_C.abi3.so: undefined symbol:
  _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb
```

**Root cause:** The precompiled vLLM wheel's C++/CUDA extensions (`.so` files) were compiled against **torch 2.10**, but NeMo RL was using **torch 2.9.0**. The undefined symbol `c10::cuda::c10_cuda_check_implementation` is a PyTorch CUDA runtime function that changed between versions.

**How to identify:** The symbol `_ZN3c104cuda29...` demangles to `c10::cuda::...`, which is PyTorch's internal C++ library (`libc10_cuda.so`). An undefined symbol from `c10` always indicates a torch version mismatch between compile time and runtime.

**Resolution:** Bump torch to 2.10.0 in `pyproject.toml`:

```toml
# In [project].dependencies:
"torch==2.10.0",

# In [dependency-groups].build:
"torch==2.10.0",

# In override-dependencies:
"torch==2.10.0",
```

**Debug commands:**

```bash
python -c "import torch; print(torch.__version__)"   # Check runtime torch version
c++filt _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb  # Demangle C++ symbol
```

---

### Issue 6: vLLM server crashes — `No module named 'flashinfer.gdn_prefill'`

**Error:**

```
ModuleNotFoundError: No module named 'flashinfer.gdn_prefill'
```

**Root cause:** flashinfer-python 0.5.3 was installed instead of 0.6.4. The `gdn_prefill` module only exists in flashinfer 0.6.x+. Two sub-issues caused this:

1. **torch 2.9.0 blocked flashinfer 0.6.4**: flashinfer 0.6.4 requires torch >= 2.10, so uv couldn't resolve it and fell back to 0.5.3 (fixed by Issue 5's torch bump).

2. **flashinfer was an optional extra in vLLM**: vLLM v0.16 moved flashinfer to an optional `[flashinfer]` extra. NeMo RL's `vllm` extra just installed `"vllm"` without enabling vLLM's own flashinfer extra, so `==0.6.4` was never activated in the resolution.

3. **nvidia-cutlass-dsl blocked resolution**: flashinfer 0.6.4 requires `nvidia-cutlass-dsl>=4.3.4`, but other workspace members constrained it. Without an override, uv couldn't resolve 0.6.4.

**Resolution (all three parts):**

```toml
# 1. Add flashinfer-python directly to NeMo RL's vllm extra:
vllm = ["cuda-python", "...", "vllm", "flashinfer-python==0.6.4"]

# 2. Override nvidia-cutlass-dsl to allow 4.4.0+:
override-dependencies = [
  "nvidia-cutlass-dsl>=4.2.1",
]

# 3. Torch was already bumped to 2.10.0 (Issue 5)
```

**Debug commands:**

```bash
# Check what version is actually installed
uv run --extra vllm python -c "import flashinfer; print(flashinfer.__version__)"

# Check what's in the lock file
grep flashinfer uv.lock

# Check if the package exists on PyPI
uv pip install flashinfer-python==0.6.4 --dry-run

# Force install into a worker venv for testing
source /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/activate
uv pip install flashinfer-python==0.6.4
```

---

### Issue 7: transformers version bump

**Requirement:** vLLM v0.16 and Qwen3.5 model support required a newer version of `transformers`.

**Resolution:** Add a git source for a specific commit with the needed fix:

```toml
# In [tool.uv.sources]:
transformers = { git = "https://github.com/huggingface/transformers.git", rev = "c58e711a8c688d5a154f6dbbfae54da2f31328bb" }

# In override-dependencies (to relax version constraints from other packages):
"transformers>=4.57.1"
```

---

## Summary of All `pyproject.toml` Changes

### `[project].dependencies`

| Change | Before | After | Reason |
|--------|--------|-------|--------|
| torch | `torch==2.9.0` | `torch==2.10.0` | vLLM v0.16 + precompiled wheel ABI |
| setuptools_scm | not present | `"setuptools_scm"` | Added by `build-custom-vllm.sh` for vLLM versioning |

### `[project.optional-dependencies].vllm`

| Change | Before | After | Reason |
|--------|--------|-------|--------|
| flashinfer-python | not present | `"flashinfer-python==0.6.4"` | vLLM moved flashinfer to optional extra; must be explicitly requested |

### `[tool.uv.sources]`

| Change | Before | After | Reason |
|--------|--------|-------|--------|
| vllm | PyPI/pinned version | `{path = "3rdparty/vllm", editable = true}` | Added by `build-custom-vllm.sh` |
| transformers | PyPI | `{ git = "...", rev = "c58e711..." }` | Qwen3.5 bug fix commit |

### `override-dependencies`

| Override | Reason |
|----------|--------|
| `"torch==2.10.0"` | Updated from 2.9.0; vLLM v0.16 requires 2.10 |
| `"flashinfer-python>=0.5.0,<0.7.0"` | Relax megatron-core's `<0.6` ceiling; allow each split to resolve independently |
| `"opentelemetry-api>=1.33.1"` | Relax megatron-core's `<1.34` ceiling; needed for protobuf 6.x compatibility with ray |
| `"nvidia-cutlass-dsl>=4.2.1"` | Allow cutlass 4.4.0 needed by flashinfer 0.6.4 |
| `"transformers>=4.57.1"` | Relax version constraints from other workspace members |

### `conflicts`

| Conflict | Reason |
|----------|--------|
| `[{ extra = "vllm" }, { extra = "mcore" }]` | flashinfer version requirements are mutually exclusive; these extras never share a venv |

---

## Key Concepts Learned

### uv workspace members always participate in resolution
Even when an extra is excluded via `conflicts`, workspace members listed in `[tool.uv.workspace].members` contribute their constraints globally. Use `override-dependencies` to relax workspace member constraints that block resolution.

### `override-dependencies` apply globally
An override like `"flashinfer-python==0.6.4"` forces that version in ALL resolution splits. Use range overrides (e.g., `>=0.5.0,<0.7.0`) to let each split resolve independently.

### `conflicts` declare mutual exclusivity, not compatibility
The `conflicts` section tells uv that two extras will **never** be installed together, allowing separate resolutions.

### Precompiled wheels must match torch version
vLLM's precompiled `.so` files are linked against a specific torch C++ ABI. Mismatched torch versions cause `ImportError: undefined symbol` errors from `c10::*` or `at::*` namespaces.

### Driver venv vs worker venvs
- `/opt/nemo_rl_venv` — driver environment (base deps, controlled by `UV_PROJECT_ENVIRONMENT`)
- `/opt/ray_venvs/<actor_fqn>/` — worker environments (per-actor-type deps)
- `uv run --extra vllm` syncs into the driver venv (for ad-hoc testing)
- `NRL_FORCE_REBUILD_VENVS=true` only rebuilds worker venvs during NeMo RL runs

---

## Useful Debugging Commands

```bash
# Check lock file resolution for a specific package
grep <package> uv.lock

# Verbose lock resolution (find what's blocking a package)
uv lock -v 2>&1 | grep -i <package>

# Test if a package version is installable
uv pip install <package>==<version> --dry-run

# Check installed version in a specific venv
source /opt/ray_venvs/<actor_fqn>/bin/activate
uv pip list | grep <package>

# Force install a package into a worker venv (for testing)
source /opt/ray_venvs/<actor_fqn>/bin/activate
uv pip install <package>==<version>

# Rebuild lock file from scratch
rm uv.lock && uv lock

# Force reinstall a specific package in the driver venv
uv sync --extra vllm --reinstall-package <package>

# Demangle C++ symbols from ImportError
c++filt <mangled_symbol>

# Check torch version
python -c "import torch; print(torch.__version__)"

# Check flashinfer version
uv run --extra vllm python -c "import flashinfer; print(flashinfer.__version__)"

# Check vLLM commit in 3rdparty
cd 3rdparty/vllm && git log --oneline -1
```
