"""Diagnostic Probe A (v2): does per-token RHT / SR perturb the FORWARD output?

The per-token RHT/SR flags are read at IMPORT time (dataclass field defaults
`per_token_rht = os.getenv(...)`), so they must be set BEFORE `import
transformer_engine`. This driver therefore runs ONE FRESH CHILD PROCESS per flag
config (env set before python starts), each saving its forward output + wgrad to a
file; the parent then diffs them.

Decision: if RHT/SR change the forward output beyond the deterministic noise floor,
the forward cast is contaminated (root cause). If not, the cause is downstream.
wgrad diff is the sanity that the flags actually took effect this time.
"""
import os, sys, subprocess, json

CFGS = [("base", 0, 0), ("base2", 0, 0), ("rht", 1, 0), ("sr", 0, 1), ("rhtsr", 1, 1)]
OUT = "/tmp/probeA"

if os.environ.get("PROBE_CHILD") == "1":
    import torch
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe
    tag = os.environ["PROBE_TAG"]
    torch.manual_seed(0)  # identical module + input across children
    dev, dt = torch.device("cuda"), torch.bfloat16
    M, N, K = 512, 1024, 512
    lin = te.Linear(K, N, bias=False, params_dtype=dt).to(dev)
    for p in lin.parameters():
        with torch.no_grad():
            p.copy_(p.to(dt))
    x = torch.randn(M, K, dtype=dt, device=dev)
    r = recipe.NVFP4BlockScaling()
    # record the recipe's resolved flags so we PROVE the env took effect
    flags = dict(disable_rht=bool(r.disable_rht),
                 disable_sr=bool(r.disable_stochastic_rounding),
                 per_token=bool(r.nvfp4_per_token()))
    xin = x.clone().requires_grad_(True)
    with te.fp8_autocast(enabled=True, fp8_recipe=r):
        y = lin(xin)
    y.sum().backward()
    torch.save({"y": y.detach().float().cpu(),
                "g": lin.weight.grad.detach().float().cpu(),
                "flags": flags}, f"{OUT}_{tag}.pt")
    print(json.dumps({"tag": tag, **flags}))
    sys.exit(0)

# ---- parent ----
os.makedirs(os.path.dirname(OUT), exist_ok=True)
for tag, rht, sr in CFGS:
    env = {**os.environ, "PROBE_CHILD": "1", "PROBE_TAG": tag,
           "NVTE_NVFP4_PER_TOKEN": "1",
           "NVTE_NVFP4_PER_TOKEN_RHT": str(rht),
           "NVTE_NVFP4_PER_TOKEN_SR": str(sr)}
    p = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True)
    print(f"[{tag}] rc={p.returncode} {p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr.strip()[-200:]}")

import torch
d = {tag: torch.load(f"{OUT}_{tag}.pt") for tag, _, _ in CFGS}
def rel(a, b):
    return (a - b).abs().max().item(), (a - b).norm().item() / (b.norm().item() + 1e-12)
base = d["base"]["y"]; baseg = d["base"]["g"]
print("\n=== resolved recipe flags per child (proves env took effect) ===")
for tag, _, _ in CFGS:
    print(f"  {tag:6s}: {d[tag]['flags']}")
print(f"\n{'config':7s} | {'FWD maxΔ':>12s} {'FWD relL2':>12s} | {'WGRAD relL2':>12s}")
for tag, _, _ in CFGS:
    fm, fr = rel(d[tag]["y"], base)
    _, gr = rel(d[tag]["g"], baseg)
    print(f"{tag:7s} | {fm:12.4g} {fr:12.4g} | {gr:12.4g}")
print("\nREAD: base2 FWD relL2 = noise floor (should be ~0, deterministic fwd).")
print("      rht/sr/rhtsr FWD relL2 >> floor => flag CONTAMINATES the forward (root cause).")
print("      WGRAD relL2 should be > 0 for rht/sr (proves the flags took effect this run).")
print("PROBE_A_DONE")
