import os, traceback, torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
torch.manual_seed(0); dev,dt=torch.device("cuda"),torch.bfloat16
M,N,K=512,1024,512
lin=te.Linear(K,N,bias=False,params_dtype=dt).to(dev)
for p in lin.parameters():
    with torch.no_grad(): p.copy_(p.to(dt))
x=torch.randn(M,K,dtype=dt,device=dev)
r=recipe.NVFP4BlockScaling()
print("resolved: disable_rht=",r.disable_rht,"per_token=",r.nvfp4_per_token())
print("=== FORWARD-ONLY (no grad) ===")
try:
    with torch.no_grad(), te.fp8_autocast(enabled=True,fp8_recipe=r):
        y=lin(x)
    print("forward-only: OK, y.absmax=",y.detach().float().abs().max().item())
except Exception:
    print("forward-only: RAISED:\n"+traceback.format_exc())
print("=== FORWARD (grad) + BACKWARD ===")
try:
    xin=x.clone().requires_grad_(True)
    with te.fp8_autocast(enabled=True,fp8_recipe=r):
        y=lin(xin)
    print("  forward(grad-enabled): OK")
    y.sum().backward()
    print("  backward: OK")
except Exception:
    print("fwd+bwd: RAISED:\n"+traceback.format_exc())
