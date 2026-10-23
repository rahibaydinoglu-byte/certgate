import numpy as np
rng=np.random.default_rng(2); d,R,N=5,10,2000
W=rng.normal(0,3,(R,d)); b=rng.normal(0,1,R)
def soft(a): a=a-a.max(); e=np.exp(a); return e/e.sum()
def interval(L,U):
    lo=np.empty(R); hi=np.empty(R)
    for r in range(R):
        o=np.arange(R)!=r
        lo[r]=1/(1+np.exp(U[o]-L[r]).sum()); hi[r]=1/(1+np.exp(L[o]-U[r]).sum())
    return lo,hi
viol=cert=ok=0
for i in range(N):
    x=rng.uniform(0,1,d); eps=rng.choice([0.005,0.01,0.02,0.05])
    g=W@x+b; rad=eps*np.abs(W).sum(1); lo,hi=interval(g-rad,g+rad); w=soft(g)
    for _ in range(100):
        w2=soft(W@(x+eps*rng.uniform(-1,1,d))+b)
        viol+=np.any(w2<lo-1e-12) or np.any(w2>hi+1e-12)
    t=w.argmax(); o=np.arange(R)!=t
    if lo[t]>hi[o].max():
        cert+=1; ok+=all(soft(W@(x+eps*rng.uniform(-1,1,d))+b).argmax()==t for _ in range(300))
print(f"linear gating: violations {viol}/{N*100}; top-1 certified {cert}/{N}; empirically stable {ok}/{cert}")
