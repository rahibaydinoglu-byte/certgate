import numpy as np
rng=np.random.default_rng(0)
d,R=5,10
C=rng.uniform(0,1,(R,d)); S=rng.uniform(0.15,0.4,(R,d))
def g(x): return -(((x-C)**2)/(2*S**2)).sum(1)
def wbar(x):
    a=g(x); a-=a.max(); e=np.exp(a); return e/e.sum()
def bound(x,eps):
    U=(eps*(np.abs(x-C)+eps)/S**2).sum(1)   # |grad g_r . delta| upper bound over the eps-box
    return 2*U.max()
viol=0; ratios=[]; cert=0; certok=0; N=2000
for i in range(N):
    x=rng.uniform(0,1,d); eps=rng.choice([0.005,0.01,0.02,0.05])
    w=wbar(x); B=bound(x,eps)
    # empirical worst case: random corners + random-sign search
    best=0
    for _ in range(300):
        dl=eps*rng.choice([-1,1],d)*rng.uniform(0.5,1,d)
        best=max(best,np.abs(wbar(x+dl)-w).sum())
    viol+= best>B+1e-12; ratios.append(best/B)
    srt=np.sort(w)[::-1]; m=srt[0]-srt[1]
    if m>B:
        cert+=1
        top=w.argmax()
        ok=all(wbar(x+eps*rng.choice([-1,1],d)*rng.uniform(0,1,d)).argmax()==top for _ in range(300))
        certok+=ok
print(f"bound violations: {viol}/{N}")
print(f"tightness empirical/bound: median {np.median(ratios):.3f}, p90 {np.percentile(ratios,90):.3f}")
print(f"top-1 rule certified: {cert}/{N}; certified cases empirically stable: {certok}/{cert}")

# Tighter bound: exact range of separable quadratic g_r over the eps-box, then softmax interval bounds
def g_range(x,eps):
    lo=x-eps; hi=x+eps
    # per-dim min of (z-c)^2 over [lo,hi]
    dmin=np.where((C>=lo)&(C<=hi),0,np.minimum((lo-C)**2,(hi-C)**2))
    dmax=np.maximum((lo-C)**2,(hi-C)**2)
    L=-(dmax/(2*S**2)).sum(1); U=-(dmin/(2*S**2)).sum(1)
    return L,U
def wbar_interval(x,eps):
    L,U=g_range(x,eps); R_=len(L)
    lo=np.empty(R_); hi=np.empty(R_)
    for r in range(R_):
        o=np.arange(R_)!=r
        m=max(U.max(),L.max())
        lo[r]=np.exp(L[r]-m)/(np.exp(L[r]-m)+np.exp(U[o]-m).sum())
        hi[r]=np.exp(U[r]-m)/(np.exp(U[r]-m)+np.exp(L[o]-m).sum())
    return lo,hi
rng=np.random.default_rng(1); cert2=0; certok2=0; viol2=0
for i in range(N):
    x=rng.uniform(0,1,d); eps=rng.choice([0.005,0.01,0.02,0.05])
    w=wbar(x); lo,hi=wbar_interval(x,eps)
    for _ in range(100):
        w2=wbar(np.clip(x+eps*rng.uniform(-1,1,d),x-eps,x+eps))
        viol2+= np.any(w2<lo-1e-12) or np.any(w2>hi+1e-12)
    top=w.argmax(); o=np.arange(R)!=top
    if lo[top]>hi[o].max():
        cert2+=1
        certok2+=all(wbar(x+eps*rng.uniform(-1,1,d)).argmax()==top for _ in range(300))
print(f"interval bound violations: {viol2}/{N*100}")
print(f"top-1 rule certified (interval bound): {cert2}/{N}; empirically stable: {certok2}/{cert2}")
