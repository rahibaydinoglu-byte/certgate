import numpy as np, itertools
rng=np.random.default_rng(4)
def make(R,d):
    return rng.uniform(0,1,(R,d)), rng.uniform(0.15,0.4,(R,d))
def logmu(z,c,s): return -(z-c)**2/(2*s**2)
def g_min(x,C,S): return logmu(x,C,S).min(1)          # min t-norm, log firing

def pair_min_mintnorm(r,s,lo,hi,C,S):
    d=len(lo); best=np.inf
    phimin=np.array([ -logmu(np.clip(C[s,i],lo[i],hi[i]),C[s,i],S[s,i]) for i in range(d)])
    for j in range(d):
        K=np.max(np.delete(phimin,j)) if d>1 else -np.inf
        A=lambda z: logmu(z,C[r,j],S[r,j]); phi=lambda z: -logmu(z,C[s,j],S[s,j])
        F=lambda z: A(z)+np.maximum(phi(z),K)
        cand=[lo[j],hi[j]]
        if K>0:
            rt=S[s,j]*np.sqrt(2*K); cand+= [C[s,j]-rt, C[s,j]+rt]
        a=1/(2*S[s,j]**2)-1/(2*S[r,j]**2)
        if a>0:
            b=C[r,j]/S[r,j]**2-C[s,j]/S[s,j]**2; cand.append(-b/(2*a))
        cand=np.clip(np.array(cand),lo[j],hi[j])
        best=min(best,F(cand).min())
    return best

# 1) exactness vs brute-force grid, d=2 (grid min of the true function)
R,d=6,2; C,S=make(R,d); errs=[]
for t in range(200):
    x=rng.uniform(0,1,d); eps=rng.choice([0.02,0.05,0.1]); lo,hi=x-eps,x+eps
    r,s=rng.choice(R,2,replace=False)
    ex=pair_min_mintnorm(r,s,lo,hi,C,S)
    gx=np.linspace(lo[0],hi[0],401); gy=np.linspace(lo[1],hi[1],401)
    Z=np.array(np.meshgrid(gx,gy)).reshape(2,-1).T
    G=logmu(Z[:,None,:],C,S).min(2)
    bf=(G[:,r]-G[:,s]).min()
    errs.append(bf-ex)
errs=np.array(errs)
print(f"d=2 exact-vs-grid: min(bf-exact)={errs.min():.2e} (must be >= ~0), max={errs.max():.2e} (grid resolution gap)")

# 2) soundness + witness, d=5
R,d,N=10,5,1000; C,S=make(R,d); cert=viol=0
for t in range(N):
    x=rng.uniform(0,1,d); eps=rng.choice([0.005,0.01,0.02,0.05]); lo,hi=x-eps,x+eps
    gx=g_min(x,C,S); top=gx.argmax()
    m=min(pair_min_mintnorm(top,s,lo,hi,C,S) for s in range(R) if s!=top)
    if m>0:
        cert+=1
        viol+=any(g_min(x+eps*rng.uniform(-1,1,d),C,S).argmax()!=top for _ in range(200))
print(f"min t-norm d=5: certified {cert}/{N}, sampled violations among certified {viol}")
