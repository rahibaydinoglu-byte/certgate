import numpy as np
rng=np.random.default_rng(5)
def cvt(R,d,iters=60,n=40000):
    P=rng.uniform(0,1,(n,d)); C=rng.uniform(0,1,(R,d))
    for _ in range(iters):
        lab=((P[:,None,:]-C[None])**2).sum(2).argmin(1)
        for r in range(R):
            m=lab==r
            if m.any(): C[r]=P[m].mean(0)
    return C
def uncert_frac(C,eps,n=40000):
    d=C.shape[1]; X=rng.uniform(0,1,(n,d))
    W=C; b=-0.5*(C**2).sum(1)          # Voronoi = linear gating
    G=X@W.T+b; t=G.argmax(1)
    marg=G[np.arange(n),t][:,None]-G  # g_t - g_s
    D=np.abs(W[t][:,None,:]-W[None]).sum(2)   # ||w_t-w_s||_1
    # box clipped to [0,1]^d: exact min of linear fn over clipped box
    lo=np.clip(X-eps,0,1); hi=np.clip(X+eps,0,1)
    dW=W[t][:,None,:]-W[None]                  # n x R x d
    zmin=np.where(dW>0,lo[:,None,:],hi[:,None,:])
    mins=(dW*zmin).sum(2)+(b[t][:,None]-b[None])
    mins[np.arange(n),t]=np.inf
    vols=np.bincount(t,minlength=len(C))/n
    return (mins.min(1)<=0).mean(), vols
for d in (2,3):
    print(f"d={d}, eps=0.01")
    Rs=[2,4,8,16,32,64]; fr=[]
    for R in Rs:
        f,v=uncert_frac(cvt(R,d),0.01); fr.append(f)
        print(f"  R={R:3d}: uncertified={f:.4f}  ratio to R^(1/d)={f/R**(1/d):.4f}  cell-vol min/max={v.min()*R:.2f}/{v.max()*R:.2f}")
    slope=np.polyfit(np.log(Rs[1:]),np.log(fr[1:]),1)[0]
    print(f"  log-log slope (R>=4): {slope:.3f}  (theory lower-bound rate 1/d={1/d:.3f})")
# linearity in eps
C=cvt(16,2); print("eps-linearity, d=2,R=16:",[f"{e}:{uncert_frac(C,e)[0]:.4f}" for e in (0.0025,0.005,0.01,0.02)])
