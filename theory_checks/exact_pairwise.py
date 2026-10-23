import numpy as np
rng=np.random.default_rng(3); d,R,N=5,10,2000
C=rng.uniform(0,1,(R,d)); S=rng.uniform(0.15,0.4,(R,d))
def g(x): return -(((x-C)**2)/(2*S**2)).sum(1)
def soft(a): a=a-a.max(); e=np.exp(a); return e/e.sum()

def pair_min(r,s,lo,hi,want_arg=False):
    # min over box of h(z)=g_r(z)-g_s(z), separable: sum_j a_j z^2+b_j z+c_j
    a=1/(2*S[s]**2)-1/(2*S[r]**2)
    b=C[r]/S[r]**2-C[s]/S[s]**2
    c=-C[r]**2/(2*S[r]**2)+C[s]**2/(2*S[s]**2)
    cand=[lo,hi]
    v=np.where(np.abs(a)>1e-15,-b/(2*np.where(np.abs(a)>1e-15,a,1)),lo)
    vin=(a>0)&(v>lo)&(v<hi)
    cand.append(np.where(vin,v,lo))
    Z=np.stack(cand)                       # 3 x d candidates per dim
    H=a*Z**2+b*Z+c
    idx=H.argmin(0); zbest=Z[idx,np.arange(d)]
    val=H.min(0).sum()
    return (val,zbest) if want_arg else val

def pair_max(r,s,lo,hi): return -pair_min(s,r,lo,hi)   # max(g_r-g_s) = -min(g_s-g_r)

def interval_cert(x,eps):
    lo=x-eps; hi=x+eps
    dmin=np.where((C>=lo)&(C<=hi),0,np.minimum((lo-C)**2,(hi-C)**2))
    dmax=np.maximum((lo-C)**2,(hi-C)**2)
    L=-(dmax/(2*S**2)).sum(1); U=-(dmin/(2*S**2)).sum(1)
    t=g(x).argmax(); o=np.arange(R)!=t
    lo_t=1/(1+np.exp(U[o]-L[t]).sum())
    hi_o=np.array([1/(1+np.exp(np.delete(L,s)-U[s]).sum()) for s in np.where(o)[0]])
    return lo_t>hi_o.max(), lo_t

stats=dict(int_cert=0,exact_cert=0,witness_ok=0,witness_n=0,sound_viol=0,pairLB_tighter=0,pairLB_viol=0)
for i in range(N):
    x=rng.uniform(0,1,d); eps=rng.choice([0.005,0.01,0.02,0.05]); lo,hi=x-eps,x+eps
    t=g(x).argmax(); others=[s for s in range(R) if s!=t]
    ic,lo_int=interval_cert(x,eps); stats['int_cert']+=ic
    mins=[pair_min(t,s,lo,hi,True) for s in others]
    exact=min(m[0] for m in mins)>0; stats['exact_cert']+=exact
    if exact:  # soundness check by sampling
        for _ in range(200):
            if g(x+eps*rng.uniform(-1,1,d)).argmax()!=t: stats['sound_viol']+=1; break
    else:      # exactness: the minimiser must be a witness (top-1 changes or ties)
        k=int(np.argmin([m[0] for m in mins])); z=mins[k][1]
        gz=g(z); stats['witness_n']+=1
        stats['witness_ok']+= gz[others[k]]>=gz[t]-1e-9
    # pairwise lower bound on pi_t vs interval bound
    lb_pair=1/(1+np.exp([pair_max(s,t,lo,hi) for s in others]).sum())
    stats['pairLB_tighter']+= lb_pair>=lo_int-1e-12
    for _ in range(50):
        if soft(g(x+eps*rng.uniform(-1,1,d)))[t]<lb_pair-1e-12: stats['pairLB_viol']+=1; break
for k,v in stats.items(): print(k,v)
