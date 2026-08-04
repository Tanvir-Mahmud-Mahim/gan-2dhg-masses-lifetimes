import os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from gan2dhg import kp6_well as W
t0=time.time()
sol = W.self_consistent(p_s_cm2=4.6e13, N=161, n_kt=16, max_iter=90, mix=0.6,
                        tol=2e-5, verbose=True)
def _c(v):
    import numpy as _n
    if isinstance(v,_n.ndarray): return v.tolist()
    if isinstance(v,(_n.bool_,)): return bool(v)
    if isinstance(v,(_n.floating,)): return float(v)
    if isinstance(v,(_n.integer,)): return int(v)
    return v
out = {k:_c(v) for k,v in sol.items() if k!='strain'}
out['strain']=list(sol['strain']); out['centroid_nm']=W.centroid(sol); out['rms_nm']=W.rms_width(sol)
out['masses']=[]
for s in range(6):
    n = sol['per_subband_nm2'][s]
    if n<=0: continue
    kF,m = W.cyclotron_mass_refined(sol,s)
    out['masses'].append({'subband':s,'n_cm2':float(n*1e14),'kF_per_nm':float(kF),
                          'm_CR':float(m),'E_edge_meV':float(1000*(sol['E_of_k'][0,s]-sol['E_of_k'][0,0]))})
json.dump(out, open(os.path.join(os.path.dirname(__file__),'..','results','well.json'),'w'), indent=1)
print(f"DONE {time.time()-t0:.0f} s converged={sol['converged']}")
