import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from gan2dhg import kp6_well as W
# The density sweep behind the null prediction of Fig. 1(c).  Masses are taken
# from a LOCAL derivative at each branch's own Fermi wavevector, not from the
# coarse wavevector grid carried through the self-consistent loop, because the
# light branch is strongly non-parabolic there and the coarse derivative
# overstates its mass by about a quarter.
out=[]
for ns in (2.0e13, 3.0e13, 4.0e13, 4.6e13, 5.5e13, 7.0e13):
    t0=time.time()
    s = W.self_consistent(p_s_cm2=ns, N=161, n_kt=16, max_iter=60, mix=0.6,
                          tol=2e-5, verbose=False)
    rec={'p_s_cm2':ns,'EF_meV':1000*s['EF'],'centroid_nm':W.centroid(s),
         'rms_nm':W.rms_width(s),'converged':bool(s['converged']),'bands':[]}
    for b in range(6):
        n=s['per_subband_nm2'][b]
        if n<=0: continue
        kF,m=W.cyclotron_mass_refined(s,b)
        rec['bands'].append({'n_cm2':float(n*1e14),'kF':float(kF),'m':float(m),
                             'edge_meV':float(1000*(s['E_of_k'][0,b]-s['E_of_k'][0,0]))})
    out.append(rec)
    print(f"ns={ns:.1e} done {time.time()-t0:.0f}s conv={s['converged']}", flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__),'..','results','well_sweep.json'),'w'), indent=1)
print("ALL DONE")
