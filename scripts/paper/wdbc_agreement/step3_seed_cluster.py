# Recovered verbatim from the inline command run at 2026-11-11T05:51:08.184Z (session transcript); its stdout was saved as results_wdbc_features/step2_seed_cluster.txt.
# Run from the repository root: .venv/bin/python scripts/paper/wdbc_agreement/step3_seed_cluster.py | tee results_wdbc_features/step2_seed_cluster.txt
import pandas as pd, numpy as np
from math import comb
w=pd.read_pickle('results_wdbc_features/wdbc_masks.pkl'); D=30
R=[1,23,24]; G=[[20,22,23],[1,21],[4,24]]
w['key']=w.idx.map(tuple)
print('distinct masks per (method, seed) across the 5 classifiers (1 = classifier-independent):')
print(w.groupby(['method','seed']).key.nunique().groupby('method').max().to_dict())
def ph(g,s): return 1-comb(D-g,s)/comb(D,s)
w['dS']=w.idx.map(lambda x: sum(i in x for i in R)-3*len(x)/D)
w['dR']=w.idx.map(lambda x: sum(any(i in x for i in g) for g in G)-sum(ph(len(g),len(x)) for g in G))
rng=np.random.default_rng(7); seeds=np.array(sorted(w.seed.unique()))
print('\nSUPPLEMENTARY seed-cluster bootstrap (resample the 5 seeds with replacement, all runs of a seed kept), 10,000 resamples')
out=[]
for m,g in w.groupby('method'):
    per=g.groupby('seed')[['dS','dR']].mean().reindex(seeds).values
    b=per[rng.integers(0,5,(10000,5))].mean(1)
    lo=np.percentile(b,2.5,axis=0); hi=np.percentile(b,97.5,axis=0)
    out.append((m,round(per[:,0].mean(),3),round(lo[0],3),round(hi[0],3),round(per[:,1].mean(),3),round(lo[1],3),round(hi[1],3)))
T=pd.DataFrame(out,columns=['method','strict_diff','s_lo','s_hi','relaxed_diff','r_lo','r_hi']); print(T.to_string(index=False))
T.to_csv('results_wdbc_features/metrics_seed_cluster_ci.csv',index=False)
