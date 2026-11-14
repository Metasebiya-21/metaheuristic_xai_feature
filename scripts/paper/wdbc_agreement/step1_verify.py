# Recovered verbatim from the inline command run at 2026-11-11T05:50:18.268Z (session transcript); its stdout was saved as results_wdbc_features/step1_verify.txt.
# Run from the repository root: .venv/bin/python scripts/paper/wdbc_agreement/step1_verify.py | tee results_wdbc_features/step1_verify.txt
import pandas as pd, numpy as np, re, os
d=pd.read_csv('results_5seed/all_runs.csv')
w=d[d.dataset=='wdbc'].copy()
print('wdbc rows',len(w),'| methods',w.method.nunique(),'| classifiers',w.classifier.nunique(),'| seeds',sorted(int(s) for s in w.seed.unique()))
def parse(x):
    if isinstance(x,(list,np.ndarray)): return sorted(int(i) for i in x)
    s=str(x).strip()
    if s in ('[]',''): return []
    vals=re.findall(r'np\.int64\((\d+)\)',s) or re.findall(r'(?<![\w.])(\d+)(?![\w.(])',s)
    return sorted(int(v) for v in vals)
w['idx']=w.selected_indices.map(parse)
ok=0; bad=[]
for _,r in w.iterrows():
    f=f"results_5seed/wdbc_{r.classifier}_{r.method}_seed{r.seed}.parquet"
    if not os.path.exists(f): bad.append((f,'missing parquet')); continue
    q=pd.read_parquet(f).iloc[0]; qi=parse(q.selected_indices)
    checks=dict(acc=abs(float(q.accuracy)-float(r.accuracy))<=1e-9,
                nsel=int(q.selected_features)==int(r.selected_features),
                len_eq_nsel=len(r.idx)==int(r.selected_features),
                idx_equal=qi==r.idx)
    if all(checks.values()): ok+=1
    else: bad.append((f,checks))
print('verified',ok,'of',len(w),'rows (accuracy |diff|<=1e-9, n_selected equal, len(indices)==n_selected, index sets equal to parquet)')
for b in bad: print('MISMATCH',b)
allidx=[i for l in w.idx for i in l]; print('index range',min(allidx),max(allidx))
print('rows per method',w.groupby('method').size().to_dict())
w[['dataset','classifier','method','seed','accuracy','selected_features','idx']].to_pickle('results_wdbc_features/wdbc_masks.pkl')
