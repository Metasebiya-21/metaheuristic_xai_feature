# Recovered verbatim from the inline command run at 2026-11-11T05:50:42.826Z (session transcript); its stdout was saved as results_wdbc_features/step2_metrics.txt.
# Run from the repository root: .venv/bin/python scripts/paper/wdbc_agreement/step2_metrics.py | tee results_wdbc_features/step2_metrics.txt
import pandas as pd, numpy as np
from math import comb
from sklearn.datasets import load_breast_cancer
w=pd.read_pickle('results_wdbc_features/wdbc_masks.pkl')
D=30; names=list(load_breast_cancer().feature_names)
R={'mean texture':1,'worst area':23,'worst smoothness':24}
G={'size':[20,22,23],'texture':[1,21],'smoothness':[4,24]}
for k,v in G.items(): assert [names[i] for i in v], k
print('groups',{k:[names[i] for i in v] for k,v in G.items()})
def p_hit(g,s): return 1-comb(D-g,s)/comb(D,s) if s<=D else 1.0
rng=np.random.default_rng(20261111)
def boot(x):
    x=np.asarray(x,float); n=len(x); idx=rng.integers(0,n,(10000,n)); m=x[idx].mean(1)
    return np.percentile(m,[2.5,97.5])
order=['ga','bpso','sa','gwo','nsga2','shap','lime','rfe','lasso','boruta','full_features']
label={'ga':'GA','bpso':'BPSO','sa':'SA','gwo':'GWO','nsga2':'NSGA-II','shap':'SHAP','lime':'LIME','rfe':'RFE','lasso':'LASSO','boruta':'Boruta','full_features':'Full feature set'}
rows=[]
for m in order:
    g=w[w.method==m]; S=[set(x) for x in g.idx]; s=np.array([len(x) for x in S])
    strict=np.array([sum(i in x for i in R.values()) for x in S]); exp_s=3*s/D; dS=strict-exp_s
    rel=np.array([sum(any(i in x for i in v) for v in G.values()) for x in S])
    exp_r=np.array([sum(p_hit(len(v),k) for v in G.values()) for k in s]); dR=rel-exp_r
    ciS=boot(dS); ciR=boot(dR)
    row=dict(method=label[m],n=len(g),mean_s=s.mean(),strict_mean=strict.mean(),
        **{f'incl_{k}':np.mean([v in x for x in S]) for k,v in R.items()},
        all3=np.mean(strict==3),strict_expected=exp_s.mean(),strict_diff=dS.mean(),strict_ci_lo=ciS[0],strict_ci_hi=ciS[1],
        relaxed_mean=rel.mean(),relaxed_expected=exp_r.mean(),relaxed_diff=dR.mean(),relaxed_ci_lo=ciR[0],relaxed_ci_hi=ciR[1],
        **{f'grp_{k}':np.mean([any(i in x for i in v) for x in S]) for k,v in G.items()})
    rows.append(row)
T=pd.DataFrame(rows); T.to_csv('results_wdbc_features/metrics_by_method.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
print(T.round(3).to_string(index=False))
# top-10
top={}
for m in ['ga','bpso','boruta','shap','rfe']:
    g=w[w.method==m]; c=pd.Series([i for x in g.idx for i in x]).value_counts()/len(g)
    t=c.sort_values(ascending=False,kind='stable').head(10)
    top[m]=[(names[i],round(v,2)) for i,v in t.items()]
    print('\nTOP10',label[m]); [print(f'  {n:25s} {v:.2f}') for n,v in top[m]]
pd.DataFrame([(label[m],r+1,n,v) for m in top for r,(n,v) in enumerate(top[m])],columns=['method','rank','feature','inclusion_rate']).to_csv('results_wdbc_features/top10_features.csv',index=False)
# per-classifier breakdown of strict hits (context)
w['strict']=w.idx.map(lambda x: sum(i in x for i in R.values()))
print('\nstrict mean by method x classifier'); print(w.pivot_table(index='method',columns='classifier',values='strict').round(2))
# correlations on full raw data
X=pd.DataFrame(load_breast_cancer().data,columns=names)
for a,b in [('worst area','worst radius'),('worst area','worst perimeter'),('mean texture','worst texture'),('mean smoothness','worst smoothness'),('worst radius','worst perimeter')]:
    print('PEARSON',a,'|',b,round(X[a].corr(X[b]),4))
