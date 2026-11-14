# Recovered verbatim from the inline command run at 2026-11-11T05:49:44.328Z (session transcript); its stdout was saved as results_wdbc_features/step0_names.txt.
# Run from the repository root: .venv/bin/python scripts/paper/wdbc_agreement/step0_names.py | tee results_wdbc_features/step0_names.txt
import sys; sys.path.insert(0,'src')
import numpy as np
from sklearn.datasets import load_breast_cancer
from metaheuristic_xai.datasets import load_dataset
ref=list(load_breast_cancer().feature_names)
names=None
for seed in [42,43,44,45,46]:
    b=load_dataset('wdbc',random_state=seed); fn=[str(x) for x in b.feature_names]
    print('seed',seed,'n_features',len(fn),'X_train cols',b.X_train.shape[1],'identical to sklearn order:',fn==ref)
    if names is None: names=fn
    assert fn==names
for i,n in enumerate(names): print(i,n)
for t in ['mean texture','worst area','worst smoothness','worst radius','worst perimeter','worst texture','mean smoothness']:
    print('INDEX',t,names.index(t))
