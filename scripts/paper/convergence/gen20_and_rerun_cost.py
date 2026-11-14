# Recovered verbatim from two inline commands (session transcript); their stdout was appended to
# results_convergence/analysis_output.txt after analyze.py. Run from the repository root, after analyze.py:
#   .venv/bin/python scripts/paper/convergence/gen20_and_rerun_cost.py | tee -a results_convergence/analysis_output.txt

# --- part 1, run at 2026-11-11T10:51:04.690Z ---
import pandas as pd
R=pd.read_csv('results_convergence/per_run_milestones.csv'); g=R[(R.method=='ga')&(~R.flat)]
t=g.groupby('dataset').ga_prog_gen20.agg(n='size',ge50=lambda s:(s>=.5).mean(),ge80=lambda s:(s>=.8).mean(),ge90=lambda s:(s>=.9).mean(),median='median').round(2)
print(t); print('pooled ge50',round((g.ga_prog_gen20>=.5).mean(),3))
w=g[g.dataset.isin(['wdbc','ionosphere'])]; print('WDBC+Iono RF only: ge80',round((w[w.classifier=='random_forest'].ga_prog_gen20>=.8).mean(),2),'ge90',round((w[w.classifier=='random_forest'].ga_prog_gen20>=.9).mean(),2),'n',len(w[w.classifier=='random_forest']))

# --- part 2, run at 2026-11-11T10:52:00.271Z ---
import pandas as pd
d=pd.read_csv('results_5seed/all_runs.csv',usecols=['dataset','classifier','method','seed','runtime_seconds'])
s=d[(d.classifier=='random_forest')&(d.dataset.isin(['wdbc','ionosphere','colon']))&(d.method.isin(['ga','bpso','sa','gwo','nsga2']))]
print('runs',len(s)); print(s.groupby(['method','dataset']).runtime_seconds.sum().unstack().round(0))
n=s[s.method=='nsga2']; print('NSGA-II only: runs',len(n),'serial s',round(n.runtime_seconds.sum()),'min',round(n.runtime_seconds.sum()/60,1))
print('all 5 methods: serial min',round(s.runtime_seconds.sum()/60,1),'| with 6 workers ~min',round(s.runtime_seconds.sum()/60/6,1))
