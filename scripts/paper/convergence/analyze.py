"""Convergence analysis of stored histories (read-only on results_5seed)."""
import glob, json, numpy as np, pandas as pd
OUT = 'results_convergence'
M = ['ga', 'bpso', 'sa', 'gwo']
LABEL = {'ga': 'GA', 'bpso': 'BPSO', 'sa': 'SA', 'gwo': 'GWO'}
# nominal oracle lookups at which entry i is recorded (see REPORT.md section 1)
def evals_at(m, n):
    i = np.arange(n)
    if m == 'ga':   return 50 + 50 * (i + 1)      # initial pop (50) + 50 offspring/generation, upper bound
    if m == 'bpso': return 50 * (i + 1)           # swarm of 50 evaluated per iteration, then logged
    if m == 'sa':   return i + 1                  # entry 0 = initial state (1 eval), then 1 per move
    if m == 'gwo':  return 50 * (i + 1)           # entry 0 = initial pack (50), then 50 per iteration
TOTAL = {'ga': 2050, 'bpso': 2000, 'sa': 2001, 'gwo': 2050}
rows, curves, ver = [], {m: [] for m in M}, []
grid = np.linspace(0, 1, 201)
for m in M:
    for f in sorted(glob.glob(f'results_5seed/*_{m}_seed*.parquet')):
        r = pd.read_parquet(f).iloc[0]
        h = np.asarray(r.convergence_history, float)
        ev = evals_at(m, len(h)); frac_budget = ev / TOTAL[m]
        mono = bool(np.all(np.diff(h) <= 1e-12))
        ver.append(dict(method=m, dataset=r.dataset, classifier=r.classifier, seed=int(r.seed), len=len(h),
                        last=h[-1], val_fitness=float(r.val_fitness), match=abs(h[-1] - float(r.val_fitness)) <= 1e-9,
                        monotone=mono))
        tot = h[0] - h[-1]
        rec = dict(method=m, dataset=r.dataset, classifier=r.classifier, seed=int(r.seed), total_impr=tot)
        if tot <= 1e-12:
            rec['flat'] = True
        else:
            rec['flat'] = False
            prog = (h[0] - h) / tot
            for p in (0.5, 0.8, 0.9, 0.95):
                k = int(np.argmax(prog >= p - 1e-12))
                rec[f'ev{int(p*100)}'] = ev[k]; rec[f'fb{int(p*100)}'] = frac_budget[k]
            last10 = frac_budget >= 0.9
            k0 = np.argmax(last10)  # first index in last 10% of budget
            rec['improving_last10'] = bool(h[-1] < h[max(k0 - 1, 0)] - 1e-12)
            curves[m].append(np.interp(grid, frac_budget, prog, left=0.0))
            if m == 'ga':
                g20 = 19  # entry 19 = after generation 20
                rec['ga_prog_gen20'] = prog[g20]
        rows.append(rec)
V = pd.DataFrame(ver); V.to_csv(f'{OUT}/verification.csv', index=False)
R = pd.DataFrame(rows); R.to_csv(f'{OUT}/per_run_milestones.csv', index=False)
np.savez(f'{OUT}/curves.npz', grid=grid, **{m: np.array(curves[m]) for m in M})
print('VERIFY: runs', len(V), '| last==val_fitness', int(V.match.sum()), '| mismatches', int((~V.match).sum()),
      '| non-monotone', int((~V.monotone).sum()))
print(V.groupby('method').agg(n=('len', 'size'), length=('len', 'first'), match=('match', 'sum'), monotone=('monotone', 'sum')))
if (~V.match).any(): print(V[~V.match].head(20).to_string())
print('\nflat runs (no improvement) per method:', R.groupby('method').flat.sum().to_dict())
def summ(g):
    out = {'n': len(g)}
    for p in (50, 80, 90, 95):
        q = g[f'ev{p}'].quantile([.25, .5, .75]).values
        out[f'{p}%'] = f"{q[1]:.0f} [{q[0]:.0f}-{q[2]:.0f}]"
    out['still improving last 10%'] = f"{100*g.improving_last10.mean():.0f}%"
    return pd.Series(out)
NF = R[~R.flat]
T1 = NF.groupby('method').apply(summ).reindex(M); T1.index = [LABEL[m] for m in M]
print('\nPOOLED (evaluations to reach X% of total improvement: median [IQR])'); print(T1.to_string())
T1.to_csv(f'{OUT}/milestones_pooled.csv')
T2 = NF.groupby(['method', 'dataset']).apply(summ).reset_index(); T2['method'] = T2.method.map(LABEL)
T2.to_csv(f'{OUT}/milestones_by_dataset.csv', index=False)
print('\nBY DATASET'); print(T2.to_string(index=False))
ga = NF[NF.method == 'ga']
print('\nGA: share of runs with >=80%/90% of improvement by generation 20:',
      round((ga.ga_prog_gen20 >= 0.8).mean(), 3), round((ga.ga_prog_gen20 >= 0.9).mean(), 3),
      '| median progress at gen 20', round(ga.ga_prog_gen20.median(), 3), 'n', len(ga))
print('GA by dataset, share >=90% by gen 20:', ga.groupby('dataset').ga_prog_gen20.apply(lambda s: round((s >= .9).mean(), 2)).to_dict())
print('\nstill improving in last 10% by dataset:'); print(NF.pivot_table(index='method', columns='dataset', values='improving_last10').round(2))
print('\nmedian fraction of budget to 90%:', NF.groupby('method').fb90.median().round(3).to_dict())
