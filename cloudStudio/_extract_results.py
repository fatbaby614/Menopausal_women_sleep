import json, os

base = r"D:\TanHuangWork\Menopausal_women_sleep\cloudStudio\experiment_results"
out = []

files = [
    ('Sleep-EDF', 'paper_results_sleep-edf_20260511_204059.json'),
    ('ISRUC', 'paper_results_isruc_20260511_210349.json'),
    ('DREAMS', 'paper_results_dreams_20260511_212236.json'),
]

for ds_name, fn in files:
    path = os.path.join(base, fn)
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    
    out.append(f"\n{'='*60}")
    out.append(f"  {ds_name}")
    out.append(f"{'='*60}")
    
    gc = d.get('group_comparison', {})
    for g in ['menopausal_women', 'young_women', 'young_men']:
        if g in gc:
            r = gc[g]
            acc = r['mean_accuracy'] * 100
            f1 = r['mean_f1'] * 100
            n = r['n_subjects']
            ne = r['total_epochs']
            out.append(f"\n  [{r['name']}] n={n}, epochs={ne}")
            out.append(f"    Acc: {acc:.1f}%  F1: {f1:.1f}%")
            cp = r.get('class_percentages', {})
            out.append(f"    Stage%: W={cp.get('Wake',0):.1f} N1={cp.get('N1',0):.1f} N2={cp.get('N2',0):.1f} N3={cp.get('N3',0):.1f} REM={cp.get('REM',0):.1f}")
    
    scm = d.get('sleep_clinical_metrics', {})
    for g in ['menopausal_women', 'young_women', 'young_men']:
        if g in scm:
            info = scm[g]
            sm = info.get('summary', {})
            st = info.get('stage_pct_tst', {})
            out.append(f"\n  --- {g} Clinical ---")
            for m in ['TST_min', 'sleep_efficiency_pct', 'WASO_min', 'SOL_min', 'transition_rate_per_hour']:
                v = sm.get(m, {})
                mu = v.get('mean', '-')
                sd = v.get('std', '-')
                vals = v.get('values', [])
                out.append(f"    {m}: mean={mu}, std={sd}, n={len(vals)}")
            for s in ['N1', 'N2', 'N3', 'REM']:
                v = st.get(s, {})
                out.append(f"    {s}%TST: mean={v.get('mean','-')}")

with open(r"D:\TanHuangWork\Menopausal_women_sleep\cloudStudio\_results_summary.txt", 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print("Summary written.")
