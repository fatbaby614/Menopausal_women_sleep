import json, os

base = r'D:\TanHuangWork\Menopausal_women_sleep\cloudStudio\experiment_results'

out_lines = []

# === SOTA ===
out_lines.append('='*70)
out_lines.append('SOTA COMPARISON')
out_lines.append('='*70)
d = json.load(open(os.path.join(base, 'sota_comparison_20260512_181815.json'), 'r'))
for k in sorted(d):
    v = d[k]
    out_lines.append(f'{k}: Acc={v["accuracy"]*100:.1f}+-{v["accuracy_std"]*100:.1f}%  F1={v["f1_macro"]*100:.1f}+-{v["f1_macro_std"]*100:.1f}%  Time={v["training_time"]:.0f}s  N1_F1={v["f1_per_class_mean"][1]*100:.1f}%')

# === THREE-GROUP ===
for ds, fn in [('Sleep-EDF','paper_results_sleep-edf_20260512_185318.json'),
               ('ISRUC','paper_results_isruc_20260512_190034.json'),
               ('DREAMS','paper_results_dreams_20260512_190656.json')]:
    out_lines.append(f'\n{"="*70}')
    out_lines.append(f'THREE-GROUP: {ds}')
    out_lines.append('='*70)
    d = json.load(open(os.path.join(base, fn), 'r'))
    gc = d['group_comparison']
    scm = d.get('sleep_clinical_metrics', {})
    for g in ['menopausal_women', 'young_women', 'young_men']:
        r = gc[g]
        out_lines.append(f'\n{r["name"]} (n={r["n_subjects"]}, epochs={r["total_epochs"]}):')
        out_lines.append(f'  Acc={r["mean_accuracy"]*100:.1f}+-{r["std_accuracy"]*100:.1f}%  F1={r["mean_f1"]*100:.1f}+-{r["std_f1"]*100:.1f}%')
        if g in scm:
            sm = scm[g]['summary']
            st = scm[g]['stage_pct_tst']
            for m in ['TST_min','sleep_efficiency_pct','WASO_min','SOL_min','transition_rate_per_hour']:
                v = sm.get(m, {})
                out_lines.append(f'  {m}: mean={v.get("mean","-")}, std={v.get("std","-")}')
            for s in ['N1','N2','N3','REM']:
                v = st.get(s, {})
                out_lines.append(f'  {s}%TST: mean={v.get("mean","-")}')

with open(r'D:\TanHuangWork\Menopausal_women_sleep\cloudStudio\_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out_lines))
print('Report written')
