import json

old_path = r'D:\TanHuangWork\Menopausal_women_sleep\Paper\dataUsed\paper_results_sleep-edf_20260511_204059.json'
new_path = r'D:\TanHuangWork\Menopausal_women_sleep\cloudStudio\experiment_results\paper_results_sleep-edf_20260512_141738.json'

old = json.load(open(old_path, 'r'))
new = json.load(open(new_path, 'r'))

out = []
for label, d in [('OLD (20260511)', old), ('NEW (20260512)', new)]:
    out.append(f'\n=== {label} ===')
    gc = d['group_comparison']
    scm = d.get('sleep_clinical_metrics', {})
    for g in ['menopausal_women', 'young_women', 'young_men']:
        r = gc[g]
        acc = r['mean_accuracy'] * 100
        f1 = r['mean_f1'] * 100
        if g in scm:
            n3 = scm[g]['stage_pct_tst']['N3']['mean']
            n1 = scm[g]['stage_pct_tst']['N1']['mean']
            n2 = scm[g]['stage_pct_tst']['N2']['mean']
            rem = scm[g]['stage_pct_tst']['REM']['mean']
            tst = scm[g]['summary']['TST_min']['mean']
            sol = scm[g]['summary']['SOL_min']['mean']
            se = scm[g]['summary']['sleep_efficiency_pct']['mean']
            waso = scm[g]['summary']['WASO_min']['mean']
            trans = scm[g]['summary']['transition_rate_per_hour']['mean']
        else:
            n3 = n1 = n2 = rem = tst = sol = se = waso = trans = '-'
        out.append(f'  {r["name"]}: Acc={acc:.1f}% F1={f1:.1f}%')
        out.append(f'    Clinical: N3={n3} N1={n1} N2={n2} REM={rem} TST={tst} SOL={sol} SE={se} WASO={waso} Trans={trans}')

with open(r'D:\TanHuangWork\Menopausal_women_sleep\cloudStudio\_results_summary.txt', 'w') as f:
    f.write('\n'.join(out))
print('Written comparison')
