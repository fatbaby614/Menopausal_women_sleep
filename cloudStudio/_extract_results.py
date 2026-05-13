# Extract ablation results and three-group metrics
import json, os

base = r"D:\TanHuangWork\Menopausal_women_sleep\cloudStudio"

# Ablation
print("=" * 60)
print("ABLATION RESULTS")
print("=" * 60)
with open(os.path.join(base, 'experiment_results/ablation_results_20260512_151702.json'), 'r') as f:
    abl = json.load(f)
for k in sorted(abl):
    v = abl[k]
    acc = v.get('accuracy', 0) * 100
    f1 = v.get('f1_macro', 0) * 100
    f1c = v.get('f1_per_class_mean', [])
    print(f'{k}: Acc={acc:.1f}%  F1={f1:.1f}%  Wake={f1c[0]:.4f} N1={f1c[1]:.4f} N2={f1c[2]:.4f} N3={f1c[3]:.4f} REM={f1c[4]:.4f}')

print()

# Three-Group Sleep-EDF
print("=" * 60)
print("THREE-GROUP Sleep-EDF")
print("=" * 60)
with open(os.path.join(base, 'experiment_results/paper_results_sleep-edf_20260512_141738.json'), 'r') as f:
    sedf = json.load(f)
gc = sedf['group_comparison']
scm = sedf.get('sleep_clinical_metrics', {})
for g in ['menopausal_women', 'young_women', 'young_men']:
    r = gc[g]
    print(f"\n{r['name']} (n={r['n_subjects']}): Acc={r['mean_accuracy']*100:.1f}%, F1={r['mean_f1']*100:.1f}%")
    # Per-class F1
    if 'f1_per_class' in r:
        print(f"  Per-class F1: W={r['f1_per_class'][0]:.3f} N1={r['f1_per_class'][1]:.3f} N2={r['f1_per_class'][2]:.3f} N3={r['f1_per_class'][3]:.3f} REM={r['f1_per_class'][4]:.3f}")
    # Clinical metrics
    if g in scm:
        sm = scm[g]['summary']
        st = scm[g]['stage_pct_tst']
        for m in ['TST_min', 'sleep_efficiency_pct', 'WASO_min', 'SOL_min', 'transition_rate_per_hour']:
            v = sm.get(m, {})
            print(f"  {m}: {v.get('mean','-')} +/- {v.get('std','-')}")
        for s in ['N1', 'N2', 'N3', 'REM']:
            v = st.get(s, {})
            print(f"  {s}%TST: {v.get('mean','-')}")
