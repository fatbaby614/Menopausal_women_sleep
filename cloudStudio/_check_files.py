import json, os

# Check which files exist
base1 = r'D:\TanHuangWork\Menopausal_women_sleep\cloudStudio\experiment_results'
base2 = r'D:\TanHuangWork\Menopausal_women_sleep\Paper\dataUsed'
base3 = r'D:\TanHuangWork\Menopausal_women_sleep\cloudStudio\experiment_results_0511'

out = []
for base_name, base in [('exp_results', base1), ('paper_dataUsed', base2), ('exp_results_0511', base3)]:
    out.append(f'\n=== {base_name} ===')
    try:
        for f in sorted(os.listdir(base)):
            if 'sleep-edf' in f.lower() or 'sota' in f.lower() or 'ablation' in f.lower() or 'isruc' in f.lower() or 'dreams' in f.lower():
                out.append(f'  {f}')
    except Exception as e:
        out.append(f'  ERROR: {e}')

with open(r'D:\TanHuangWork\Menopausal_women_sleep\cloudStudio\_files_check.txt', 'w') as f:
    f.write('\n'.join(out))
print('Written')
