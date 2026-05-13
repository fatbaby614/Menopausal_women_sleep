import json

with open('experiment_results/paper_results_sleep-edf_20260512_185318.json', 'r') as f:
    data = json.load(f)

print('=== Sleep-EDF SOL Analysis ===')
for subj in list(data['subject_details']['menopausal_women'])[:8]:
    sid = subj['subject_id']
    n1 = subj.get('night1', {}).get('clinical_metrics', {})
    n2 = subj.get('night2', {}).get('clinical_metrics', {})
    print(f'Subject {sid}: Night1 SOL={n1.get("SOL_min")}, Night2 SOL={n2.get("SOL_min")}')

print('\n=== 10-Band vs Baseline Analysis ===')
with open('experiment_results/ablation_results_20260512_184228.json', 'r') as f:
    ablation = json.load(f)

baseline = ablation.get('1. Full Model (Baseline)', {})
band10 = ablation.get('10. 10-Bands (Extended)', {})

print(f'Baseline accuracy: {baseline.get("accuracy")}')
print(f'10-Bands accuracy: {band10.get("accuracy")}')
print(f'Diff: {baseline.get("accuracy") - band10.get("accuracy")}')

print(f'\nBaseline f1_macro: {baseline.get("f1_macro")}')
print(f'10-Bands f1_macro: {band10.get("f1_macro")}')
print(f'F1 Diff: {baseline.get("f1_macro") - band10.get("f1_macro")}')

print(f'\nBaseline conf_matrix[0][:3]: {baseline.get("conf_matrix_mean", [[]])[0][:3]}')
print(f'10-Bands conf_matrix[0][:3]: {band10.get("conf_matrix_mean", [[]])[0][:3]}')