"""
MenoSCA-FBTS Ablation Study
===================================
Verify contribution of each algorithm component

Ablation components:
1. Menopause Features
2. Riemannian geometry metric (Riemann vs Euclid)
3. Temporal Smoothing
4. Frequency Band Combinations
5. Classifier (Ensemble vs LDA vs SVM)
6. Covariance estimator (OAS vs LWF vs SCM)

Dataset: Sleep-EDF Menopausal Women
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import json
import time
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings('ignore')

import mne
mne.set_log_level('ERROR')

sys_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, sys_path)

import platform
IS_WINDOWS = platform.system() == 'Windows'

if IS_WINDOWS:
    SLEEP_EDF_DIR = r"E:\datasets\Sleep\sleep-edf-database-expanded-1.0.0"
    OUTPUT_DIR = r"experiment_results"
else:
    SLEEP_EDF_DIR = r"/mnt/data1/home/tanhuang/datasets/sleep-edf-database-expanded-1.0.0"
    OUTPUT_DIR = r"./experiment_results"

from sca_fbts_woman import MenoSCA_FBTS

FS = 100
EPOCH_LENGTH_S = 30
SAMPLES_PER_EPOCH = FS * EPOCH_LENGTH_S

SLEEP_STAGE_LABELS = {
    'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2,
    'Sleep stage 3': 3, 'Sleep stage 4': 3, 'Sleep stage R': 4,
}
STAGE_NAMES = ['Wake', 'N1', 'N2', 'N3', 'REM']

MENOPAUSAL_SUBJECTS = ['20', '21', '23', '24', '26', '27', '29', '80']

def load_edf_data(subject_id, night=1):
    """Load Sleep-EDF data"""
    psg_pattern = f"SC4{str(subject_id).zfill(2)}{night}*-PSG.edf"
    psg_files = glob.glob(os.path.join(SLEEP_EDF_DIR, 'sleep-cassette', psg_pattern))
    if not psg_files:
        return None, None
    hyp_pattern = f"SC4{str(subject_id).zfill(2)}{night}*-Hypnogram.edf"
    hyp_files = glob.glob(os.path.join(SLEEP_EDF_DIR, 'sleep-cassette', hyp_pattern))
    if not hyp_files:
        return None, None
    try:
        raw = mne.io.read_raw_edf(psg_files[0], preload=True, verbose=False)
        if raw.info['sfreq'] != FS:
            raw.resample(FS)
        eeg_channels = [ch for ch in raw.ch_names if 'EEG' in ch]
        target_channels = []
        for ch in ['EEG Fpz-Cz', 'EEG Pz-Oz', 'Fpz-Cz', 'Pz-Oz']:
            if ch in eeg_channels:
                target_channels.append(ch)
        if len(target_channels) < 2:
            target_channels = eeg_channels[:2]
        if len(target_channels) < 2:
            return None, None
        raw.pick_channels(target_channels)
        data = raw.get_data()
        annotations = mne.read_annotations(hyp_files[0])
        labels_per_sample = np.full(data.shape[1], -1, dtype=int)
        for onset, duration, desc in zip(annotations.onset, annotations.duration, annotations.description):
            start_sample = int(onset * FS)
            end_sample = min(int((onset + duration) * FS), data.shape[1])
            mapped_label = SLEEP_STAGE_LABELS.get(desc, -1)
            if mapped_label >= 0:
                labels_per_sample[start_sample:end_sample] = mapped_label
        n_epochs = data.shape[1] // SAMPLES_PER_EPOCH
        epochs_data, epochs_labels = [], []
        for i in range(n_epochs):
            start, end = i * SAMPLES_PER_EPOCH, (i + 1) * SAMPLES_PER_EPOCH
            epoch_labels = labels_per_sample[start:end]
            valid_labels = epoch_labels[epoch_labels >= 0]
            if len(valid_labels) == 0:
                continue
            label = np.bincount(valid_labels).argmax()
            epoch_eeg = data[:, start:end]
            if epoch_eeg.shape[1] < SAMPLES_PER_EPOCH:
                epoch_eeg = np.pad(epoch_eeg, ((0, 0), (0, SAMPLES_PER_EPOCH - epoch_eeg.shape[1])), mode='edge')
            epochs_data.append(epoch_eeg)
            epochs_labels.append(label)
        return np.array(epochs_data), np.array(epochs_labels)
    except Exception:
        return None, None

def run_ablation_experiment(name, config):
    """Run single ablation experiment configuration"""
    print(f"\n  Config: {name}")
    print(f"    - enable_menopause_features: {config.get('enable_menopause_features', True)}")
    print(f"    - metric: {config.get('metric', 'riemann')}")
    print(f"    - temporal_smoothing: {config.get('temporal_smoothing', True)}")
    print(f"    - classifier: {config.get('classifier', 'ensemble')}")
    print(f"    - estimator: {config.get('estimator', 'oas')}")
    print(f"    - freq_bands: {config.get('freq_bands', 'default')}")
    
    fold_results = {'accuracy': [], 'f1_macro': [], 'f1_per_class': [], 'conf_matrix': []}
    total_start = time.time()
    
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        fold_start = time.time()
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        clf = MenoSCA_FBTS(
            n_bands=config.get('n_bands', 7),  # Default to 7 bands (not 10!)
            estimator=config.get('estimator', 'oas'),
            metric=config.get('metric', 'riemann'),
            classifier=config.get('classifier', 'ensemble'),
            n_features=config.get('n_features', 200),
            fs=FS,
            freq_bands=config.get('freq_bands', None),
            temporal_smoothing=config.get('temporal_smoothing', True),
            smoothing_window=config.get('smoothing_window', 3),
            enable_menopause_features=config.get('enable_menopause_features', True)
        )
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        f1_per_class = f1_score(y_test, y_pred, average=None)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3, 4])
        
        fold_results['accuracy'].append(acc)
        fold_results['f1_macro'].append(f1)
        fold_results['f1_per_class'].append(f1_per_class)
        fold_results['conf_matrix'].append(cm)
        
        fold_time = time.time() - fold_start
        print(f"    Fold {fold_idx+1}: Acc={acc:.4f}, F1={f1:.4f}, Time={fold_time:.1f}s")
    
    total_time = time.time() - total_start
    
    result = {
        'accuracy': np.mean(fold_results['accuracy']),
        'accuracy_std': np.std(fold_results['accuracy']),
        'f1_macro': np.mean(fold_results['f1_macro']),
        'f1_macro_std': np.std(fold_results['f1_macro']),
        'f1_per_class_mean': np.mean(fold_results['f1_per_class'], axis=0).tolist(),
        'conf_matrix_mean': np.mean(fold_results['conf_matrix'], axis=0).tolist(),
        'training_time': total_time,
        'config': config
    }
    
    print(f"    Average: Acc={result['accuracy']:.4f}±{result['accuracy_std']:.4f}")
    print(f"    Total training time: {total_time:.1f}s")
    
    return result

def run_experiment():
    """Run full ablation experiment"""
    print("=" * 70)
    print("MenoSCA-FBTS Ablation Study")
    print("=" * 70)
    print(f"Dataset: Sleep-EDF Menopausal Women")
    print(f"Subjects: {MENOPAUSAL_SUBJECTS}")
    print("=" * 70)
    
    global X, y, groups, gkf
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_epochs, all_labels, all_subjects = [], [], []
    print("\n[1/3] Loading data...")
    for subj_id in MENOPAUSAL_SUBJECTS:
        print(f"  Subject {subj_id}...", end=" ")
        epochs, labels = load_edf_data(subj_id, night=1)
        if epochs is not None:
            print(f"{len(epochs)} epochs OK")
            all_epochs.extend(epochs)
            all_labels.extend(labels)
            all_subjects.extend([subj_id] * len(labels))
        else:
            print("FAILED")
    
    X = np.array(all_epochs)
    y = np.array(all_labels)
    groups = np.array(all_subjects)
    gkf = GroupKFold(n_splits=5)
    
    print(f"\n  Total: {len(X)} epochs")
    print(f"  Label distribution: ", end="")
    for i, name in enumerate(STAGE_NAMES):
        count = np.sum(y == i)
        pct = count / len(y) * 100
        print(f"{name}={count}({pct:.1f}%) ", end="")
    print()
    
    results = {}
    
    print("\n[2/3] Ablation experiment configurations:")
    
    ablation_configs = {
        '1. Full Model (Baseline)': {
            'enable_menopause_features': True,
            'metric': 'riemann',
            'temporal_smoothing': True,
            'classifier': 'ensemble',
            'estimator': 'oas',
        },
        '2. -Menopause Features': {
            'enable_menopause_features': False,
            'metric': 'riemann',
            'temporal_smoothing': True,
            'classifier': 'ensemble',
            'estimator': 'oas',
        },
        '3. Euclid vs Riemann': {
            'enable_menopause_features': True,
            'metric': 'euclid',
            'temporal_smoothing': True,
            'classifier': 'ensemble',
            'estimator': 'oas',
        },
        '4. -Temporal Smoothing': {
            'enable_menopause_features': True,
            'metric': 'riemann',
            'temporal_smoothing': False,
            'classifier': 'ensemble',
            'estimator': 'oas',
        },
        '5. Classifier-LDA': {
            'enable_menopause_features': True,
            'metric': 'riemann',
            'temporal_smoothing': True,
            'classifier': 'lda',
            'estimator': 'oas',
        },
        '6. Classifier-SVM': {
            'enable_menopause_features': True,
            'metric': 'riemann',
            'temporal_smoothing': True,
            'classifier': 'svm',
            'estimator': 'oas',
        },
        '7. Estimator-LWF': {
            'enable_menopause_features': True,
            'metric': 'riemann',
            'temporal_smoothing': True,
            'classifier': 'ensemble',
            'estimator': 'lwf',
        },
        '8. Estimator-SCM': {
            'enable_menopause_features': True,
            'metric': 'riemann',
            'temporal_smoothing': True,
            'classifier': 'ensemble',
            'estimator': 'scm',
        },
        '9. 5-Bands (Baseline)': {
            'enable_menopause_features': True,
            'metric': 'riemann',
            'temporal_smoothing': True,
            'classifier': 'ensemble',
            'estimator': 'oas',
            'freq_bands': [
                (0.5, 4), (4, 8), (8, 12), (12, 30), (30, 40)
            ],
            'n_bands': 5,
        },
        '10. 10-Bands (Extended)': {
            'enable_menopause_features': True,
            'metric': 'riemann',
            'temporal_smoothing': True,
            'classifier': 'ensemble',
            'estimator': 'oas',
            'freq_bands': [
                (0.5, 2), (2, 4), (4, 6), (6, 8), (8, 10),
                (10, 12), (12, 15), (15, 20), (20, 30), (30, 40)
            ],
            'n_bands': 10,
        },
    }
    
    print("\n[3/3] Running ablation experiments...")
    for name, config in ablation_configs.items():
        results[name] = run_ablation_experiment(name, config)
    
    print("\n" + "=" * 70)
    print("Ablation Study Results Summary")
    print("=" * 70)
    print(f"{'Config':<30} {'Accuracy':<15} {'F1-Macro':<15} {'Δ Acc':<10}")
    print("-" * 70)
    
    baseline_acc = results['1. Full Model (Baseline)']['accuracy']
    baseline_f1 = results['1. Full Model (Baseline)']['f1_macro']
    
    for name, res in results.items():
        delta = res['accuracy'] - baseline_acc
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        print(f"{name:<30} {res['accuracy']:.4f}±{res['accuracy_std']:.4f}   {res['f1_macro']:.4f}±{res['f1_macro_std']:.4f}   {delta_str}")
    print("=" * 70)
    
    print("\nF1 scores for each sleep stage:")
    print(f"{'Config':<30} {'Wake':<8} {'N1':<8} {'N2':<8} {'N3':<8} {'REM':<8}")
    print("-" * 70)
    for name, res in results.items():
        f1_per_class = res['f1_per_class_mean']
        print(f"{name:<30} {f1_per_class[0]:.4f}   {f1_per_class[1]:.4f}   {f1_per_class[2]:.4f}   {f1_per_class[3]:.4f}   {f1_per_class[4]:.4f}")
    print("-" * 70)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(OUTPUT_DIR, f"ablation_results_{timestamp}.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {result_file}")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    config_names = list(results.keys())
    accs = [results[m]['accuracy'] for m in config_names]
    stds = [results[m]['accuracy_std'] for m in config_names]
    
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(config_names)), accs, yerr=stds, capsize=3, color='steelblue')
    ax1.axhline(y=baseline_acc, color='red', linestyle='--', label=f'Baseline ({baseline_acc:.4f})')
    ax1.set_xticks(range(len(config_names)))
    ax1.set_xticklabels([n.split('.')[0] for n in config_names], rotation=45, ha='right')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Ablation Study: Accuracy Comparison')
    ax1.legend()
    ax1.set_ylim(0, 1)
    
    ax2 = axes[0, 1]
    f1s = [results[m]['f1_macro'] for m in config_names]
    f1_stds = [results[m]['f1_macro_std'] for m in config_names]
    bars = ax2.bar(range(len(config_names)), f1s, yerr=f1_stds, capsize=3, color='darkorange')
    ax2.axhline(y=baseline_f1, color='red', linestyle='--', label=f'Baseline ({baseline_f1:.4f})')
    ax2.set_xticks(range(len(config_names)))
    ax2.set_xticklabels([n.split('.')[0] for n in config_names], rotation=45, ha='right')
    ax2.set_ylabel('F1-Macro')
    ax2.set_title('Ablation Study: F1-Macro Comparison')
    ax2.legend()
    ax2.set_ylim(0, 1)
    
    ax3 = axes[1, 0]
    conf_matrices = np.array(results['1. Full Model (Baseline)']['conf_matrix_mean'])
    im = ax3.imshow(conf_matrices, cmap='Blues')
    ax3.set_xticks(range(5))
    ax3.set_yticks(range(5))
    ax3.set_xticklabels(STAGE_NAMES)
    ax3.set_yticklabels(STAGE_NAMES)
    ax3.set_title('Confusion Matrix: Full Model (Baseline)')
    for i in range(5):
        for j in range(5):
            text = ax3.text(j, i, f'{conf_matrices[i, j]:.0f}', ha='center', va='center', color='black')
    plt.colorbar(im, ax=ax3)
    
    ax4 = axes[1, 1]
    improvements = [(results[m]['accuracy'] - baseline_acc) * 100 for m in config_names]
    colors = ['green' if x >= 0 else 'red' for x in improvements]
    bars = ax4.bar(range(len(config_names)), improvements, color=colors)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_xticks(range(len(config_names)))
    ax4.set_xticklabels([n.split('.')[0] for n in config_names], rotation=45, ha='right')
    ax4.set_ylabel('Δ Accuracy (%)')
    ax4.set_title('Ablation: Accuracy Change vs Baseline')
    
    plt.tight_layout()
    fig_file = os.path.join(OUTPUT_DIR, f"ablation_results_{timestamp}.png")
    plt.savefig(fig_file, dpi=150)
    print(f"Figure saved: {fig_file}")
    
    print("\n" + "=" * 70)
    print("Key Findings:")
    print("=" * 70)
    
    delta_menopause = results['2. -Menopause Features']['accuracy'] - baseline_acc
    delta_riemann = results['3. Euclid vs Riemann']['accuracy'] - baseline_acc
    delta_smoothing = results['4. -Temporal Smoothing']['accuracy'] - baseline_acc
    
    print(f"1. Menopause Features contribution: {delta_menopause*100:+.2f}% (accuracy {'decreased' if delta_menopause < 0 else 'increased'} after removal)")
    print(f"2. Riemannian geometry contribution: {delta_riemann*100:+.2f}% (Euclidean vs Riemann)")
    print(f"3. Temporal smoothing contribution: {delta_smoothing*100:+.2f}% (accuracy {'decreased' if delta_smoothing < 0 else 'increased'} after removal)")
    
    return results

if __name__ == '__main__':
    import glob
    results = run_experiment()