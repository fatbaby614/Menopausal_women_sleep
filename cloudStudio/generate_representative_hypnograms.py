"""
Generate representative hypnograms and hypnodensity plots for 3 groups
=====================================================================
For paper display, runs in two phases:

Phase 1: Compute (train model + predict + save results)
  python generate_representative_hypnograms.py --compute

Phase 2: Plot (generate figures from saved results)
  python generate_representative_hypnograms.py --plot

Saves intermediate results so plotting can be re-run quickly.
"""

import platform
import os
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mne
import glob
import warnings
warnings.filterwarnings('ignore')

mne.set_log_level('ERROR')

sys_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, sys_path)
from sca_fbts_woman import MenoSCA_FBTS

IS_WINDOWS = platform.system() == 'Windows'
if IS_WINDOWS:
    DATA_DIR = r"E:\datasets\Sleep\sleep-edf-database-expanded-1.0.0"
else:
    DATA_DIR = r"/mnt/data1/home/tanhuang/datasets/sleep-edf-database-expanded-1.0.0"
FS = 100
EPOCH_LENGTH = 30

SLEEP_STAGE_LABELS = {
    'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2,
    'Sleep stage 3': 3, 'Sleep stage 4': 3, 'Sleep stage R': 4,
}

# Representative subjects (selected based on N1/N3 percentages close to group median from experimental results)
REPRESENTATIVE_SUBJECTS = {
    'menopausal_women': {
        'subject_id': '27', 'night': 1,
        'name': 'Menopausal Woman (SC427)',
        'group_label': 'Menopausal Women',
    },
    'young_women': {
        'subject_id': '0', 'night': 1,
        'name': 'Young Woman (SC400)',
        'group_label': 'Young Women',
    },
    'young_men': {
        'subject_id': '15', 'night': 1,
        'name': 'Young Man (SC415)',
        'group_label': 'Young Men',
    }
}

STAGE_NAMES = ['Wake', 'N1', 'N2', 'N3', 'REM']
STAGE_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#2C3E50', '#9B59B6']

# ========== Data Loading ==========

def load_edf_data(subject_id, night=1):
    """Load single subject's EDF data for one night"""
    psg_pattern = f"SC4{str(subject_id).zfill(2)}{night}*-PSG.edf"
    psg_files = glob.glob(os.path.join(DATA_DIR, 'sleep-cassette', psg_pattern))
    if not psg_files:
        return None, None, None
    hyp_pattern = f"SC4{str(subject_id).zfill(2)}{night}*-Hypnogram.edf"
    hyp_files = glob.glob(os.path.join(DATA_DIR, 'sleep-cassette', hyp_pattern))
    if not hyp_files:
        return None, None, None
    hyp_file = hyp_files[0]

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
            return None, None, None
        raw.pick_channels(target_channels)
        data = raw.get_data()

        annotations = mne.read_annotations(hyp_file)
        labels_per_sample = np.full(data.shape[1], -1, dtype=int)
        for onset, duration, desc in zip(annotations.onset, annotations.duration, annotations.description):
            start_sample = int(onset * FS)
            end_sample = min(int((onset + duration) * FS), data.shape[1])
            mapped_label = SLEEP_STAGE_LABELS.get(desc, -1)
            if mapped_label >= 0:
                labels_per_sample[start_sample:end_sample] = mapped_label

        samples_per_epoch = FS * EPOCH_LENGTH
        n_epochs = data.shape[1] // samples_per_epoch
        epochs_data, epochs_labels = [], []
        for i in range(n_epochs):
            start = i * samples_per_epoch
            end = start + samples_per_epoch
            epoch_labels = labels_per_sample[start:end]
            valid_labels = epoch_labels[epoch_labels >= 0]
            if len(valid_labels) == 0:
                continue
            label = np.bincount(valid_labels).argmax()
            epoch_eeg = data[:, start:end]
            if epoch_eeg.shape[1] < samples_per_epoch:
                epoch_eeg = np.pad(epoch_eeg, ((0, 0), (0, samples_per_epoch - epoch_eeg.shape[1])), mode='edge')
            epochs_data.append(epoch_eeg)
            epochs_labels.append(label)

        if len(epochs_data) == 0:
            return None, None, None
        return np.array(epochs_data), np.array(epochs_labels), len(epochs_data)
    except Exception as e:
        print(f"  Error loading: {e}")
        return None, None, None

# ========== Model Training ==========

def train_model_and_predict(X, y):
    """Train model and predict (using 7-band MenoSCA-FBTS)"""
    model = MenoSCA_FBTS(
        n_bands=7,        # Consistent with experiment code
        estimator='oas',
        metric='riemann',
        classifier='ensemble',
        n_features=200,
        fs=FS,
        temporal_smoothing=True,
        smoothing_window=3,
        enable_menopause_features=True
    )
    model.fit(X, y)
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    return y_pred, y_prob

# ========== Computation Phase ==========

def compute_all():
    """Phase 1: Compute and save prediction results"""
    print("=" * 70)
    print("[Phase 1] Computing representative subject predictions")
    print("=" * 70)

    results_dir = os.path.join(sys_path, 'results_pytorch')
    os.makedirs(results_dir, exist_ok=True)

    all_results = {}

    for group_key, config in REPRESENTATIVE_SUBJECTS.items():
        subject_id = config['subject_id']
        night = config['night']
        subject_name = f"SC{subject_id} Night{night}"
        group_label = config['group_label']
        print(f"\nProcessing {group_label} ({subject_name})...")

        print(f"  Loading data...")
        X, y, n_epochs = load_edf_data(subject_id, night)
        if X is None:
            print(f"  FAILED")
            continue
        print(f"  OK - {n_epochs} epochs")

        print(f"  Training model and predicting...")
        y_pred, y_prob = train_model_and_predict(X, y)

        # Find main sleep period onset (continuous sleep block containing N3/REM, avoid daytime naps)
        is_sleep = (y != 0)
        sleep_indices = np.where(is_sleep)[0]
        first_sleep = sleep_indices[0] if len(sleep_indices) > 0 else 0
        last_sleep = sleep_indices[-1] if len(sleep_indices) > 0 else len(y) - 1

        # Build sleep block list
        sleep_blocks, in_block, blk_start = [], False, 0
        for i in range(len(y)):
            if is_sleep[i] and not in_block:
                blk_start = i; in_block = True
            elif not is_sleep[i] and in_block:
                sleep_blocks.append((blk_start, i - 1))
                in_block = False
        if in_block:
            sleep_blocks.append((blk_start, len(y) - 1))

        # Select main sleep block: ≥60 epochs and containing N3/REM
        deep_labels = {3, 4}
        main_block = None
        for start, end in sleep_blocks:
            bl = end - start + 1
            if bl < 60: continue
            if deep_labels & set(y[start:end + 1]):
                if main_block is None or bl > (main_block[1] - main_block[0] + 1):
                    main_block = (start, end)

        if main_block is not None:
            first_sleep = main_block[0]
            last_sleep = main_block[1]

        time_hours = np.linspace(0, n_epochs * 30 / 3600, n_epochs)
        accuracy = float(np.mean(y_pred == y))

        all_results[group_key] = {
            'config': config,
            'y_true': y.tolist(),
            'y_pred': y_pred.tolist(),
            'y_prob': y_prob.tolist(),
            'time_hours': time_hours.tolist(),
            'n_epochs': n_epochs,
            'first_sleep_epoch': int(first_sleep),
            'last_sleep_epoch': int(last_sleep),
            'accuracy': accuracy,
            'stage_pct': {
                name: float(np.sum(y == i) / n_epochs * 100)
                for i, name in enumerate(STAGE_NAMES)
            },
        }

        print(f"  OK - Accuracy: {accuracy*100:.1f}%")
        print(f"     W={all_results[group_key]['stage_pct']['Wake']:.1f}% "
              f"N1={all_results[group_key]['stage_pct']['N1']:.1f}% "
              f"N2={all_results[group_key]['stage_pct']['N2']:.1f}% "
              f"N3={all_results[group_key]['stage_pct']['N3']:.1f}% "
              f"REM={all_results[group_key]['stage_pct']['REM']:.1f}%")

    if not all_results:
        print("No data processed successfully")
        return

    save_path = os.path.join(results_dir, 'representative_results.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        # numpy array converted to list, serializable
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {save_path}")
    print(f"   {len(all_results)} group(s)")
    print(f"\nNext: python generate_representative_hypnograms.py --plot")

# ========== Plotting Phase ==========

def plot_all():
    """Phase 2: Read saved results and generate figures"""
    print("=" * 70)
    print("[Phase 2] Generating Hypnogram and Hypnodensity plots")
    print("=" * 70)

    results_dir = os.path.join(sys_path, 'results_pytorch')
    save_path = os.path.join(results_dir, 'representative_results.json')

    if not os.path.exists(save_path):
        print(f"Results file not found: {save_path}")
        print(f"   Please run: python generate_representative_hypnograms.py --compute")
        return

    with open(save_path, 'r', encoding='utf-8') as f:
        all_results = json.load(f)

    n_groups = len(all_results)
    if n_groups == 0:
        print("Results file is empty")
        return

    # ====== Hypnogram ======
    fig_hyp, axes_hyp = plt.subplots(n_groups, 1, figsize=(16, 2.2 * n_groups))
    if n_groups == 1:
        axes_hyp = [axes_hyp]

    # ====== Hypnodensity ======
    fig_dens, axes_dens = plt.subplots(n_groups, 1, figsize=(16, 2.8 * n_groups))
    if n_groups == 1:
        axes_dens = [axes_dens]

    for idx, (group_key, result) in enumerate(all_results.items()):
        y_true = np.array(result['y_true'])
        y_pred = np.array(result['y_pred'])
        y_prob = np.array(result['y_prob'])
        time_hours = np.array(result['time_hours'])
        config = result['config']
        accuracy = result['accuracy']

        subject_name = config['name']
        group_label = config['group_label']

        # Only plot data within sleep window
        first_sleep = result['first_sleep_epoch']
        last_sleep = result['last_sleep_epoch']
        sleep_window = slice(first_sleep, last_sleep + 1)

        is_last = (idx == n_groups - 1)
        _plot_hypnogram(
            y_true[sleep_window], y_pred[sleep_window],
            time_hours[sleep_window] - time_hours[first_sleep],
            subject_name, group_label, accuracy,
            ax=axes_hyp[idx], show_xlabel=is_last, add_legend=is_last
        )
        _plot_hypnodensity(
            y_prob[sleep_window],
            time_hours[sleep_window] - time_hours[first_sleep],
            subject_name, group_label,
            ax=axes_dens[idx],
            add_legend=is_last, show_xlabel=is_last
        )

    hyp_path = os.path.join(results_dir, 'hypnogram_comparison.png')
    dens_path = os.path.join(results_dir, 'hypnodensity_comparison.png')

    fig_hyp.subplots_adjust(hspace=0.35)
    fig_dens.subplots_adjust(hspace=0.40)

    fig_hyp.savefig(hyp_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
    fig_dens.savefig(dens_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)

    plt.close(fig_hyp)
    plt.close(fig_dens)

    print(f"\nHypnogram saved:  {hyp_path}")
    print(f"Hypnodensity saved: {dens_path}")


def _plot_hypnogram(y_true, y_pred, time_hours, subject_name, group_label, accuracy, ax, show_xlabel=True, add_legend=False):
    """Plot hypnogram for a single subject"""
    n = len(y_true)
    colors_true = [STAGE_COLORS[l] for l in y_true]
    colors_pred = [STAGE_COLORS[l] for l in y_pred]

    y_t = np.ones(n) * 0.3
    y_p = np.ones(n) * 0.7

    for i in range(n):
        ax.fill_between(
            [time_hours[i], time_hours[i+1] if i+1 < n else time_hours[-1] + 0.0083],
            [y_t[i]-0.15, y_t[i]-0.15], [y_t[i]+0.15, y_t[i]+0.15],
            color=colors_true[i], alpha=0.7
        )
    for i in range(n):
        ax.fill_between(
            [time_hours[i], time_hours[i+1] if i+1 < n else time_hours[-1] + 0.0083],
            [y_p[i]-0.15, y_p[i]-0.15], [y_p[i]+0.15, y_p[i]+0.15],
            color=colors_pred[i], alpha=0.45
        )

    ax.set_xlim([0, time_hours[-1] if len(time_hours) > 0 else 1])
    ax.set_ylim([0, 1])
    ax.set_yticks([0.3, 0.7])
    ax.set_yticklabels(['True', 'Pred'], fontsize=9)
    if show_xlabel:
        ax.set_xlabel('Time from sleep onset (hours)', fontsize=10)
    ax.set_title(
        f'{group_label} — {subject_name} (Acc={accuracy*100:.1f}%)',
        fontsize=11, fontweight='bold'
    )
    if add_legend:
        legend_patches = [
            mpatches.Patch(color=STAGE_COLORS[i], label=STAGE_NAMES[i])
            for i in range(5)
        ]
        ax.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.0, -0.12),
                  ncol=5, fontsize=8, framealpha=0.9)
    ax.grid(axis='x', alpha=0.3)


def _plot_hypnodensity(y_prob, time_hours, subject_name, group_label, ax, add_legend=False, show_xlabel=True):
    """Plot hypnodensity for a single subject"""
    n = y_prob.shape[0]
    window = 5
    y_prob_smoothed = np.zeros_like(y_prob)
    for i in range(n):
        start = max(0, i - window)
        end = min(n, i + window + 1)
        y_prob_smoothed[i] = np.mean(y_prob[start:end], axis=0)

    stack_data = [y_prob_smoothed[:, i] for i in range(5)]
    ax.stackplot(time_hours, *stack_data, colors=STAGE_COLORS, alpha=0.8)

    ax.set_xlim([0, time_hours[-1] if len(time_hours) > 0 else 1])
    ax.set_ylim([0, 1])
    ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9])
    ax.set_yticklabels(['Wake', 'N1', 'N2', 'N3', 'REM'], fontsize=9)
    if show_xlabel:
        ax.set_xlabel('Time from sleep onset (hours)', fontsize=10)
    ax.set_ylabel('Probability', fontsize=10)
    ax.set_title(f'{group_label} — {subject_name} Hypnodensity', fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    if add_legend:
        legend_patches = [
            mpatches.Patch(color=STAGE_COLORS[i], label=STAGE_NAMES[i])
            for i in range(5)
        ]
        ax.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.0, -0.12),
                  ncol=5, fontsize=8, framealpha=0.9)


# ========== Main Entry Point ==========

def main():
    parser = argparse.ArgumentParser(
        description='Generate representative hypnograms for 3 groups'
    )
    parser.add_argument('--compute', action='store_true',
                       help='Phase 1: train model and save predictions')
    parser.add_argument('--plot', action='store_true',
                       help='Phase 2: generate figures from saved results')
    parser.add_argument('--all', action='store_true',
                       help='Run both compute and plot')

    args = parser.parse_args()

    # Default: run all
    if not any([args.compute, args.plot, args.all]):
        args.all = True

    if args.compute or args.all:
        compute_all()
    if args.plot or args.all:
        plot_all()


if __name__ == '__main__':
    main()
