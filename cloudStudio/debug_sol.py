import numpy as np
import mne

# Load one Sleep-EDF subject to check
edf_dir = r'e:\datasets\Sleep\sleep-edf-database-expanded-1.0.0\sleep-edf-database-expanded-1.0.0\sleep-edf-cassette\SC'
import os

# Find a menopausal subject
files = [f for f in os.listdir(edf_dir) if f.startswith('SC')]
print(f'Available files: {files[:5]}')

# Load subject 20 (menopausal)
edf_path = os.path.join(edf_dir, 'SC4002E0-PSG.edf')
if os.path.exists(edf_path):
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    print(f'\nSubject 20 (SC4002E0):')
    print(f'  Filename: {edf_path}')
    print(f'  Duration: {raw.n_times / raw.info["sfreq"] / 60:.1f} minutes')

    # Load annotations
    annot_path = edf_path.replace('-PSG.edf', '-Annotations.mat')
    if os.path.exists(annot_path):
        from scipy.io import loadmat
        annot = loadmat(annot_path)
        print(f'  Annotations available')

# Check labels
from experiment_three_groups_paper import load_edf_data

epochs, labels, sol_override = load_edf_data('20', night=1)
if epochs is not None:
    print(f'\nLoaded {len(labels)} epochs')
    print(f'Label distribution:')
    for label in sorted(set(labels)):
        count = np.sum(labels == label)
        print(f'  Label {label}: {count} epochs ({count/len(labels)*100:.1f}%)')

    # Find first non-Wake (sleep onset)
    sleep_indices = np.where(labels != 0)[0]
    if len(sleep_indices) > 0:
        first_sleep = sleep_indices[0]
        sol_epochs = first_sleep
        sol_min = sol_epochs * 30 / 60
        print(f'\nFirst sleep epoch index: {first_sleep}')
        print(f'SOL (30s epochs): {sol_epochs} epochs = {sol_min:.1f} minutes')
else:
    print('Failed to load data')