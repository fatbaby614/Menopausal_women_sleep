"""
MenoSCA-FBTS Three-Group Sleep Staging Experiment (Full Paper Edition)
==============================================
Supports Sleep-EDF, ISRUC-Sleep, and DREAMS datasets.
Use --dataset sleep-edf|isruc|dreams to switch.
"""
import os, json, sys, argparse, time, platform
import numpy as np
import pandas as pd
import mne
import glob
from scipy import signal, stats
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
import warnings
warnings.filterwarnings('ignore')

mne.set_log_level('ERROR')

sys_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, sys_path)

# ============ Command-Line Arguments ============
parser = argparse.ArgumentParser(description='MenoSCA-FBTS Three-Group Sleep Staging Experiment')
parser.add_argument('--dataset', type=str, default='sleep-edf',
                    choices=['sleep-edf', 'isruc', 'dreams'],
                    help='Dataset: sleep-edf (default), isruc, or dreams')
args, _ = parser.parse_known_args()
DATASET = args.dataset

# ============ Path Configuration ============
IS_WINDOWS = platform.system() == 'Windows'

if IS_WINDOWS:
    SLEEP_EDF_DIR = r"E:\datasets\Sleep\sleep-edf-database-expanded-1.0.0"
    ISRUC_DIR = r"E:\datasets\Sleep\ISRUC-SLEEP"
    DREAMS_DIR = r"E:\datasets\Sleep\DREAMS\DatabasePatients"
    OUTPUT_DIR = r"experiment_results"
    MENOPAUSAL_JSON = r"json\menopausal_women_subjects.json"
    YOUNG_WOMEN_JSON = r"json\young_women_control_group.json"
    YOUNG_MEN_JSON = r"json\young_men_control_group.json"
else:
    SLEEP_EDF_DIR = os.environ.get("SLEEP_EDF_DIR",
        r"/mnt/data1/home/tanhuang/datasets/sleep-edf-database-expanded-1.0.0")
    ISRUC_DIR = os.environ.get("ISRUC_DIR",
        r"/mnt/data1/home/tanhuang/datasets/ISRUC-SLEEP")
    DREAMS_DIR = os.environ.get("DREAMS_DIR",
        r"/mnt/data1/home/tanhuang/datasets/DREAMS/DatabasePatients")
    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", r"./experiment_results")
    MENOPAUSAL_JSON = os.environ.get("MENOPAUSAL_JSON",
        r"json/menopausal_women_subjects.json")
    YOUNG_WOMEN_JSON = os.environ.get("YOUNG_WOMEN_JSON",
        r"json/young_women_control_group.json")
    YOUNG_MEN_JSON = os.environ.get("YOUNG_MEN_JSON",
        r"json/young_men_control_group.json")


from sca_fbts_woman import MenoSCA_FBTS

# ============ Configuration Parameters ============
FS = 100
if DATASET == 'isruc':
    FS_RAW = 200  # ISRUC original sampling rate
else:
    FS_RAW = 100
EPOCH_LENGTH_S = 30
SAMPLES_PER_EPOCH = FS * EPOCH_LENGTH_S

SLEEP_STAGE_LABELS = {
    'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2,
    'Sleep stage 3': 3, 'Sleep stage 4': 3, 'Sleep stage R': 4,
}
STAGE_NAMES = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
STAGE_LIST = ['Wake', 'N1', 'N2', 'N3', 'REM']

FREQ_BANDS = [
    (0.5, 4, 'Delta'),      (4, 6, 'Low_Theta'),
    (6, 8, 'High_Theta'),   (8, 12, 'Alpha'),
    (12, 15, 'Sigma'),      (15, 30, 'Beta'),
    (30, 40, 'Gamma'),
]
BAND_NAMES = [b[2] for b in FREQ_BANDS]

# ISRUC sleep stage label mapping (0=W, 1=N1, 2=N2, 3=N3, 5=REM)
ISRUC_LABEL_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4}

# ============ Subject Configuration ============
SLEEP_EDF_GROUP_CONFIG = {
    'menopausal_women': {
        'name': 'Menopausal Women', 'age_range': (45, 59), 'gender': 'F',
        'subject_ids': ['20', '21', '23', '24', '26', '27', '29', '80']
    },
    'young_women': {
        'name': 'Young Women', 'age_range': (25, 34), 'gender': 'F',
        'subject_ids': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    },
    'young_men': {
        'name': 'Young Men', 'age_range': (25, 38), 'gender': 'M',
        'subject_ids': ['10', '11', '12', '13', '14', '15', '16', '17', '18', '19']
    }
}

ISRUC_GROUP_CONFIG = {
    'menopausal_women': {
        'name': 'Menopausal Women', 'age_range': (45, 59), 'gender': 'F',
        'subject_ids': ['5', '10', '19', '23', '44', '59', '61', '64']
    },
    'young_women': {
        'name': 'Young Women', 'age_range': (20, 36), 'gender': 'F',
        'subject_ids': ['25', '27', '33', '36', '48', '65', '67']
    },
    'young_men': {
        'name': 'Young Men', 'age_range': (20, 38), 'gender': 'M',
        'subject_ids': ['4', '6', '18', '31', '38', '46', '50', '57']
    }
}

DREAMS_GROUP_CONFIG = {
    'menopausal_women': {
        'name': 'Menopausal Women', 'age_range': (45, 59), 'gender': 'F',
        'subject_ids': ['subject2', 'subject4', 'subject5', 'subject7', 'subject12', 'subject14']
    },
    'young_women': {
        'name': 'Young Women', 'age_range': (20, 30), 'gender': 'F',
        'subject_ids': ['subject1', 'subject3', 'subject8', 'subject9', 'subject10', 'subject11', 'subject13', 'subject15', 'subject16']
    },
    'young_men': {
        'name': 'Young Men', 'age_range': (20, 27), 'gender': 'M',
        'subject_ids': ['subject17', 'subject18', 'subject19', 'subject20']
    }
}

if DATASET == 'isruc':
    GROUP_CONFIG = ISRUC_GROUP_CONFIG
elif DATASET == 'dreams':
    GROUP_CONFIG = DREAMS_GROUP_CONFIG
else:
    GROUP_CONFIG = SLEEP_EDF_GROUP_CONFIG

# =====================================================================
# Part 1: Data Loading
# =====================================================================

def load_subject_metadata():
    """Load subject metadata"""
    subject_info = {}
    if DATASET == 'sleep-edf':
        csv_path = os.path.normpath(os.path.join(SLEEP_EDF_DIR, '..', 'SC-subjects.csv'))
        for p in [csv_path, os.path.join(SLEEP_EDF_DIR, 'SC-subjects.csv')]:
            if os.path.exists(p):
                csv_path = p; break
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                sid = str(row['subject'])
                if sid not in subject_info:
                    subject_info[sid] = {
                        'age': int(row['age']),
                        'gender': 'F' if row['sex (F=1)'] == 1 else 'M',
                        'nights': [],
                    }
                subject_info[sid]['nights'].append(int(row['night']))
        for json_path in [MENOPAUSAL_JSON, YOUNG_WOMEN_JSON, YOUNG_MEN_JSON]:
            try:
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    for ds_name, ds_data in data.get('datasets', {}).items():
                        if ds_name != 'Sleep-EDF': continue
                        for entry in ds_data.get('female_subjects', []) + ds_data.get('control_subjects', []):
                            sid = str(entry.get('subject_id', ''))
                            if sid in subject_info:
                                if entry.get('diagnosis'):
                                    subject_info[sid]['diagnosis'] = entry['diagnosis']
                                if entry.get('other_problems'):
                                    subject_info[sid]['other_problems'] = entry['other_problems']
            except Exception:
                pass
    elif DATASET == 'dreams':
        # DREAMS: read metadata from JSON config file
        for json_path in [MENOPAUSAL_JSON, YOUNG_WOMEN_JSON, YOUNG_MEN_JSON]:
            try:
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    for ds_name, ds_data in data.get('datasets', {}).items():
                        if ds_name != 'DREAMS': continue
                        for entry in ds_data.get('menopausal_subjects', []) + ds_data.get('control_subjects', []):
                            sid = str(entry.get('subject_id', ''))
                            if not sid: continue
                            subject_info[sid] = {
                                'age': entry.get('age', 'N/A'),
                                'gender': entry.get('gender', 'N/A'),
                                'nights': entry.get('nights', [1]),
                                'recording_duration': entry.get('recording_duration', None),
                            }
            except Exception:
                pass
    else:
        # ISRUC: read metadata from CSV
        csv_path = os.path.join(ISRUC_DIR, 'Details_subgroup_I_Submission.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, skiprows=2)
            for _, row in df.iterrows():
                try:
                    sid_val = str(row['Subject']).strip()
                    if not sid_val.isdigit():
                        continue
                    age_val = str(row['Age']).strip()
                    if not age_val.isdigit():
                        continue
                    sid = str(int(float(sid_val)))
                    subject_info[sid] = {
                        'age': int(age_val),
                        'gender': str(row['Sex']).strip(),
                        'subject_id': sid,
                        'diagnosis': str(row.get('Diagnosis', '')).strip(),
                        'other_problems': str(row.get('Other problems', '')).strip(),
                        'nights': [1],
                    }
                except (ValueError, KeyError):
                    pass
    return subject_info


def load_edf_data(subject_id, night=1):
    """Load Sleep-EDF data
    Modified: filter 24h recording to keep only main sleep period (longest continuous sleep block)
    """
    psg_pattern = f"SC4{str(subject_id).zfill(2)}{night}*-PSG.edf"
    psg_files = glob.glob(os.path.join(SLEEP_EDF_DIR, 'sleep-cassette', psg_pattern))
    if not psg_files:
        return None, None, None
    hyp_pattern = f"SC4{str(subject_id).zfill(2)}{night}*-Hypnogram.edf"
    hyp_files = glob.glob(os.path.join(SLEEP_EDF_DIR, 'sleep-cassette', hyp_pattern))
    if not hyp_files:
        return None, None, None
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

        annotations = mne.read_annotations(hyp_files[0])
        
        # ========== NEW: Sleep window filtering (find longest continuous sleep block) ==========
        # 1. Build stage sequence
        stage_sequence = []  # (start_time, duration, stage_label)
        for onset, duration, desc in zip(annotations.onset, annotations.duration, annotations.description):
            if 'Sleep stage' in desc:
                stage_sequence.append((onset, duration, desc))
        
        if not stage_sequence:
            return None, None, None
        
        # 2. Merge adjacent identical stages
        merged_stages = []
        for onset, dur, desc in stage_sequence:
            if merged_stages and merged_stages[-1][2] == desc:
                prev_onset, prev_dur, prev_desc = merged_stages.pop()
                merged_stages.append((prev_onset, prev_dur + dur, prev_desc))
            else:
                merged_stages.append((onset, dur, desc))
        
        # 3. Find longest sleep block (containing N3 or REM, allowing small Wake interruptions)
        sleep_window_start = 0.0
        sleep_window_end = data.shape[1] / FS
        
        # Definition: small Wake threshold (Wake shorter than this is treated as brief arousal within sleep)
        small_wake_threshold = 300  # 5 minutes
        
        max_sleep_block = None
        max_sleep_duration = 0
        
        # Iterate through each stage as starting point
        for i in range(len(merged_stages)):
            current_start = merged_stages[i][0]
            current_end = current_start
            has_deep_sleep = False
            has_rem = False
            total_wake_within = 0  # Wake duration within window
            
            for j in range(i, len(merged_stages)):
                onset_j, dur_j, desc_j = merged_stages[j]
                
                # If prolonged Wake encountered, stop extending
                if desc_j == 'Sleep stage W' and dur_j > small_wake_threshold:
                    # Check if current accumulated sleep block is valid
                    if has_deep_sleep or has_rem:
                        block_duration = current_end - current_start
                        if block_duration > max_sleep_duration:
                            max_sleep_duration = block_duration
                            max_sleep_block = (current_start, current_end)
                    break
                
                # Expand window
                current_end = onset_j + dur_j
                
                # Record whether deep sleep or REM is present
                if desc_j == 'Sleep stage 3' or desc_j == 'Sleep stage 4':
                    has_deep_sleep = True
                if desc_j == 'Sleep stage R':
                    has_rem = True
            
            # Check last window
            if has_deep_sleep or has_rem:
                block_duration = current_end - current_start
                if block_duration > max_sleep_duration:
                    max_sleep_duration = block_duration
                    max_sleep_block = (current_start, current_end)
        
        # Find first sleep onset for SOL calculation (from original recording start)
        # SOL = time from recording start to first sleep epoch
        first_sleep_onset_for_sol = 0.0  # Default value

        if max_sleep_block:
            sleep_window_start = max(0.0, max_sleep_block[0] - 300)  # 5-min buffer before
            sleep_window_end = min(data.shape[1] / FS, max_sleep_block[1] + 300)  # 5-min buffer after
            first_sleep_onset_for_sol = max_sleep_block[0]
        else:
            sleep_stages = [(o, d, desc) for o, d, desc in stage_sequence if desc != 'Sleep stage W']
            if sleep_stages:
                sleep_window_start = max(0.0, sleep_stages[0][0] - 300)
                last_sleep = sleep_stages[-1]
                sleep_window_end = min(data.shape[1] / FS, last_sleep[0] + last_sleep[1] + 300)
                first_sleep_onset_for_sol = sleep_stages[0][0]
            else:
                sleep_window_start = 0.0
                sleep_window_end = data.shape[1] / FS
        
        # Convert time to sample points
        start_sample = int(sleep_window_start * FS)
        end_sample = int(sleep_window_end * FS)
        
        # Only keep data within sleep window
        data = data[:, start_sample:end_sample]
        # =======================================

        labels_per_sample = np.full(data.shape[1], -1, dtype=int)
        for onset, duration, desc in zip(annotations.onset, annotations.duration, annotations.description):
            # Adjust onset for truncated data
            adjusted_onset = onset - sleep_window_start
            if adjusted_onset < 0:
                adjusted_duration = duration + adjusted_onset
                adjusted_onset = 0
            else:
                adjusted_duration = duration
            
            if adjusted_onset >= data.shape[1] / FS:
                continue
            
            start_sample = int(adjusted_onset * FS)
            end_sample = min(int((adjusted_onset + adjusted_duration) * FS), data.shape[1])
            mapped_label = SLEEP_STAGE_LABELS.get(desc, -1)
            if mapped_label >= 0:
                labels_per_sample[start_sample:end_sample] = mapped_label

        n_epochs = data.shape[1] // SAMPLES_PER_EPOCH
        epochs_data, epochs_labels = [], []
        for i in range(n_epochs):
            start = i * SAMPLES_PER_EPOCH
            end = start + SAMPLES_PER_EPOCH
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

        if len(epochs_data) == 0:
            return None, None, None
        # Return SOL calculated from original recording start (not sleep window start)
        sol_from_recording_start = first_sleep_onset_for_sol / 60.0 if first_sleep_onset_for_sol > 0 else 0.0
        return np.array(epochs_data), np.array(epochs_labels), sol_from_recording_start
    except Exception as e:
        print(f"    Load failed: {e}")
        return None, None, None


def load_isruc_data(subject_id, night=1):
    """Load ISRUC-Sleep data (using pyedflib)"""
    import pyedflib
    subj_dir = os.path.join(ISRUC_DIR, 'subgroup1', str(subject_id))
    rec_path = os.path.join(subj_dir, f'{subject_id}.rec')
    if not os.path.exists(rec_path):
        return None, None, None

    txt_path = os.path.join(subj_dir, f'{subject_id}_{night}.txt')
    if not os.path.exists(txt_path):
        return None, None, None

    try:
        # Read PSG signals
        f = pyedflib.EdfReader(rec_path)
        n_signals = f.signals_in_file

        # Select EEG channels: C3-A2 (index 3) and O1-A2 (index 4)
        # Mimic Sleep-EDF central + occipital dual-channel
        eeg_channels = []
        target_labels = ['C3-A2', 'O1-A2', 'C3', 'O1', 'EEG C3-A2', 'EEG O1-A2']
        for i in range(n_signals):
            label = f.getLabel(i).strip()
            if label in target_labels or ('C3' in label and 'A2' in label):
                eeg_channels.append(i)
                if len(eeg_channels) == 2:
                    break
        # If specified channels not found, use first two EEG-like channels
        if len(eeg_channels) < 2:
            for i in range(n_signals):
                label = f.getLabel(i).strip()
                if any(k in label.upper() for k in ['EEG', 'C3', 'C4', 'O1', 'O2', 'F3', 'F4']):
                    if i not in eeg_channels:
                        eeg_channels.append(i)
                        if len(eeg_channels) == 2:
                            break
        if len(eeg_channels) < 2:
            eeg_channels = [0, 1]

        # Read raw data
        raw_signals = []
        for ch_idx in eeg_channels:
            sig = f.readSignal(ch_idx)
            raw_signals.append(sig)
        f.close()
        data = np.array(raw_signals)

        # Downsample 200 Hz -> 100 Hz
        if FS_RAW != FS:
            from scipy.signal import resample
            n_samples = int(data.shape[1] * FS / FS_RAW)
            data_resampled = np.zeros((data.shape[0], n_samples))
            for ch in range(data.shape[0]):
                data_resampled[ch] = resample(data[ch], n_samples)
            data = data_resampled

        # Read staging labels
        labels = np.loadtxt(txt_path, dtype=int)
        labels = np.array([ISRUC_LABEL_MAP.get(l, 0) for l in labels])

        # 30s epoch segmentation
        n_epochs = data.shape[1] // SAMPLES_PER_EPOCH
        epochs_data, epochs_labels = [], []
        for i in range(min(n_epochs, len(labels))):
            start = i * SAMPLES_PER_EPOCH
            end = start + SAMPLES_PER_EPOCH
            epoch_eeg = data[:, start:end]
            epochs_data.append(epoch_eeg)
            epochs_labels.append(labels[i])

        if len(epochs_data) == 0:
            return None, None, None
        return np.array(epochs_data), np.array(epochs_labels), None
    except Exception as e:
        print(f"    Load failed: {e}")
        return None, None, None


def load_dreams_data(subject_id, night=1):
    """Load DREAMS data

    DREAMS DatabasePatients file structure:
    - patientX.edf (PSG file)
    - HypnogramAASM_patientX.txt (AASM format label file)
    """
    # Extract the numeric part from subject_id (e.g., "subject2" → "2")
    subject_num = subject_id.replace('subject', '') if subject_id.startswith('subject') else subject_id
    
    # Find PSG file (patient1.edf, patient2.edf, ...)
    psg_pattern = f"patient{subject_num}.edf"
    psg_files = glob.glob(os.path.join(DREAMS_DIR, psg_pattern))
    if not psg_files:
        # Try other possible naming formats
        alt_patterns = [
            f"Patient{subject_num}.edf",
            f"{subject_num}.edf",
        ]
        for pat in alt_patterns:
            psg_files = glob.glob(os.path.join(DREAMS_DIR, pat))
            if psg_files:
                break
    if not psg_files:
        return None, None, None

    # Find Hypnogram file (HypnogramAASM_patientX.txt)
    hyp_pattern = f"HypnogramAASM_patient{subject_num}.txt"
    hyp_files = glob.glob(os.path.join(DREAMS_DIR, hyp_pattern))
    if not hyp_files:
        # Try other formats
        alt_patterns = [
            f"HypnogramAASM_patient{subject_num}.txt",
            f"HypnogramR&K_patient{subject_num}.txt",
            f"HypnogramAASM_{subject_num}.txt",
            f"HypnogramR&K_{subject_num}.txt",
        ]
        for pat in alt_patterns:
            hyp_files = glob.glob(os.path.join(DREAMS_DIR, pat))
            if hyp_files:
                break
    if not hyp_files:
        return None, None, None

    try:
        # Load EEG data
        raw = mne.io.read_raw_edf(psg_files[0], preload=True, verbose=False)

        # DREAMS sampling rate is typically 200 Hz, check and downsample
        dreams_fs = raw.info['sfreq']
        if dreams_fs != FS:
            raw.resample(FS)

        # Select EEG channels (DREAMS typically has C3-A2, C4-A1, etc.)
        eeg_channels = [ch for ch in raw.ch_names if any(k in ch.upper() for k in ['EEG', 'C3', 'C4', 'O1', 'O2', 'FP1', 'FP2'])]
        if len(eeg_channels) == 0:
            # If no EEG-labeled channels found, try all channels
            eeg_channels = raw.ch_names
        
        target_channels = []
        for ch in ['EEG C3-A2', 'EEG C4-A1', 'C3-A2', 'C4-A1', 'EEG O1-A2', 'EEG O2-A1', 
                   'EEG FP1-A2', 'EEG FP2-A1', 'FP1-A2', 'FP2-A1']:
            if ch in eeg_channels:
                target_channels.append(ch)
        if len(target_channels) < 2:
            target_channels = eeg_channels[:2]
        if len(target_channels) < 2:
            print(f"      ❌ Not enough EEG channels: {raw.ch_names}")
            return None, None, None

        raw.pick_channels(target_channels)
        data = raw.get_data()

        # Load labels - try EDF annotation file
        epochs_labels = []
        found_labels = False

        for hyp_file in hyp_files:
            if hyp_file.endswith('.edf'):
                # Try EDF format labels
                try:
                    annotations = mne.read_annotations(hyp_file)
                    labels_per_sample = np.full(data.shape[1], -1, dtype=int)
                    for onset, duration, desc in zip(annotations.onset, annotations.duration, annotations.description):
                        start_sample = int(onset * FS)
                        end_sample = min(int((onset + duration) * FS), data.shape[1])
                        # DREAMS label mapping
                        desc = desc.strip().upper()
                        if 'W' in desc or 'WAKE' in desc:
                            mapped_label = 0
                        elif 'N1' in desc or 'S1' in desc or '1' in desc:
                            mapped_label = 1
                        elif 'N2' in desc or 'S2' in desc or '2' in desc:
                            mapped_label = 2
                        elif 'N3' in desc or 'S3' in desc or 'S4' in desc or '3' in desc or '4' in desc:
                            mapped_label = 3
                        elif 'R' in desc or 'REM' in desc or '5' in desc:
                            mapped_label = 4
                        else:
                            mapped_label = -1
                        if mapped_label >= 0:
                            labels_per_sample[start_sample:end_sample] = mapped_label

                    # Convert to epoch labels
                    n_epochs = data.shape[1] // SAMPLES_PER_EPOCH
                    for i in range(n_epochs):
                        start = i * SAMPLES_PER_EPOCH
                        end = start + SAMPLES_PER_EPOCH
                        epoch_labels = labels_per_sample[start:end]
                        valid_labels = epoch_labels[epoch_labels >= 0]
                        if len(valid_labels) == 0:
                            continue
                        label = np.bincount(valid_labels).argmax()
                        epochs_labels.append(label)
                    if len(epochs_labels) > 0:
                        found_labels = True
                        break
                except Exception:
                    pass

            if hyp_file.endswith('.txt') and not found_labels:
                # Try TXT format labels (DREAMS AASM/R&K format)
                try:
                    with open(hyp_file, 'r') as f:
                        lines = f.readlines()
                    # Skip header (e.g. [HypnogramAASM])
                    raw_labels = []
                    for line in lines:
                        line = line.strip()
                        if line.startswith('['):
                            continue  # Skip header lines
                        try:
                            raw_labels.append(int(line))
                        except ValueError:
                            continue
                    
                    # DREAMS AASM label conversion (special encoding!)
                    # Per DREAMS documentation:
                    # 5 = Wake, 4 = REM, 3 = N1, 2 = N2, 1 = N3
                    # 0, -1, -2, -3 = unknown (treated as Wake)
                    # 
                    # Note: DREAMS AASM labels are one value per 5 seconds, need to merge into 30s epochs
                    converted_labels = []
                    for l in raw_labels:
                        if l == 5:
                            converted_labels.append(0)  # Wake
                        elif l == 3:
                            converted_labels.append(1)  # N1
                        elif l == 2:
                            converted_labels.append(2)  # N2
                        elif l == 1:
                            converted_labels.append(3)  # N3
                        elif l == 4:
                            converted_labels.append(4)  # REM
                        else:
                            converted_labels.append(0)  # Unknown labels treated as Wake
                    
                    # Merge 5s labels into 30s epochs (6 labels of 5s each → 1 epoch of 30s)
                    epochs_labels = []
                    labels_per_epoch = 6  # 5s * 6 = 30s
                    for i in range(0, len(converted_labels), labels_per_epoch):
                        epoch_labels = converted_labels[i:i+labels_per_epoch]
                        if len(epoch_labels) == 0:
                            continue
                        # Use mode as epoch label
                        counts = [0]*5
                        for label in epoch_labels:
                            if 0 <= label < 5:
                                counts[label] += 1
                        epoch_label = counts.index(max(counts))
                        epochs_labels.append(epoch_label)
                    
                    if len(epochs_labels) > 0:
                        found_labels = True
                        break
                except Exception as e:
                    print(f"      ❌ Failed to parse labels: {e}")
                    pass

        if not found_labels:
            return None, None, None

        # Segment EEG data into epochs
        n_epochs = min(data.shape[1] // SAMPLES_PER_EPOCH, len(epochs_labels))
        epochs_data = []
        for i in range(n_epochs):
            start = i * SAMPLES_PER_EPOCH
            end = start + SAMPLES_PER_EPOCH
            epoch_eeg = data[:, start:end]
            if epoch_eeg.shape[1] < SAMPLES_PER_EPOCH:
                epoch_eeg = np.pad(epoch_eeg, ((0, 0), (0, SAMPLES_PER_EPOCH - epoch_eeg.shape[1])), mode='edge')
            epochs_data.append(epoch_eeg)

        epochs_labels = epochs_labels[:n_epochs]
        if len(epochs_data) == 0:
            return None, None, None

        return np.array(epochs_data), np.array(epochs_labels), None

    except Exception as e:
        print(f"    Load failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


# =====================================================================
# Part 2: Clinical Sleep Metric Computation
# =====================================================================

def compute_sleep_clinical_metrics(epochs_labels, epoch_length_s=EPOCH_LENGTH_S, sol_min_override=None):
    """Compute clinical sleep architecture metrics(using sleep window: first sleep → last awakening)

    Changes from v1:
    - TIB = time window from first to last Sleep epoch
    - SE = TST / TIB (not full recording)
    - WASO = Wake epochs within sleep window
    - SOL = from recording start to first sleep epoch (standard clinical definition)

    Args:
        epochs_labels: array of sleep stage labels (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM)
        epoch_length_s: epoch length in seconds (default 30)
        sol_min_override: if provided, use this SOL value instead of calculating from labels
                         (useful when data has been windowed and first epoch is not recording start)

    Returns:
        dict: {
            'TST_min': Total Sleep Time (min, non-Wake only),
            'TIB_min': Time in Bed (min, within sleep window),
            'sleep_efficiency_pct': Sleep Efficiency (%),
            'WASO_min': Wake After Sleep Onset (min),
            'SOL_min': Sleep Onset Latency (min, None = no sleep throughout night),
            'stage_min': {minutes per stage},
            'stage_pct_tst': {percentage of each sleep stage in TST (excluding Wake)},
            'stage_pct_tib': {percentage of each stage in TIB},
            'transition_count': total stage transition count,
            'transition_rate_per_hour': transitions per hour,
        }
    """
    n_epochs = len(epochs_labels)
    total_recording_min = n_epochs * epoch_length_s / 60
    is_sleep = epochs_labels != 0  # True for non-Wake
    sleep_indices = np.where(is_sleep)[0]

    if len(sleep_indices) == 0:
        return {
            'TST_min': 0, 'TIB_min': round(total_recording_min, 1),
            'sleep_efficiency_pct': 0, 'WASO_min': 0, 'SOL_min': None,  # None = no sleep throughout night
            'stage_min': {s: 0 for s in STAGE_LIST},
            'stage_pct_tst': {s: 0 for s in STAGE_LIST if s != 'Wake'},
            'stage_pct_tib': {s: 0 for s in STAGE_LIST},
            'transition_count': 0, 'transition_rate_per_hour': 0,
        }

    # ============ SOL: Standard Clinical Definition =============
    if sol_min_override is not None:
        sol_min = sol_min_override
    else:
        first_sleep_for_sol = sleep_indices[0]
        sol_min = first_sleep_for_sol * epoch_length_s / 60
    # =========================================

    # Find main sleep period onset (exclude daytime drowsiness/naps, for TIB/TST/WASO calculation)
    # Strategy: find continuous sleep block containing N3/REM, trace back to its actual start
    min_consecutive = 5
    first_sleep = sleep_indices[0]

    # Build sleep block list: (start, end) ranges of consecutive non-Wake epochs
    sleep_blocks = []
    in_block = False
    block_start = 0
    for i in range(n_epochs):
        if is_sleep[i] and not in_block:
            block_start = i
            in_block = True
        elif not is_sleep[i] and in_block:
            sleep_blocks.append((block_start, i - 1))
            in_block = False
    if in_block:
        sleep_blocks.append((block_start, n_epochs - 1))

    # Select main sleep block: longest block with length ≥ 30min (60 epochs) and containing N3/REM
    deep_sleep_labels = {3, 4}
    main_block = None
    for start, end in sleep_blocks:
        block_len = end - start + 1
        if block_len < 60:  # < 30 min, ignore
            continue
        block_labels = set(epochs_labels[start:end + 1])
        if deep_sleep_labels & block_labels:  # Contains N3 or REM
            if main_block is None or block_len > (main_block[1] - main_block[0] + 1):
                main_block = (start, end)

    if main_block is not None:
        first_sleep = main_block[0]
    else:
        # Fallback: use first sleep epoch containing deep sleep
        deep_sleep_idx = np.where((epochs_labels == 3) | (epochs_labels == 4))[0]
        if len(deep_sleep_idx) > 0:
            first_deep = deep_sleep_idx[0]
            # Trace back from this deep sleep epoch to continuous sleep onset
            first_sleep = first_deep
            while first_sleep > 0 and is_sleep[first_sleep - 1]:
                first_sleep -= 1

    last_sleep = sleep_indices[-1]

    # TIB = sleep window (first to last sleep epoch)
    tib_epochs = last_sleep - first_sleep + 1
    tib_min = tib_epochs * epoch_length_s / 60

    # TST: sleep epochs within TIB window
    window_labels = epochs_labels[first_sleep:last_sleep + 1]
    sleep_epochs_in_window = np.sum(window_labels != 0)
    tst_min = sleep_epochs_in_window * epoch_length_s / 60

    # SE
    se_pct = (tst_min / tib_min) * 100 if tib_min > 0 else 0

    # WASO: Wake epochs within TIB
    waso_epochs = np.sum(window_labels == 0)
    waso_min = waso_epochs * epoch_length_s / 60

    # Stage durations (within TIB window)
    stage_min = {}
    stage_pct_tst = {}
    stage_pct_tib = {}
    sleep_stages = [k for k, v in STAGE_NAMES.items() if v != 0]  # N1/N2/N3/REM

    for stage_id, stage_name in STAGE_NAMES.items():
        cnt_in_window = int(np.sum(window_labels == stage_id))
        mins = cnt_in_window * epoch_length_s / 60
        stage_min[stage_name] = mins
        stage_pct_tib[stage_name] = round(100 * cnt_in_window / tib_epochs, 2)

        if stage_id != 0:  # Sleep stages only (not Wake)
            stage_pct_tst[stage_name] = round(
                100 * cnt_in_window / sleep_epochs_in_window, 2
            ) if sleep_epochs_in_window > 0 else 0

    # Transitions within TIB
    transition_count = int(np.sum(window_labels[1:] != window_labels[:-1]))
    transition_rate = round(transition_count / (tst_min / 60), 2) if tst_min > 0 else 0

    return {
        'TST_min': round(tst_min, 1),
        'TIB_min': round(tib_min, 1),
        'sleep_efficiency_pct': round(se_pct, 1),
        'WASO_min': round(waso_min, 1),
        'SOL_min': round(sol_min, 1),
        'stage_min': stage_min,
        'stage_pct_tst': stage_pct_tst,
        'stage_pct_tib': stage_pct_tib,
        'transition_count': transition_count,
        'transition_rate_per_hour': transition_rate,
    }


def compute_transition_matrix(labels_sequence):
    """Compute sleep stage transition probability matrix (5×5)
    rows=current stage, cols=next stage
    """
    n_stages = 5
    counts = np.zeros((n_stages, n_stages), dtype=int)
    for i in range(len(labels_sequence) - 1):
        curr, nxt = labels_sequence[i], labels_sequence[i + 1]
        if 0 <= curr < n_stages and 0 <= nxt < n_stages:
            counts[curr, nxt] += 1

    matrix = {}
    for i, src in enumerate(STAGE_LIST):
        row_total = counts[i].sum()
        matrix[src] = {}
        for j, dst in enumerate(STAGE_LIST):
            if row_total > 0:
                matrix[src][dst] = round(int(counts[i, j]) / row_total, 4)
            else:
                matrix[src][dst] = 0.0
    return matrix


# =====================================================================
# Part 3: Power Spectrum Computation
# =====================================================================

def compute_power_spectra(epochs_data, epochs_labels, fs=FS):
    """Compute absolute power, relative power, and power ratios

    Returns:
        dict: {
            'absolute_power': {stage: {band: {mean, std}}},
            'relative_power': {stage: {band: {mean, std}}},
            'power_ratios': {stage: {ratio_name: value}},
        }
    """
    samples_per_epoch = fs * EPOCH_LENGTH_S
    stage_powers = {s: {b: [] for b in BAND_NAMES} for s in STAGE_LIST}

    for i in range(len(epochs_labels)):
        stage = STAGE_NAMES.get(epochs_labels[i])
        if stage is None:
            continue
        epoch_data = epochs_data[i]
        for low, high, bname in FREQ_BANDS:
            bp = []
            for ch in range(epoch_data.shape[0]):
                f, pxx = signal.welch(epoch_data[ch], fs=fs, nperseg=min(128, samples_per_epoch))
                mask = (f >= low) & (f <= high)
                bp.append(float(np.mean(pxx[mask])) if np.any(mask) else 0.0)
            stage_powers[stage][bname].append(np.mean(bp))

    # --- Absolute Power ---
    absolute_power = {}
    for s in STAGE_LIST:
        absolute_power[s] = {}
        for b in BAND_NAMES:
            vals = stage_powers[s][b]
            if vals:
                absolute_power[s][b] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                    'n_epochs': len(vals),
                }

    # --- Relative Power (band/total×100%) ---
    relative_power = {}
    for s in STAGE_LIST:
        relative_power[s] = {}
        first_band = BAND_NAMES[0]
        if first_band not in stage_powers.get(s, {}) or len(stage_powers[s][first_band]) == 0:
            continue
        total_power_per_epoch = []
        # Compute total power per epoch (sum of all bands)
        n_ep = len(stage_powers[s][first_band])
        for ep_idx in range(n_ep):
            total = sum(stage_powers[s][b][ep_idx] for b in BAND_NAMES)
            total_power_per_epoch.append(total if total > 0 else 1e-10)

        for b in BAND_NAMES:
            vals = stage_powers[s][b]
            if not vals:
                continue
            rel_vals = [100 * vals[i] / total_power_per_epoch[i] for i in range(len(vals))]
            relative_power[s][b] = {
                'mean': float(np.mean(rel_vals)),
                'std': float(np.std(rel_vals)),
                'n_epochs': len(vals),
            }

    # --- Power Ratios (clinically common) ---
    power_ratios = {}
    for s in STAGE_LIST:
        abs_vals = absolute_power.get(s, {})
        if not abs_vals:
            continue
        ap = {b: abs_vals[b]['mean'] for b in BAND_NAMES if b in abs_vals}
        ratios = {
            'Alpha_Delta_ratio': round(ap.get('Alpha', 0) / (ap.get('Delta', 1e-10)), 4),
            'Theta_Beta_ratio': round((ap.get('Low_Theta', 0) + ap.get('High_Theta', 0)) /
                                       (ap.get('Beta', 1e-10)), 4),
            'Alpha_Theta_ratio': round(ap.get('Alpha', 0) / (ap.get('Low_Theta', 0) + ap.get('High_Theta', 1e-10)), 4),
            'Sigma_Delta_ratio': round(ap.get('Sigma', 0) / (ap.get('Delta', 1e-10)), 4),
            'Slow_Fast_ratio': round(ap.get('Delta', 0) /
                                      (ap.get('Beta', 1e-10) + ap.get('Gamma', 1e-10)), 4),
            'Alpha_power': round(ap.get('Alpha', 0), 4),
            'Delta_power': round(ap.get('Delta', 0), 4),
            'Theta_power': round(ap.get('Low_Theta', 0) + ap.get('High_Theta', 0), 4),
            'Beta_power': round(ap.get('Beta', 0), 4),
            'Gamma_power': round(ap.get('Gamma', 0), 4),
        }
        power_ratios[s] = ratios

    return {
        'absolute_power': absolute_power,
        'relative_power': relative_power,
        'power_ratios': power_ratios,
    }


# =====================================================================
# Part 4: Statistical Tests
# =====================================================================

def cohens_d(x, y):
    """Compute Cohen's d effect size"""
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return 0.0
    s1, s2 = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_se = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled_se == 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled_se)


def perform_statistical_tests(group_metrics_dict):
    """Perform statistical tests for three groups

    Args:
        group_metrics_dict: {group_name: [values_per_subject]}

    Returns:
        dict with KW test, post-hoc, Cohen's d
    """
    group_names = list(group_metrics_dict.keys())
    data_groups = [np.array(v) for v in group_metrics_dict.values()]

    # Filter empty groups
    valid = [(g, d) for g, d in zip(group_names, data_groups) if len(d) > 0]
    if len(valid) < 2:
        return {'error': 'insufficient_groups'}

    group_names = [v[0] for v in valid]
    data_groups = [v[1] for v in valid]

    # Kruskal-Wallis H test
    try:
        kw_stat, kw_p = stats.kruskal(*data_groups) if len(data_groups) >= 2 else (0, 1.0)
    except ValueError as e:
        # Handle "All numbers are identical" error
        if 'All numbers are identical' in str(e):
            kw_stat, kw_p = 0.0, 1.0
        else:
            return {'error': f'kruskal_error: {str(e)}'}

    # Post-hoc: Mann-Whitney U + Bonferroni
    post_hoc = {}
    n_comparisons = len(data_groups) * (len(data_groups) - 1) // 2
    for i in range(len(data_groups)):
        for j in range(i + 1, len(data_groups)):
            pair_key = f"{group_names[i]}_vs_{group_names[j]}"
            if len(data_groups[i]) > 0 and len(data_groups[j]) > 0:
                try:
                    u_stat, raw_p = stats.mannwhitneyu(data_groups[i], data_groups[j],
                                                       alternative='two-sided')
                    post_hoc[pair_key] = {
                        'U_statistic': int(u_stat),
                        'raw_p_value': float(raw_p),
                        'bonferroni_p': min(float(raw_p * n_comparisons), 1.0),
                        'cohens_d': cohens_d(data_groups[i], data_groups[j]),
                    }
                except ValueError as e:
                    # Handle Mann-Whitney U test error
                    post_hoc[pair_key] = {
                        'U_statistic': 0,
                        'raw_p_value': 1.0,
                        'bonferroni_p': 1.0,
                        'cohens_d': 0.0,
                        'error': str(e),
                    }

    return {
        'test': 'Kruskal-Wallis + Mann-Whitney U (Bonferroni corrected)',
        'n_groups': len(group_names),
        'group_names': group_names,
        'kw_H_statistic': float(kw_stat),
        'kw_p_value': float(kw_p),
        'kw_significant': bool(kw_p < 0.05),
        'post_hoc': post_hoc,
    }


def compute_all_statistics(all_group_metrics):
    """Run statistical tests for all numerical metrics

    Args:
        all_group_metrics: {
            'metric_name': {
                'menopausal_women': [val1, val2, ...],
                'young_women': [val1, val2, ...],
                'young_men': [val1, val2, ...],
            }
        }
    """
    results = {}
    for metric_name, group_dict in all_group_metrics.items():
        results[metric_name] = perform_statistical_tests(group_dict)
    return results


# =====================================================================
# Part 5: Feature Importance Extraction
# =====================================================================

def extract_feature_importance(model):
    """Extract feature importance from fitted MenoSCA_FBTS model

    Returns:
        dict: {
            'per_band_importance': {band_name: importance_score},
            'top_features': [(feature_idx, score), ...],
        }
    """
    importance_info = {}

    # Get feature scores from SelectKBest
    if hasattr(model, 'feature_selector') and model.feature_selector is not None:
        scores = model.feature_selector.scores_
        if scores is not None:
            # Features grouped by band (each band produces covariance features + temporal features)
            features_per_band = 140  # ~140 features per band (7-band version)
            band_scores = {}
            for idx, (low, high, bname) in enumerate(FREQ_BANDS):
                start = idx * features_per_band
                end = min(start + features_per_band, len(scores))
                if start < len(scores):
                    band_scores[bname] = float(np.mean(scores[start:end]))

            # Normalize
            total = sum(band_scores.values())
            if total > 0:
                band_scores = {k: round(v / total, 4) for k, v in band_scores.items()}

            importance_info['per_band_importance'] = band_scores

    # Get weights from ensemble classifier
    classifier_weights = None
    if hasattr(model, 'classifier') and model.classifier is not None:
        clf = model.classifier
        # Use estimators (no underscore): returns list of (name, model) tuples
        if hasattr(clf, 'estimators'):
            estimator_names = [e[0] for e in clf.estimators]
            if hasattr(clf, 'weights') and clf.weights is not None:
                classifier_weights = dict(zip(estimator_names, clf.weights))

    if classifier_weights:
        importance_info['classifier_weights'] = classifier_weights

    return importance_info


# =====================================================================
# Part 6: Per-Subject Accuracy (aggregated across folds)
# =====================================================================

def compute_per_subject_accuracy(subject_ids_for_epochs, epochs_labels, y_pred_all_folds, fold_test_indices):
    """Compute per-subject mean accuracy in cross-validation

    Args:
        subject_ids_for_epochs: [subj_id] subject ID for each epoch
        epochs_labels: true labels
        y_pred_all_folds: {fold: list of predictions}
        fold_test_indices: {fold: list of test indices}

    Returns:
        dict: {subject_id: {'accuracy': ..., 'n_test_epochs': ...}}
    """
    subj_correct = {}
    subj_total = {}

    for fold, test_idx in fold_test_indices.items():
        fold_preds = y_pred_all_folds[fold]
        for pos_in_fold, global_idx in enumerate(test_idx):
            subj = subject_ids_for_epochs[global_idx]
            if subj not in subj_correct:
                subj_correct[subj] = 0
                subj_total[subj] = 0
            subj_total[subj] += 1
            if fold_preds[pos_in_fold] == epochs_labels[global_idx]:
                subj_correct[subj] += 1

    result = {}
    for subj in sorted(subj_correct.keys()):
        result[str(subj)] = {
            'accuracy': round(subj_correct[subj] / subj_total[subj], 4) if subj_total[subj] > 0 else 0,
            'n_test_epochs': subj_total[subj],
            'correct': subj_correct[subj],
        }
    return result


# =====================================================================
# Part 7: Core Pipeline - Load Group Data
# =====================================================================

def load_group_data(group_name, config, subject_info):
    """Load complete data for one group (including all additional metrics)"""
    print(f"\n[{config['name']}] Loading data...")
    all_epochs, all_labels = [], []
    all_subject_ids = []
    subject_details, night_comparisons = [], []
    group_clinical_metrics = []

    # === Dynamic Subject Selection via Age Range ===
    # If config has age_range+gender AND subject_info has richer data than the
    # hardcoded list, dynamically filter all eligible subjects.
    # This is the key mechanism that makes the age-range expansion actually
    # increase sample size instead of being just metadata.
    age_range = config.get('age_range', None)
    gender = config.get('gender', None)
    subject_ids_to_process = config.get('subject_ids', [])
    use_dynamic = (age_range is not None and gender is not None
                   and len(subject_info) > len(subject_ids_to_process))
    if use_dynamic:
        age_min, age_max = age_range
        dynamic_ids = []
        for sid, info in subject_info.items():
            age = info.get('age', None)
            g = info.get('gender', None)
            if isinstance(age, (int, float)) and g == gender:
                if age_min <= age <= age_max:
                    dynamic_ids.append(sid)
        if dynamic_ids:
            try:
                dynamic_ids.sort(key=lambda x: int(x) if str(x).lstrip('-').isdigit() else str(x))
            except (ValueError, AttributeError):
                dynamic_ids.sort()
            subject_ids_to_process = dynamic_ids

    for subj_id in subject_ids_to_process:
        print(f"  Subject {subj_id}...", end=" ")
        if DATASET == 'isruc':
            epochs, labels, sol_override = load_isruc_data(subj_id, night=1)
        elif DATASET == 'dreams':
            epochs, labels, sol_override = load_dreams_data(subj_id, night=1)
        else:
            epochs, labels, sol_override = load_edf_data(subj_id, night=1)

        if epochs is not None:
            info = subject_info.get(subj_id, {})
            n1_clinical = compute_sleep_clinical_metrics(labels, sol_min_override=sol_override)
            stage_counts = {s: int(np.sum(labels == i)) for i, s in STAGE_NAMES.items()}
            stage_pcts = {s: round(100 * c / len(labels), 1) for s, c in stage_counts.items()}
            trans_matrix = compute_transition_matrix(labels)

            subj_entry = {
                'subject_id': str(subj_id),
                'age': info.get('age', 'N/A'),
                'gender': info.get('gender', 'N/A'),
                'diagnosis': info.get('diagnosis', None),
                'other_problems': info.get('other_problems', None),
                'nights_available': info.get('nights', [1]),
                'night1': {
                    'n_epochs': len(labels),
                    'sleep_stage_distribution': stage_counts,
                    'sleep_stage_percentages': stage_pcts,
                    'clinical_metrics': n1_clinical,
                    'transition_matrix': trans_matrix,
                },
            }

            n2_data = None
            if DATASET == 'isruc':
                epochs2, labels2, sol_override2 = load_isruc_data(subj_id, night=2)
            elif DATASET == 'dreams':
                epochs2, labels2, sol_override2 = None, None, None
            else:
                epochs2, labels2, sol_override2 = load_edf_data(subj_id, night=2)
            if epochs2 is not None:
                n2_clinical = compute_sleep_clinical_metrics(labels2, sol_min_override=sol_override2)
                stage_counts2 = {s: int(np.sum(labels2 == i)) for i, s in STAGE_NAMES.items()}
                trans_matrix2 = compute_transition_matrix(labels2)
                subj_entry['night2'] = {
                    'n_epochs': len(labels2),
                    'sleep_stage_distribution': stage_counts2,
                    'sleep_stage_percentages': {s: round(100 * c / len(labels2), 1) for s, c in stage_counts2.items()},
                    'clinical_metrics': n2_clinical,
                    'transition_matrix': trans_matrix2,
                }
                n2_data = labels2
                print(f"N1:{len(labels)} N2:{len(labels2)} OK")
            else:
                print(f"{len(labels)} OK")

            # Merge two nights (ensure correct order: N1 first, N2 second)
            all_epochs.append(epochs)
            all_labels.append(labels)
            all_subject_ids.extend([str(subj_id)] * len(labels))
            
            if epochs2 is not None:
                all_epochs.append(epochs2)
                all_labels.append(labels2)
                all_subject_ids.extend([str(subj_id)] * len(labels2))

            # Night-to-night comparison
            if n2_data is not None:
                night_comparisons.append({
                    'subject_id': str(subj_id),
                    'night1_TST': n1_clinical['TST_min'],
                    'night2_TST': n2_clinical['TST_min'],
                    'night1_SE': n1_clinical['sleep_efficiency_pct'],
                    'night2_SE': n2_clinical['sleep_efficiency_pct'],
                    'night1_SOL': n1_clinical['SOL_min'],
                    'night2_SOL': n2_clinical['SOL_min'],
                })
            group_clinical_metrics.append(n1_clinical)
            subject_details.append(subj_entry)
        else:
            print("FAILED")

    if not all_epochs:
        print(f"  FAILED to load {config['name']} data")
        return None, None, None, None, None

    X = np.concatenate(all_epochs, axis=0)
    y = np.concatenate(all_labels, axis=0)
    subj_ids_arr = np.array(all_subject_ids)

    print(f"\n  Total: {X.shape[0]} epochs, {len(subject_details)} subjects")
    for i, name in enumerate(STAGE_LIST):
        cnt = int(np.sum(y == i))
        print(f"  {name}: {cnt} ({100*cnt/len(y):.1f}%)")

    print("  Computing power spectra...")
    power_spectra = compute_power_spectra(X, y)

    # Group-level transition matrix (concatenate all epochs)
    group_transition = compute_transition_matrix(y)

    # Aggregate clinical metrics for statistics
    agg_clinical = {}
    metrics_keys = ['TST_min', 'sleep_efficiency_pct', 'WASO_min', 'SOL_min',
                    'transition_rate_per_hour']
    for mk in metrics_keys:
        agg_clinical[mk] = [cm[mk] for cm in group_clinical_metrics]

    # Sleep stage percentages (stage_pct_tst excludes Wake)
    for s in STAGE_LIST:
        if s == 'Wake':
            continue  # stage_pct_tst excludes Wake
        agg_clinical[f'{s}_pct'] = [cm['stage_pct_tst'][s] for cm in group_clinical_metrics]

    # Power ratio aggregation
    for s in STAGE_LIST:
        if s in power_spectra['power_ratios']:
            for ratio_name in ['Alpha_Delta_ratio', 'Theta_Beta_ratio', 'Sigma_Delta_ratio']:
                key = f'{s}_{ratio_name}'
                # Per-subject values needed here but only group-level available -> annotate single-group in stats phase later
                pass

    meta = {
        'subjects': subject_details,
        'power_spectra': power_spectra,
        'transition_matrix': group_transition,
        'n_subjects': len(subject_details),
        'clinical_metrics_agg': agg_clinical,
        'night_comparisons': night_comparisons,
        'subject_ids_arr': subj_ids_arr,
    }
    return X, y, meta, group_clinical_metrics, all_subject_ids


# =====================================================================
# Part 8: Training + Per-Subject Accuracy
# =====================================================================

def run_single_group_experiment(X, y, group_name, group_config,
                                subject_ids_arr=None):
    """Run experiment for a single group"""
    print(f"\n{'='*70}")
    print(f"[{group_config['name']}] Sleep staging ({DATASET})")
    print(f"{'='*70}")

    logo = LeaveOneGroupOut()
    n_subjects = len(np.unique(subject_ids_arr)) if subject_ids_arr is not None else 1

    results = {'accuracies': [], 'f1_scores': [], 'reports': [], 'confusion_matrices': []}
    fold_times, fold_phase_times = [], []
    y_pred_all, test_idx_all = {}, {}
    per_fold_models = []

    groups = np.asarray(subject_ids_arr) if subject_ids_arr is not None else np.arange(len(y))
    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups)):
        subj_in_test = np.unique(groups[test_idx])
        print(f"\n  Testing Subject {subj_in_test[0]} (Fold {fold + 1}/{n_subjects})...")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = MenoSCA_FBTS(
            n_bands=7, estimator='oas', metric='riemann',
            classifier='ensemble', n_features=200, fs=FS,
            temporal_smoothing=True, smoothing_window=3,
            enable_menopause_features=True
        )

        t0 = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - t0

        t0 = time.time()
        y_pred = model.predict(X_test)
        predict_time = time.time() - t0
        fold_time = fit_time + predict_time

        fold_times.append(fold_time)
        fold_phase_times.append({
            'fit': round(fit_time, 2),
            'predict': round(predict_time, 2),
            'total': round(fold_time, 2),
        })

        # Save per-fold info
        y_pred_all[fold] = y_pred
        test_idx_all[fold] = test_idx

        acc = accuracy_score(y_test, y_pred)
        results['accuracies'].append(acc)
        report = classification_report(y_test, y_pred, output_dict=True)
        results['reports'].append(report)
        results['f1_scores'].append(report['macro avg']['f1-score'])
        results['confusion_matrices'].append(confusion_matrix(y_test, y_pred).tolist())
        per_fold_models.append(model)

        print(f"    Acc: {acc:.4f}, Fit: {fit_time:.1f}s, Pred: {predict_time:.1f}s")

    mean_acc, std_acc = np.mean(results['accuracies']), np.std(results['accuracies'])
    mean_f1, std_f1 = np.mean(results['f1_scores']), np.std(results['f1_scores'])
    mean_time = np.mean(fold_times)

    print(f"\n[{group_config['name']}] Mean Acc: {mean_acc:.4f}±{std_acc:.4f}, F1: {mean_f1:.4f}±{std_f1:.4f}")

    # Per-subject accuracy
    per_subj_acc = None
    if subject_ids_arr is not None:
        per_subj_acc = compute_per_subject_accuracy(
            subject_ids_arr, y, y_pred_all, test_idx_all
        )

    # Feature importance (use last fold model)
    feature_importance = extract_feature_importance(per_fold_models[-1])
    # If multiple models, average feature importance
    if len(per_fold_models) > 1:
        all_band_imps = []
        for m in per_fold_models:
            fi = extract_feature_importance(m)
            if 'per_band_importance' in fi:
                all_band_imps.append(fi['per_band_importance'])
        if all_band_imps:
            avg_imp = {}
            for bname in all_band_imps[0]:
                vals = [bi[bname] for bi in all_band_imps]
                avg_imp[bname] = round(float(np.mean(vals)), 4)
            feature_importance['per_band_importance_avg'] = avg_imp

    return {
        'group': group_name,
        'name': group_config['name'],
        'total_epochs': len(y),
        'class_distribution': {name: int(np.sum(y == i)) for i, name in enumerate(STAGE_LIST)},
        'class_percentages': {name: round(100 * int(np.sum(y == i)) / len(y), 1) for i, name in enumerate(STAGE_LIST)},
        'mean_accuracy': float(mean_acc),
        'std_accuracy': float(std_acc),
        'mean_f1': float(mean_f1),
        'std_f1': float(std_f1),
        'mean_time': float(mean_time),
        'fold_phase_times': fold_phase_times,
        'fold_results': results,
        'per_subject_accuracy': per_subj_acc,
        'feature_importance': feature_importance,
    }


# =====================================================================
# Part 9: Main Function
# =====================================================================

def run_comparative_experiment():
    """Run full three-group comparative experiment"""
    if DATASET == 'isruc':
        dataset_label = 'ISRUC-Sleep Subgroup-I'
    elif DATASET == 'dreams':
        dataset_label = 'DREAMS Subjects Database'
    else:
        dataset_label = 'Sleep-EDF'
    print("=" * 70)
    print(f"MenoSCA-FBTS Three-Group Experiment ({dataset_label})")
    print("=" * 70)
    print(f"Dataset: {dataset_label}")
    print("Evaluation: Leave-One-Group-Out Cross-Validation")
    print("=" * 70)

    # Load metadata
    print("\n[1/4] Loading subject metadata...")
    subject_info = load_subject_metadata()
    print(f"  {len(subject_info)} subject(s) loaded")

    # Load group data
    print("\n[2/4] Loading group data...")
    group_data = {}
    all_group_clinical = {}

    for group_name, config in GROUP_CONFIG.items():
        X, y, meta, clinical_list, subj_ids = load_group_data(
            group_name, config, subject_info)
        if X is None:
            continue
        group_data[group_name] = (X, y, meta, config, subj_ids)
        all_group_clinical[group_name] = {
            'clinical_list': clinical_list,
            'name': config['name'],
        }

    if not group_data:
        print("No data loaded, aborting.")
        return

    # Run classification
    print("\n[3/4] Running classification (serial)...")
    all_results = {}

    for group_name, (X, y, meta, config, subj_ids) in group_data.items():
        try:
            res = run_single_group_experiment(X, y, group_name, config, subj_ids)
            res['subject_details'] = meta['subjects']
            res['power_spectra'] = meta['power_spectra']
            res['transition_matrix'] = meta['transition_matrix']
            res['n_subjects'] = meta['n_subjects']
            res['night_comparisons'] = meta['night_comparisons']
            all_results[group_name] = res
            print(f"OK - {res['name']} completed!")
        except Exception as e:
            print(f"FAILED - {group_name}: {e}")
            import traceback
            traceback.print_exc()

    # Statistical tests
    print("\n[4/4] Running statistical tests...")
    stat_metrics = {}
    for group_name, clinical_info in all_group_clinical.items():
        cl = clinical_info['clinical_list']
        for mk in ['TST_min', 'sleep_efficiency_pct', 'WASO_min', 'SOL_min',
                    'transition_rate_per_hour']:
            if mk not in stat_metrics:
                stat_metrics[mk] = {}
            # Filter None values (no sleep throughout night)
            stat_metrics[mk][group_name] = [cm[mk] for cm in cl if cm[mk] is not None]
        for s in STAGE_LIST:
            if s == 'Wake':
                continue
            key = f'{s}_pct_tst'
            if key not in stat_metrics:
                stat_metrics[key] = {}
            stat_metrics[key][group_name] = [cm['stage_pct_tst'][s] for cm in cl]

    updated_stat_metrics = {}
    for key, group_dict in stat_metrics.items():
        vals = [v for v in group_dict.values() if len(v) > 0]
        if len(vals) >= 2:
            # Relax sample size requirement: at least 1 valid sample per group for statistical test
            valid = {g: v for g, v in group_dict.items() if len(v) >= 1}
            if len(valid) >= 2:
                updated_stat_metrics[key] = valid

    statistical_results = compute_all_statistics(updated_stat_metrics)

    # Print summary table
    print("\n" + "=" * 70)
    print("Three-Group Comparison")
    print("=" * 70)
    print(f"\n{'Group':<20} {'n':<4} {'Epochs':<10} {'Accuracy':<16} {'F1':<14} {'SE(%)':<10}")
    print("-" * 74)
    for gn, res in all_results.items():
        se_vals = [cm['sleep_efficiency_pct'] for cm in all_group_clinical[gn]['clinical_list']]
        se_mean = np.mean(se_vals)
        se_std = np.std(se_vals)
        print(f"{res['name']:<20} {res['n_subjects']:<4} {res['total_epochs']:<10} "
              f"{res['mean_accuracy']*100:<6.2f}±{res['std_accuracy']*100:<4.2f} "
              f"{res['mean_f1']*100:<6.2f}±{res['std_f1']*100:<4.2f} "
              f"{se_mean:<5.1f}±{se_std:<4.1f}")

    # Save results
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(sys_path, OUTPUT_DIR)
    os.makedirs(results_dir, exist_ok=True)
    save_name = f'paper_results_{DATASET}_{timestamp}.json'
    save_path = os.path.join(results_dir, save_name)

    output = {
        'experiment_info': {
            'description': f'MenoSCA-FBTS Three-Group Comparison ({dataset_label})',
            'algorithm': 'MenoSCA-FBTS (sca_fbts_woman)',
            'dataset': dataset_label,
            'sampling_rate': FS,
            'raw_sampling_rate': FS_RAW,
            'epoch_length_s': EPOCH_LENGTH_S,
            'n_folds': 'LOGO (Leave-One-Group-Out)',
            'freq_bands': [{'low': l, 'high': h, 'name': n} for l, h, n in FREQ_BANDS],
            'timestamp': timestamp,
        },
        'group_comparison': {
            g: {
                'name': r['name'],
                'n_subjects': r['n_subjects'],
                'total_epochs': r['total_epochs'],
                'class_distribution': r['class_distribution'],
                'class_percentages': r['class_percentages'],
                'mean_accuracy': r['mean_accuracy'],
                'std_accuracy': r['std_accuracy'],
                'mean_f1': r['mean_f1'],
                'std_f1': r['std_f1'],
                'mean_time': r['mean_time'],
                'fold_phase_times': r['fold_phase_times'],
            }
            for g, r in all_results.items()
        },
        'subject_details': {
            g: r['subject_details'] for g, r in all_results.items()
        },
        'sleep_clinical_metrics': {
            g: {
                'summary': {
                    mk: {
                        'mean': round(float(np.mean(cl_list)), 2) if cl_list else None,
                        'std': round(float(np.std(cl_list)), 2) if len(cl_list) > 1 else None,
                        'values': [round(v, 2) for v in cl_list if v is not None],
                    }
                    for mk, cl_list in {
                        mk: [cm[mk] for cm in info['clinical_list'] if cm[mk] is not None]
                        for mk in ['TST_min', 'sleep_efficiency_pct', 'WASO_min', 'SOL_min',
                                    'transition_rate_per_hour']
                    }.items()
                },
                'stage_pct_tst': {
                    s: {
                        'mean': round(float(np.mean([cm['stage_pct_tst'][s] for cm in info['clinical_list']])), 2),
                        'values': [cm['stage_pct_tst'][s] for cm in info['clinical_list']],
                    }
                    for s in STAGE_LIST if s != 'Wake'
                },
            }
            for g, info in all_group_clinical.items() if g in all_results
        },
        'power_spectra': {
            g: r['power_spectra'] for g, r in all_results.items()
        },
        'transition_matrices': {
            g: r['transition_matrix'] for g, r in all_results.items()
        },
        'feature_importance': {
            g: r['feature_importance'] for g, r in all_results.items()
        },
        'per_subject_accuracy': {
            g: r['per_subject_accuracy'] for g, r in all_results.items()
        },
        'night_comparisons': {
            g: r['night_comparisons'] for g, r in all_results.items()
        },
        'statistical_tests': statistical_results,
        'full_fold_results': {
            g: r['fold_results'] for g, r in all_results.items()
        },
    }

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    file_size_kb = os.path.getsize(save_path) / 1024
    print(f"\n{'=' * 70}")
    print(f"Results saved: {save_path}")
    print(f"  Size: {file_size_kb:.1f} KB")
    print(f"{'=' * 70}")

    return all_results, statistical_results


if __name__ == '__main__':
    run_comparative_experiment()