"""
SOTA Sleep Staging Methods Comparison Experiment
===================================
Comparison methods:
1. Random Forest (Traditional ML baseline)
2. EEGNet from braindecode (Classic deep learning)
3. TinySleepNet (Lightweight 30s epoch deep learning)
4. MiniRocket (Latest efficient method)
5. YASA (Public tool)
6. MenoSCA-FBTS (Our method)

Dataset: Sleep-EDF Menopausal Women

Command line arguments:
    --algorithms: Specify algorithms to test, separated by comma.
                  Available: rf, eegnet, tinysleepnet, minirocket, yasa, menoscafbts
                  Example: --algorithms rf,eegnet
    --list: List all available algorithms
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Fix OMP lib conflict
import sys
import json
import time
import glob
import platform
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import mne
mne.set_log_level('ERROR')

# Deep learning imports (move to top for TinySleepNet definition)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from braindecode.models import EEGNetv4
    _DEEP_LEARNING_AVAILABLE = True
    print(f"✅ PyTorch loaded, CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    _DEEP_LEARNING_AVAILABLE = False
    print(f"⚠️  Deep learning modules not fully available: {e}")

sys_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, sys_path)

IS_WINDOWS = platform.system() == 'Windows'

SLEEP_EDF_DIR = (r"E:\datasets\Sleep\sleep-edf-database-expanded-1.0.0" if IS_WINDOWS
    else os.environ.get("SLEEP_EDF_DIR",
        r"/mnt/data1/home/tanhuang/datasets/sleep-edf-database-expanded-1.0.0"))
OUTPUT_DIR = (r"experiment_results" if IS_WINDOWS
    else os.environ.get("OUTPUT_DIR",
        r"./experiment_results"))

from sca_fbts_woman import MenoSCA_FBTS

FS = 100
EPOCH_LENGTH_S = 30
SAMPLES_PER_EPOCH = FS * EPOCH_LENGTH_S

SLEEP_STAGE_LABELS = {
    'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2,
    'Sleep stage 3': 3, 'Sleep stage 4': 3, 'Sleep stage R': 4,
}
STAGE_NAMES = ['Wake', 'N1', 'N2', 'N3', 'REM']

def load_menopausal_subjects():
    """Dynamically load menopausal subjects from Sleep-EDF SC-subjects.csv (age 45-59, female)"""
    csv_path = os.path.normpath(os.path.join(SLEEP_EDF_DIR, '..', 'SC-subjects.csv'))
    for p in [csv_path, os.path.join(SLEEP_EDF_DIR, 'SC-subjects.csv')]:
        if os.path.exists(p):
            csv_path = p
            break
    if os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        menopause_ids = []
        for _, row in df.iterrows():
            age = int(row['age'])
            gender = 'F' if row['sex (F=1)'] == 1 else 'M'
            if gender == 'F' and 45 <= age <= 59:
                menopause_ids.append(str(row['subject']))
        seen = set()
        menopause_ids = [x for x in menopause_ids if not (x in seen or seen.add(x))]
        print(f"  [Auto-detect] Found {len(menopause_ids)} female subjects aged 45-59")
        return menopause_ids
    return ['20', '21', '23', '24', '26', '27', '29', '80']

MENOPAUSAL_SUBJECTS = load_menopausal_subjects()

# Algorithm configuration mapping
ALGORITHMS = {
    'rf': 'Random Forest',
    'eegnet': 'EEGNet (braindecode)',
    'tinysleepnet': 'TinySleepNet',
    'minirocket': 'MiniRocket',
    'yasa': 'YASA',
    'menoscafbts': 'MenoSCA-FBTS',
}

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

def compute_band_powers(data, fs=100):
    """Compute band power"""
    freq_bands = [
        (0.5, 4, 'Delta'), (4, 6, 'Theta'), (6, 8, 'Theta2'),
        (8, 12, 'Alpha'), (12, 15, 'Sigma'), (15, 30, 'Beta'), (30, 40, 'Gamma')
    ]
    powers = {}
    for low, high, name in freq_bands:
        bp_filtered = signal.butter(4, [low / (fs/2), high / (fs/2)], btype='band')
        filtered = signal.filtfilt(bp_filtered[0], bp_filtered[1], data, axis=-1)
        powers[name] = np.mean(filtered ** 2, axis=-1)
    return powers

def extract_features(epochs_data):
    """Extract features for traditional ML methods"""
    all_features = []
    for epoch in epochs_data:
        features = []
        for ch_data in epoch:
            powers = compute_band_powers(ch_data)
            features.extend(powers.values())
            total_power = sum(powers.values()) + 1e-10
            for name in powers:
                features.append(powers[name] / total_power)
            psd = np.abs(np.fft.rfft(ch_data))**2
            features.extend(psd[:50])
        all_features.append(features)
    return np.array(all_features)

class EEGNetClassifier:
    """EEGNet using braindecode"""
    def __init__(self, n_classes=5, fs=100, n_channels=2):
        self.n_classes = n_classes
        self.fs = fs
        self.n_channels = n_channels
        self.model = None
        self.device = None
        self.preprocessing = StandardScaler()

    def fit(self, X, y):
        try:
            import torch
            from braindecode.models import EEGNetv4
        except ImportError:
            raise ImportError("Need to install torch and braindecode: pip install torch braindecode")

        if len(X) < 100:
            raise ValueError(f"Insufficient data ({len(X)}), need at least 100 samples to train deep learning model")

        X = self._prepare_data(X)
        # Standardize by channel (standard practice for EEG data)
        X = X.copy()
        for i in range(X.shape[1]):
            X[:, i, :] = self.preprocessing.fit_transform(X[:, i, :])

        n_timesteps = X.shape[2]
        
        # Split training and validation sets (80-20 split)
        split_idx = int(0.8 * len(X))
        indices = np.random.permutation(len(X))
        train_idx, val_idx = indices[:split_idx], indices[split_idx:]
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = EEGNetv4(
            n_chans=self.n_channels,
            n_outputs=self.n_classes,
            n_times=n_timesteps,
            input_window_seconds=n_timesteps / self.fs,
            sfreq=self.fs,
            final_conv_length='auto'
        ).to(self.device)

        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        batch_size = 16
        n_epochs = 50
        
        # Early stopping mechanism (based on validation loss)
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(n_epochs):
            # Training Phase
            self.model.train()
            idx_batch = np.random.permutation(len(X_train_tensor))
            train_loss = 0
            for i in range(0, len(X_train_tensor), batch_size):
                batch_idx = idx_batch[i:i+batch_size]
                X_batch = X_train_tensor[batch_idx]
                y_batch = y_train_tensor[batch_idx]

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation Phase
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
            
            # Early Stopping Check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"    EEGNet early stopping at epoch {epoch+1}")
                    break

        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained")

        try:
            import torch
        except ImportError:
            raise ImportError("Need to install torch and braindecode: pip install torch braindecode")

        X = self._prepare_data(X)
        # Standardize by channel (consistent with fit)
        X = X.copy()  # Avoid modifying original data
        for i in range(X.shape[1]):
            X[:, i, :] = self.preprocessing.transform(X[:, i, :])

        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predictions = torch.max(outputs, dim=1)

        return predictions.cpu().numpy()

    def _prepare_data(self, X):
        if X.ndim == 3 and X.shape[1] == self.n_channels:
            return X
        if X.ndim == 3:
            return X[:, :self.n_channels, :]
        raise ValueError(f"Input data dimension error: {X.shape}, expected (n_samples, {self.n_channels}, n_times)")


class TinySleepNet(torch.nn.Module):
    """TinySleepNet: Lightweight deep learning model for 30s sleep epoch classification
    Reference: https://github.com/akaraspt/tinysleepnet
    """
    def __init__(self, n_classes=5, n_channels=2, n_times=3000):
        super(TinySleepNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Feature extraction blocks
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(n_channels, 64, kernel_size=50, stride=6, padding=22),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=8, stride=8),
            torch.nn.Dropout(0.5)
        )
        
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=3),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=8, stride=8)
        )
        
        # Dynamically calculate flattened feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, n_channels, n_times)
            out = self.conv1(dummy)
            out = self.conv2(out)
            self.flatten_dim = out.view(1, -1).shape[1]
        
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.flatten_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, n_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TinySleepNetClassifier:
    """TinySleepNet Wrapper - Optimized for 30-second independent epochs"""
    def __init__(self, n_classes=5, fs=100, n_channels=2):
        self.n_classes = n_classes
        self.fs = fs
        self.n_channels = n_channels
        self.model = None
        self.device = None
        self.preprocessing = StandardScaler()

    def fit(self, X, y):
        try:
            import torch
        except ImportError:
            raise ImportError("Need to install torch: pip install torch")

        if len(X) < 100:
            raise ValueError(f"Insufficient data ({len(X)}), need at least 100 samples to train deep learning model")

        X = self._prepare_data(X)
        # Standardize by channel
        X = X.copy()
        for i in range(X.shape[1]):
            X[:, i, :] = self.preprocessing.fit_transform(X[:, i, :])

        n_timesteps = X.shape[2]
        
        # Split training and validation sets (80-20 split)
        split_idx = int(0.8 * len(X))
        indices = np.random.permutation(len(X))
        train_idx, val_idx = indices[:split_idx], indices[split_idx:]
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = TinySleepNet(
            n_channels=self.n_channels,
            n_classes=self.n_classes,
            n_times=n_timesteps
        ).to(self.device)

        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        batch_size = 16
        n_epochs = 50
        
        # Early stopping mechanism
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(n_epochs):
            self.model.train()
            idx_batch = np.random.permutation(len(X_train_tensor))
            train_loss = 0
            for i in range(0, len(X_train_tensor), batch_size):
                batch_idx = idx_batch[i:i+batch_size]
                X_batch = X_train_tensor[batch_idx]
                y_batch = y_train_tensor[batch_idx]
                
                if X_batch.shape[0] <= 1:
                    continue

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"    TinySleepNet early stopping at epoch {epoch+1}")
                    break

        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained")

        try:
            import torch
        except ImportError:
            raise ImportError("Need to install torch: pip install torch")

        X = self._prepare_data(X)
        X = X.copy()
        for i in range(X.shape[1]):
            X[:, i, :] = self.preprocessing.transform(X[:, i, :])

        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predictions = torch.max(outputs, dim=1)

        return predictions.cpu().numpy()

    def _prepare_data(self, X):
        if X.ndim == 3 and X.shape[1] == self.n_channels:
            return X
        if X.ndim == 3:
            return X[:, :self.n_channels, :]
        raise ValueError(f"Input data dimension error: {X.shape}, expected (n_samples, {self.n_channels}, n_times)")


class MiniRocketClassifier:
    """MiniRocket implementation"""
    def __init__(self, n_classes=5, n_kernels=1000, fs=100, n_channels=2):
        self.n_classes = n_classes
        self.n_kernels = n_kernels
        self.classifier = None
        self.filters = []

    def fit(self, X, y):
        n_timesteps = X.shape[2]
        self._generate_filters(n_timesteps)
        features = self._transform(X)
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.classifier.fit(features, y)
        return self

    def predict(self, X):
        features = self._transform(X)
        return self.classifier.predict(features)

    def _generate_filters(self, n_timesteps):
        np.random.seed(42)
        for _ in range(self.n_kernels):
            d = np.random.randint(1, 10)
            l = np.random.randint(1, min(10, n_timesteps // 4))
            p = np.random.randint(0, max(1, n_timesteps - l * d))
            s = 1 if np.random.rand() > 0.5 else 2
            self.filters.append((d, l, p, s))

    def _transform(self, X):
        features = []
        for epoch in X:
            epoch_feats = []
            for ch_data in epoch:
                for d, l, p, s in self.filters:
                    samples = range(p, min(p + l * d, len(ch_data)), d)
                    if s == 1:
                        feat = np.max(ch_data[list(samples)])
                    else:
                        feat = np.mean(ch_data[list(samples)])
                    epoch_feats.append(feat)
            features.append(epoch_feats)
        return np.array(features)


class YASASleepStager:
    """YASA Sleep Staging Wrapper - Uses YASA Features + Random Forest"""
    def __init__(self, n_classes=5, fs=100, n_channels=2):
        self.n_classes = n_classes
        self.fs = fs
        self.n_channels = n_channels
        self.scaler = StandardScaler()
        self.classifier = None

    def _extract_yasa_features(self, X):
        """Extract YASA-style Features"""
        features = []
        for epoch in X:
            epoch_feats = []
            for ch_data in epoch:
                # Compute Statistical Features
                epoch_feats.append(np.std(ch_data))
                epoch_feats.append(np.percentile(ch_data, 75) - np.percentile(ch_data, 25))
                
                # Compute Zero-crossings
                zero_crossings = np.sum(np.diff(np.sign(ch_data)) != 0)
                epoch_feats.append(zero_crossings)
                
                # Compute Hjorth Parameters
                diff1 = np.diff(ch_data)
                diff2 = np.diff(diff1)
                var_sig = np.var(ch_data)
                var_d1 = np.var(diff1)
                var_d2 = np.var(diff2)
                
                mobility = np.sqrt(var_d1 / var_sig) if var_sig > 0 else 0
                complexity = np.sqrt(var_d2 / var_d1) / mobility if var_d1 > 0 else 0
                epoch_feats.append(mobility)
                epoch_feats.append(complexity)
                
                # Compute Frequency Band Power
                freqs, psd = signal.welch(ch_data, fs=self.fs, nperseg=256)
                
                # Frequency Band Definitions
                bands = {
                    'delta': (0.5, 4),
                    'theta': (4, 8),
                    'alpha': (8, 12),
                    'sigma': (12, 16),
                    'beta': (16, 30)
                }
                
                band_powers = {}
                total_power = 0
                for band_name, (low, high) in bands.items():
                    mask = (freqs >= low) & (freqs < high)
                    power = np.sum(psd[mask])
                    band_powers[band_name] = power
                    total_power += power
                
                # Add Relative Power
                for band_name in bands:
                    rel_power = band_powers[band_name] / (total_power + 1e-10)
                    epoch_feats.append(rel_power)
                
                # Add Power Ratio
                epoch_feats.append(band_powers['delta'] / (band_powers['beta'] + 1e-10))
                epoch_feats.append(band_powers['theta'] / (band_powers['beta'] + 1e-10))
                
            features.append(epoch_feats)
        return np.array(features)

    def fit(self, X, y):
        """Train Random Forest with YASA Features"""
        features = self._extract_yasa_features(X)
        features = self.scaler.fit_transform(features)
        
        self.classifier = RandomForestClassifier(
            n_estimators=200, 
            max_depth=15, 
            random_state=42, 
            n_jobs=-1
        )
        self.classifier.fit(features, y)
        return self

    def predict(self, X):
        """Predict using the trained model"""
        if self.classifier is None:
            raise ValueError("Model not trained")
        
        features = self._extract_yasa_features(X)
        features = self.scaler.transform(features)
        return self.classifier.predict(features)


def run_experiment(selected_algorithms=None):
    """Run SOTA Comparison Experiment
    
    Args:
        selected_algorithms: Optional list of algorithm keys, e.g. ['rf', 'eegnet']; None means run all algorithms
    """
    # If not specified, run all algorithms
    if selected_algorithms is None:
        selected_algorithms = list(ALGORITHMS.keys())
    
    enabled_names = [ALGORITHMS[alg] for alg in selected_algorithms]
    
    print("=" * 70)
    print("SOTA Sleep Staging Methods Comparison Experiment")
    print("=" * 70)
    print(f"Dataset: Sleep-EDF Menopausal Women")
    print(f"Subjects: {MENOPAUSAL_SUBJECTS}")
    print(f"Methods: {', '.join(enabled_names)}")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_epochs, all_labels, all_subjects = [], [], []
    print("\n[1/5] Loading Data...")
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

    print(f"\n  Total: {len(X)} epochs, {X.shape[1]} channels, {X.shape[2]} timesteps")
    print(f"  Label Distribution: ", end="")
    for i, name in enumerate(STAGE_NAMES):
        count = np.sum(y == i)
        pct = count / len(y) * 100
        print(f"{name}={count}({pct:.1f}%) ", end="")
    print()

    results = {}
    logo = LeaveOneGroupOut()
    n_subjects = len(np.unique(groups))

    print("\n[2/5] Extract Traditional ML Features...")
    X_features = extract_features(X)
    print(f"  Feature Dimensions: {X_features.shape}")

    if 'rf' in selected_algorithms:
        print("\n[3/5] Train Traditional ML Method (Random Forest)...")
        fold_results = {'accuracy': [], 'f1_macro': [], 'conf_matrix': []}
        total_start = time.time()

        for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
            X_train_feat = X_features[train_idx]
            X_test_feat = X_features[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
            clf.fit(X_train_feat, y_train)
            y_pred = clf.predict(X_test_feat)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3, 4])

            fold_results['accuracy'].append(acc)
            fold_results['f1_macro'].append(f1)
            fold_results['conf_matrix'].append(cm)

            subj_in_test = np.unique(groups[test_idx])
            print(f"    Subject {subj_in_test[0]} (Fold {fold_idx+1}/{n_subjects}): Acc={acc:.4f}, F1={f1:.4f}")

        total_time = time.time() - total_start
        results['Random Forest'] = {
            'accuracy': np.mean(fold_results['accuracy']),
            'accuracy_std': np.std(fold_results['accuracy']),
            'f1_macro': np.mean(fold_results['f1_macro']),
            'f1_macro_std': np.std(fold_results['f1_macro']),
            'conf_matrix': np.mean(fold_results['conf_matrix'], axis=0),
            'training_time': total_time
        }
        print(f"    Mean: Acc={results['Random Forest']['accuracy']:.4f}±{results['Random Forest']['accuracy_std']:.4f}")
        print(f"    Total Training Time: {total_time:.1f}s")

    # Define algorithm mapping
    deep_learning_methods = [
        ('eegnet', 'EEGNet (braindecode)', EEGNetClassifier),
        ('tinysleepnet', 'TinySleepNet', TinySleepNetClassifier),
        ('minirocket', 'MiniRocket', MiniRocketClassifier),
        ('yasa', 'YASA', YASASleepStager)
    ]
    
    # Filter selected deep learning algorithms
    selected_deep_methods = [(key, name, clf) for key, name, clf in deep_learning_methods 
                            if key in selected_algorithms]
    
    if selected_deep_methods:
        print(f"\n[4/5] Train Deep Learning Methods ({len(selected_deep_methods)} methods)...")

        for method_key, method_name, clf_class in selected_deep_methods:
            print(f"\n  Training {method_name}...")
            fold_results = {'accuracy': [], 'f1_macro': [], 'conf_matrix': []}
            total_start = time.time()

            try:
                for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    clf = clf_class(n_classes=5, fs=FS, n_channels=X.shape[1])
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)

                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='macro')
                    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3, 4])

                    fold_results['accuracy'].append(acc)
                    fold_results['f1_macro'].append(f1)
                    fold_results['conf_matrix'].append(cm)

                    subj_in_test = np.unique(groups[test_idx])
                    print(f"    Subject {subj_in_test[0]} (Fold {fold_idx+1}/{n_subjects}): Acc={acc:.4f}, F1={f1:.4f}")

                total_time = time.time() - total_start
                results[method_name] = {
                    'accuracy': np.mean(fold_results['accuracy']),
                    'accuracy_std': np.std(fold_results['accuracy']),
                    'f1_macro': np.mean(fold_results['f1_macro']),
                    'f1_macro_std': np.std(fold_results['f1_macro']),
                    'conf_matrix': np.mean(fold_results['conf_matrix'], axis=0),
                    'training_time': total_time
                }
                print(f"    Mean: Acc={results[method_name]['accuracy']:.4f}±{results[method_name]['accuracy_std']:.4f}")
                print(f"    Total Training Time: {total_time:.1f}s")

            except Exception as e:
                print(f"    {method_name} Training Failed: {e}")
                continue

    if 'menoscafbts' in selected_algorithms:
        print("\n[5/5] Train MenoSCA-FBTS (Reference Method)...")
        fold_results = {'accuracy': [], 'f1_macro': [], 'conf_matrix': []}
        total_start = time.time()

        for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf = MenoSCA_FBTS(n_bands=7, estimator='oas', metric='riemann',
                              classifier='ensemble', n_features=200, fs=FS,
                              temporal_smoothing=True, enable_menopause_features=True)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3, 4])

            fold_results['accuracy'].append(acc)
            fold_results['f1_macro'].append(f1)
            fold_results['conf_matrix'].append(cm)

            subj_in_test = np.unique(groups[test_idx])
            print(f"    Subject {subj_in_test[0]} (Fold {fold_idx+1}/{n_subjects}): Acc={acc:.4f}, F1={f1:.4f}")

        total_time = time.time() - total_start
        results['MenoSCA-FBTS'] = {
            'accuracy': np.mean(fold_results['accuracy']),
            'accuracy_std': np.std(fold_results['accuracy']),
            'f1_macro': np.mean(fold_results['f1_macro']),
            'f1_macro_std': np.std(fold_results['f1_macro']),
            'conf_matrix': np.mean(fold_results['conf_matrix'], axis=0),
            'training_time': total_time
        }
        print(f"    Mean: Acc={results['MenoSCA-FBTS']['accuracy']:.4f}±{results['MenoSCA-FBTS']['accuracy_std']:.4f}")
        print(f"    Total Training Time: {total_time:.1f}s")

    print("\n" + "=" * 70)
    print("Experiment Results Summary")
    print("=" * 70)
    print(f"{'Method':<25} {'Accuracy':<15} {'F1-Macro':<15} {'Training Time':<10}")
    print("-" * 70)
    for name, res in results.items():
        print(f"{name:<25} {res['accuracy']:.4f}±{res['accuracy_std']:.4f}   {res['f1_macro']:.4f}±{res['f1_macro_std']:.4f}   {res['training_time']:.1f}s")
    print("=" * 70)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(OUTPUT_DIR, f"sota_comparison_{timestamp}.json")
    
    # Convert numpy arrays to lists for JSON serialization
    results_serializable = {}
    for name, res in results.items():
        results_serializable[name] = {
            'accuracy': float(res['accuracy']),
            'accuracy_std': float(res['accuracy_std']),
            'f1_macro': float(res['f1_macro']),
            'f1_macro_std': float(res['f1_macro_std']),
            'conf_matrix': res['conf_matrix'].tolist() if hasattr(res['conf_matrix'], 'tolist') else res['conf_matrix'],
            'training_time': float(res['training_time'])
        }
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    print(f"\nResults Saved: {result_file}")

    plt.figure(figsize=(12, 6))
    methods_names = list(results.keys())
    accs = [results[m]['accuracy'] for m in methods_names]
    stds = [results[m]['accuracy_std'] for m in methods_names]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#9B59B6', '#2ECC71']
    bars = plt.bar(methods_names, accs, yerr=stds, capsize=5, color=colors[:len(methods_names)])
    plt.ylabel('Accuracy')
    plt.title('SOTA Sleep Staging Methods Comparison (Menopausal Women)')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{acc:.3f}', ha='center', va='bottom')
    plt.tight_layout()
    fig_file = os.path.join(OUTPUT_DIR, f"sota_comparison_{timestamp}.png")
    plt.savefig(fig_file, dpi=150)
    print(f"Figure Saved: {fig_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description='SOTA Sleep Staging Methods Comparison Experiment')
    parser.add_argument('--algorithms', '-a', type=str, default=None,
                        help='Specify algorithms to test, separated by commas. Options: rf, eegnet, minirocket, yasa, menoscafbts')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List All Available Algorithms')
    
    args = parser.parse_args()
    
    # List All Available Algorithms
    if args.list:
        print("Available Algorithms:")
        print("-" * 40)
        for key, name in ALGORITHMS.items():
            print(f"  {key:15} -> {name}")
        print("\nUsage Example:")
        print("  python experiment_sota_comparison.py --algorithms rf,eegnet")
        print("  python experiment_sota_comparison.py -a minirocket,yasa")
        return
    
    # Parse Specified Algorithms
    selected_algorithms = None
    if args.algorithms:
        selected_algorithms = [alg.strip() for alg in args.algorithms.split(',')]
        
        # Validate Algorithm Names
        valid_algs = list(ALGORITHMS.keys())
        for alg in selected_algorithms:
            if alg not in valid_algs:
                print(f"Error: Unknown Algorithm '{alg}'")
                print(f"Available Algorithms: {', '.join(valid_algs)}")
                return
    
    # Run Experiment
    results = run_experiment(selected_algorithms)


if __name__ == '__main__':
    main()