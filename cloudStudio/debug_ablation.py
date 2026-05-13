import numpy as np
import sys
sys.path.insert(0, '.')

# Test 1: Check 10-Band vs Baseline spectral features dimension
from sca_fbts_woman import MenoSCA_FBTS

# Baseline: 7 bands (default), no explicit freq_bands
baseline = MenoSCA_FBTS(n_bands=7, enable_menopause_features=False)

# 10-Bands: explicit 10-band freq_bands
band10 = MenoSCA_FBTS(n_bands=10, enable_menopause_features=False,
                       freq_bands=[(0.5,2),(2,4),(4,6),(6,8),(8,10),(10,12),(12,15),(15,20),(20,30),(30,40)])

# Create proper EEG test data (2 channels, 3000 samples = 30s at 100Hz)
np.random.seed(42)
test_eeg = np.random.randn(2, 3000).astype(np.float64)

# Extract spectral features
spec_baseline = baseline.extract_spectral_features(test_eeg)
spec_10 = band10.extract_spectral_features(test_eeg)

print('=== Issue 1: 10-Band vs Baseline spectral dimension ===')
print(f'Baseline spectral features: {len(spec_baseline)} dims')
print(f'10-Band spectral features: {len(spec_10)} dims')
print(f'Dimensions DIFFER: {len(spec_baseline) != len(spec_10)}')
print(f'Expected: baseline=14 (7 bands × 2), 10-band=20 (10 bands × 2)')

# Test 2: Check if menopause features add extra dims
baseline_mf = MenoSCA_FBTS(n_bands=7, enable_menopause_features=True)
spec_mf = baseline_mf.extract_spectral_features(test_eeg)
print(f'\nBaseline + menopause features: {len(spec_mf)} dims')
print(f'Expected extra: 3 dims (alpha/delta ratio, sigma/theta ratio, std)')

# Test 3: Full feature extraction
print('\n=== Issue 2: Full feature extraction ===')
# Need to match the actual input format used in ablation
test_input = test_eeg.reshape(1, 2, 3000)
feat_base = baseline.extract_features(test_input)
feat_10 = band10.extract_features(test_input)
print(f'Baseline full features: {len(feat_base[0])} dims')
print(f'10-Band full features: {len(feat_10[0])} dims')
print(f'Feature dimensions DIFFER: {len(feat_base[0]) != len(feat_10[0])}')

print('\n=== Summary ===')
if len(spec_baseline) != len(spec_10):
    print('✅ Bands ARE producing different spectral dimensions')
else:
    print('❌ BUG FOUND: Bands NOT producing different spectral dimensions!')