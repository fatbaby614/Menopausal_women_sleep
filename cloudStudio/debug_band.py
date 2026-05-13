import numpy as np
from sca_fbts_woman import MenoSCA_FBTS

print("=== Testing 10-Band Configuration ===")

# Create test data (100Hz, 30 seconds = 3000 samples)
np.random.seed(42)
test_eeg = np.random.randn(2, 3000).astype(np.float64)

# Test 1: Baseline (7 bands, no explicit freq_bands)
baseline = MenoSCA_FBTS(n_bands=7, enable_menopause_features=False)
spec_baseline = baseline.extract_spectral_features(test_eeg)
print(f"Baseline (n_bands=7): {len(spec_baseline)} dims, first 5: {spec_baseline[:5]}")

# Test 2: 10-Band with explicit freq_bands
band10 = MenoSCA_FBTS(n_bands=10, enable_menopause_features=False,
    freq_bands=[(0.5,2),(2,4),(4,6),(6,8),(8,10),(10,12),(12,15),(15,20),(20,30),(30,40)])
spec_10 = band10.extract_spectral_features(test_eeg)
print(f"10-Band (explicit): {len(spec_10)} dims, first 5: {spec_10[:5]}")

# Test 3: Check internal freq_bands
print(f"\nBaseline._get_freq_bands(): {baseline._get_freq_bands()}")
print(f"10-Band._get_freq_bands(): {band10._get_freq_bands()}")

# Test 4: Compare after 5-band
band5 = MenoSCA_FBTS(n_bands=5, enable_menopause_features=False)
spec_5 = band5.extract_spectral_features(test_eeg)
print(f"\n5-Band: {len(spec_5)} dims, first 5: {spec_5[:5]}")
print(f"5-Band._get_freq_bands(): {band5._get_freq_bands()}")

print(f"\n=== Summary ===")
print(f"Baseline spectral dim: {len(spec_baseline)} (expected 14 for 7 bands × 2 channels)")
print(f"5-Band spectral dim: {len(spec_5)} (expected 10 for 5 bands × 2 channels)")
print(f"10-Band spectral dim: {len(spec_10)} (expected 20 for 10 bands × 2 channels)")
print(f"10-Band matches Baseline: {len(spec_10) == len(spec_baseline)} (should be False)")