import pandas as pd
import numpy as np
from utils.timefeatures import time_features
from utils.custom_timefeatures import custom_time_features

# Create sample datetime data
dates = pd.date_range(start='2021-01-01', end='2022-01-01', freq='h')

# Compare outputs
print("Original time features ([-0.5, 0.5] encoding):")
standard_features = time_features(dates, freq='h')
print(f"Shape: {standard_features.shape}")
print(standard_features[:, :6])  # First timepoint features

print("\nCustom time features (sine-cosine encoding):")
custom_features = custom_time_features(dates, freq='h')
print(f"Shape: {custom_features.shape}")
print(custom_features[:, :6])  # First timepoint features

# Verify cyclic property
hour = dates[0].hour
expected_sine = np.sin(2 * np.pi * hour / 24)
expected_cosine = np.cos(2 * np.pi * hour / 24)
print(f"\nVerification for hour {hour}:")
print(f"Expected sine: {expected_sine:.6f}")
print(f"Expected cosine: {expected_cosine:.6f}")