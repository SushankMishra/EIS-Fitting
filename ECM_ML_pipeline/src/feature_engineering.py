import numpy as np
import pandas as pd
from scipy.integrate import simpson
def extract_features(df, freqs_to_sample=[1, 10, 100, 1000]):
    df = df.sort_values('frequency')  # Ensure increasing frequency
    df['Z_mag'] = np.sqrt(df['real impedance']**2 + df['imaginary impedance']**2)
    df['Z_phase'] = np.arctan2(df['imaginary impedance'], df['real impedance'])

    features = {}

    # Sampled point features
    for f in freqs_to_sample:
        row = df.iloc[(df['frequency'] - f).abs().argsort()[:1]]
        features[f'Z_mag_{f}Hz'] = row['Z_mag'].values[0]
        features[f'Z_phase_{f}Hz'] = row['Z_phase'].values[0]
        features[f'Re_{f}Hz'] = row['real impedance'].values[0]
        features[f'Im_{f}Hz'] = row['imaginary impedance'].values[0]

    # Geometric/statistical features
    features['Re_max'] = df['real impedance'].max()
    features['Im_min'] = df['imaginary impedance'].min()  # Bottom of the semicircle
    features['Z_mag_mean'] = df['Z_mag'].mean()
    features['Z_phase_mean'] = df['Z_phase'].mean()
    features['Z_phase_std'] = df['Z_phase'].std()
    features['Nyquist_area'] = simpson(y=-df['imaginary impedance'], x=df['real impedance'])  # Area under curve
    features['Z_mag_slope'] = np.polyfit(np.log10(df['frequency']), df['Z_mag'], 1)[0]  # Slope in log-log space

    return pd.DataFrame([features])