import os
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

    return features

def process_eis_features(base_path, soc_value=50, target_df=None, save_path="eis_feature_data.csv"):
    feature_rows = []

    for batch in range(1, 12):
        folder_name = f'B{batch:02d}'
        
        for test_num in range(1, 3):
            folder_path = os.path.join(base_path, folder_name, 'EIS measurements', f'Test_{test_num}', 'Hioki')
            if not os.path.isdir(folder_path):
                continue

            soc_keyword = f"SoC_{soc_value}"
            all_files = os.listdir(folder_path)
            matched_files = [
                os.path.join(folder_path, fname)
                for fname in all_files
                if soc_keyword in fname and fname.endswith('.csv')
            ]

            for file_path in matched_files:
                try:
                    df = pd.read_csv(file_path)

                    # Rename columns for compatibility
                    df.rename(columns={
                        df.columns[0]: 'frequency',
                        df.columns[1]: 'real impedance',
                        df.columns[2]: 'imaginary impedance'
                    }, inplace=True)


                    # Extract features
                    feature_dict = extract_features(df)
                    # Add targets if available
                    feature_rows.append(feature_dict)

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    final_df = pd.DataFrame(feature_rows)
    return final_df