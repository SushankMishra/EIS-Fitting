import pandas as pd
import numpy as np
import os

def load_eis_data(base_path, soc_value=50):
    input_rows = []
    labels = []

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
                    df.rename(columns={
                        df.columns[0]: 'frequency',
                        df.columns[1]: 'Z_re',
                        df.columns[2]: 'Z_img'
                    }, inplace=True)

                    # Sort by frequency to ensure consistency
                    df.sort_values(by='frequency', inplace=True)
                    df.reset_index(drop=True, inplace=True)

                    # Create flat input vector
                    input_vector = []
                    for _, row in df.iterrows():
                        input_vector.extend([row['Z_re'], row['Z_img']])                
                    input_rows.append(input_vector)

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    X = pd.DataFrame(input_rows)
    return X