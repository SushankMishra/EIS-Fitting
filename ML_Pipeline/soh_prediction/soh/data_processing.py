import os
import re
import pandas as pd
import numpy as np
from typing import List, Tuple

def calculate_soh(discharge_path: str) -> pd.DataFrame:
    """
    Calculates the State of Health (SoH) for each cycle based on standard discharge capacity.

    Args:
        discharge_path: Path to the 'Discharge_curve' directory.

    Returns:
        A DataFrame with 'Cycle' and 'SoH' columns, indexed by Cycle.
    """
    print(f"Searching for standard discharge files in: {discharge_path}")
    discharge_files = [
        f for f in os.listdir(discharge_path)
        if re.match(r'\d+_Discharge_std\.csv', f)
    ]
    discharge_files.sort(key=lambda x: int(x.split('_')[0]))

    if not discharge_files:
        raise FileNotFoundError(f"No standard discharge files found in {discharge_path}")

    results: List[Tuple[int, float]] = []
    for file in discharge_files:
        try:
            cycle = int(file.split('_')[0])
            file_path = os.path.join(discharge_path, file)
            df = pd.read_csv(file_path, header=None, names=['Time', 'Voltage', 'Current'], delimiter=',')
            
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            df = df.sort_values(by='Time')

            time_s = df['Time'].values
            current_a = df['Current'].values

            capacity_as = np.trapz(-current_a, time_s)
            capacity_ah = capacity_as / 3600
            results.append((cycle, capacity_ah))
        except Exception as e:
            print(f"Warning: Could not process file {file}. Error: {e}")
            continue

    soh_df = pd.DataFrame(results, columns=['Cycle', 'Discharge_Capacity_Ah'])

    if soh_df.empty:
        raise ValueError("Failed to calculate capacity from any file.")
        
    initial_capacity = soh_df.iloc[0]['Discharge_Capacity_Ah']
    soh_df['SoH'] = soh_df['Discharge_Capacity_Ah'] / initial_capacity
    
    print("SoH Calculation Complete.")
    
    return soh_df[['Cycle', 'SoH']].set_index('Cycle')

