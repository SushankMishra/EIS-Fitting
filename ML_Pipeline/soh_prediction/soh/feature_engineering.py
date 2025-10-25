import os
import re
import pandas as pd
import numpy as np
from impedance.models.circuits import CustomCircuit
from . import config

def fit_ecm(frequency: np.ndarray, impedance: np.ndarray) -> dict:
    """
    Fits an Equivalent Circuit Model (ECM) to the given impedance data.

    Args:
        frequency: Array of frequency values.
        impedance: Array of complex impedance values.

    Returns:
        A dictionary of the fitted ECM parameters, or NaNs on failure.
    """
    circuit = CustomCircuit(
        initial_guess=config.INITIAL_GUESS,
        circuit=config.CIRCUIT_STRING
    )
    
    try:
        # Filter out noisy, low-frequency data for a more stable fit
        mask = frequency > 0.1
        circuit.fit(frequency[mask], impedance[mask])
        param_names = circuit.get_param_names()[0]
        return dict(zip(param_names, circuit.parameters_))
    except (RuntimeError, ValueError) as e:
        print(f"Warning: ECM fitting failed. Error: {e}")
        param_names = ['R0', 'R1', 'CPE1_T', 'CPE1_p', 'R2', 'CPE2_T', 'CPE2_p']
        return {name: np.nan for name in param_names}

def create_feature_table(eis_path: str, soh_df: pd.DataFrame):
    """
    Creates a feature table by extracting ECM parameters from the nested EIS directory structure.

    Args:
        eis_path: Path to the 'EIS_Charge_discharge' directory.
        soh_df: DataFrame containing SoH values for each cycle, indexed by Cycle.
    """
    all_features = []
    
    # Find all 'EIS_X' subdirectories
    eis_folders = [d for d in os.listdir(eis_path) if d.startswith('EIS_') and os.path.isdir(os.path.join(eis_path, d))]

    for folder_name in eis_folders:
        cycle_num = int(folder_name.split('_')[1])
        folder_path = os.path.join(eis_path, folder_name)
        
        print(f"Processing folder: {folder_name} (Cycle {cycle_num})")
        
        try:
            soh_value = soh_df.loc[cycle_num, 'SoH']
        except KeyError:
            print(f"  - Warning: SoH not found for Cycle {cycle_num}. Skipping.")
            continue

        # Load the SoC mapping for this cycle
        soc_map_path = os.path.join(folder_path, 'SOC.csv')
        if not os.path.exists(soc_map_path):
            print(f"  - Warning: SOC.csv not found in {folder_name}. Skipping.")
            continue
        soc_map_df = pd.read_csv(soc_map_path, header=None, names=['File_Index', 'SoC'])
        soc_map_df['File_Name'] = soc_map_df['File_Index'].astype(str) + '_EIS.csv'
        soc_lookup = soc_map_df.set_index('File_Name')['SoC'].to_dict()

        # Process each EIS file in the subdirectory
        for eis_file_name in os.listdir(folder_path):
            if not eis_file_name.endswith('_EIS.csv'):
                continue

            soc_value = soc_lookup.get(eis_file_name)
            if soc_value is None:
                print(f"  - Warning: SoC not found for file {eis_file_name}. Skipping.")
                continue

            eis_file_path = os.path.join(folder_path, eis_file_name)
            df = pd.read_csv(eis_file_path, header=None, names=['Freq', 'Re', 'Im'], delimiter=',')
            
            z_complex = df['Re'].values + 1j * df['Im'].values
            
            fitted_params = fit_ecm(df['Freq'].values, z_complex)
            
            # Combine all information
            record = {
                'Cycle': cycle_num,
                'SoC': soc_value,
                'SoH': soh_value,
                **fitted_params
            }
            all_features.append(record)
            
    # Create final DataFrame
    feature_df = pd.DataFrame(all_features)
    feature_df.dropna(inplace=True) # Drop rows where ECM fitting failed
    
    feature_df.to_csv(config.FEATURE_PATH, index=False)
    print("\nFeature Table Creation Complete.")
    print(f"Total features created: {len(feature_df)}")
    print(feature_df.head())

