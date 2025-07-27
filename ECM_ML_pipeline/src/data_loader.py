import pandas as pd
import os
def load_eis_data(base_path, soc_value = 5, batch = 1):
    # for i in range(1, 12):  # 1 to 11 inclusive
    folder_name = f'B{batch:02d}'  # B01, B02, ..., B11
    matched_files = []
    for i in range(1,3):
        folder_path = os.path.join(base_path, folder_name, 'EIS measurements',f'Test_{i}','Hioki')
        if not os.path.isdir(folder_path):
            print(f"Warning: Folder {folder_path} does not exist. Skipping.")

        # Loop over all CSV files inside the folder
        soc_keyword = f"SoC_{soc_value}"
        all_files = os.listdir(folder_path)

        matched_files = [
            os.path.join(folder_path, fname)
            for fname in all_files
            if soc_keyword in fname and fname.endswith('.csv')
        ]

        if not matched_files:
            raise FileNotFoundError(f"No CSV files found for SoC = {soc_value} in {folder_path}")

    df_list = []
    for file in matched_files:
        try:
            df = pd.read_csv(file)
            df_list.append(df)
        except Exception as e:
            print(f"Warning: Could not load {file}. Error: {e}")

    if not df_list:
        raise ValueError(f"All matching files failed to load for SoC = {soc_value}")

    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.rename(columns={
    combined_df.columns[0]: 'frequency',
    combined_df.columns[1]: 'real impedance',
    combined_df.columns[2]: 'imaginary impedance'
    }, inplace=True)
    return combined_df



#     df = pd.read_csv(csv_path)
#     df.columns = [c.lower().strip() for c in df.columns]
#     df.rename(columns={
#     df.columns[0]: 'frequency',
#     df.columns[1]: 'real impedance',
#     df.columns[2]: 'imaginary impedance'
# }, inplace=True)
#     assert {'frequency', 'real impedance', 'imaginary impedance'} <= set(df.columns)
#     return df