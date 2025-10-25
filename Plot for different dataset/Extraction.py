import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
# Load the CSV file
base_dir = "/Users/sushankmishra/Desktop/MTP_Materials/EIS Fitting/SoC Estimation on Li-ion Batteries A New EIS-based Dataset for data-driven applications"

file_paths = []

# Create output directory for plots
output_dir = os.path.join("/Users/sushankmishra/Desktop/MTP_Materials/EIS Fitting/", "Nyquist_Plots")
os.makedirs(output_dir, exist_ok=True)
# Iterate through B01 to B10
for i in range(1, 12):
    folder = f"B{str(i).zfill(2)}"
    full_path1 = os.path.join(base_dir, folder, "EIS measurements", "Test_1", "Hioki")
    full_path2 = os.path.join(base_dir, folder, "EIS measurements", "Test_2", "Hioki")
    file_paths.append(full_path1)
    file_paths.append(full_path2)
    
for folder_index, folder_path in enumerate(file_paths):
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    for csv_index, csv_file in enumerate(csv_files):
        try:
            df = pd.read_csv(csv_file)
            R = df["R(ohm)"]
            X = df["X(ohm)"]
            
            # Determine batch and test
            batch = f"B{str((folder_index // 2) + 1).zfill(2)}"
            test = f"Test_{1 if folder_index % 2 == 0 else 2}"

            # Create batch-specific output directory
            batch_output_dir = os.path.join(output_dir, batch)
            os.makedirs(batch_output_dir, exist_ok=True)

            # Generate unique plot filename
            file_label = f"{batch}_{test}_File{csv_index + 1}"
            output_file = os.path.join(batch_output_dir, f"{file_label}.png")
            
            # Plot and save
            plt.figure()
            plt.plot(R, -X, label=file_label)
            plt.xlabel("Z' (Real Part) [Ohm]")
            plt.ylabel("-Z'' (Imaginary Part) [Ohm]")
            plt.title(f"Nyquist Plot - {file_label}")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_file)
            plt.close()

            print(f"Saved plot: {output_file}")
        
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

# Display the first few rows of the dataframe
