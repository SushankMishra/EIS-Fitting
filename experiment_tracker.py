import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

DB_FILE = 'model_comparison_results.json'

def _load_db():
    """Helper function to load the results database."""
    if os.path.exists(DB_FILE):
        with open(DB_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {} # Return empty dict if file is corrupted
    else:
        return {}

def _convert_to_native_types(data):
    """Recursively converts numpy types to native Python types for JSON serialization."""
    if isinstance(data, dict):
        return {k: _convert_to_native_types(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_convert_to_native_types(i) for i in data]
    if isinstance(data, (np.float32, np.float64, np.float16)):
        return float(data)
    if isinstance(data, (np.int32, np.int64, np.int16)):
        return int(data)
    return data

def save_experiment_results(experiment_name, model_architecture, final_rmse, learning_curves):
    """
    Saves the results of a single experiment to the JSON database.

    Args:
        experiment_name (str): A unique name for this run (e.g., "CNN_7_Tower_LR_0.0005").
        model_architecture (str): A name for the model type (e.g., "CNN_7_Tower").
        final_rmse (dict): A dictionary of final RMSE scores {'SoH': 0.05, 'L1': 0.01, ...}.
        learning_curves (dict): A dictionary of lists {'train_loss': [0.9, 0.8, ...], 'val_soh': [0.1, ...], ...}.
    """
    print(f"\n--- Saving results for experiment: {experiment_name} ---")
    
    all_results = _load_db()
    
    # Create the new entry with native Python types
    new_entry = {
        'timestamp': datetime.now().isoformat(),
        'model_architecture': model_architecture,
        'final_rmse': _convert_to_native_types(final_rmse),
        'learning_curves': _convert_to_native_types(learning_curves)
    }
    
    all_results[experiment_name] = new_entry
    
    # Write back to the file
    with open(DB_FILE, 'w') as f:
        json.dump(all_results, f, indent=4)
        
    print(f"--- Results saved successfully to {DB_FILE} ---")

def plot_comparison():
    """
    Loads all saved experiments and generates comparison plots.
    Run this function directly: python experiment_tracker.py
    """
    print(f"--- Loading results from {DB_FILE} for plotting ---")
    all_results = _load_db()
    if not all_results:
        print(f"Error: Database file not found or is empty: {DB_FILE}")
        print("Run a training script first to save some results.")
        return

    # --- 1. Plot Final RMSE Scores ---
    
    # Extract RMSE data into a format pandas can read:
    # {'SoH': {'Exp1': 0.05, 'Exp2': 0.04}, 'L1': {'Exp1': 0.01, 'Exp2': 0.02}, ...}
    rmse_data = {}
    for exp_name, data in all_results.items():
        for param_name, rmse_value in data.get('final_rmse', {}).items():
            if param_name not in rmse_data:
                rmse_data[param_name] = {}
            rmse_data[param_name][exp_name] = rmse_value
            
    if not rmse_data:
        print("No final_rmse data found in any experiment.")
        return
        
    # Convert to DataFrame: Rows=Params, Columns=Experiments
    df_rmse = pd.DataFrame.from_dict(rmse_data, orient='index')
    
    # Plot as a grouped bar chart
    ax_rmse = df_rmse.plot(
        kind='bar', 
        figsize=(15, 8), 
        width=0.8,
        title='Final Model RMSE Comparison (Log Scale)',
        logy=True # Use log scale because L1/W1 are tiny
    )
    ax_rmse.set_ylabel("Final RMSE (Log Scale)")
    ax_rmse.set_xlabel("Predicted Parameter")
    ax_rmse.legend(title='Experiment Name', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('comparison_final_rmse.png')
    print("Saved 'comparison_final_rmse.png'")
    plt.show()

    
    # --- 2. Plot Learning Curves ---
    
    # Find all unique loss keys (e.g., 'train_loss', 'val_soh', 'val_r1', etc.)
    all_loss_keys = set()
    for data in all_results.values():
        all_loss_keys.update(data.get('learning_curves', {}).keys())
        
    if not all_loss_keys:
        print("No learning_curves data found in any experiment.")
        return

    # Get a list of experiment names
    exp_names = list(all_results.keys())

    # Create a separate plot for each loss type
    for loss_key in all_loss_keys:
        plt.figure(figsize=(12, 7))
        ax_lc = plt.gca()
        
        has_data = False
        for exp_name in exp_names:
            curves = all_results[exp_name].get('learning_curves', {})
            if loss_key in curves and curves[loss_key]: # Check if key exists and list is not empty
                ax_lc.plot(curves[loss_key], label=exp_name, alpha=0.8)
                has_data = True
        
        if has_data:
            ax_lc.set_title(f"Learning Curve Comparison: {loss_key}")
            ax_lc.set_xlabel("Epoch")
            ax_lc.set_ylabel("Loss")
            # Use log scale for all losses *except* SoH (which is already tiny)
            if 'soh' not in loss_key.lower():
                ax_lc.set_yscale('log')
                ax_lc.set_ylabel("Loss (Log Scale)")
                
            ax_lc.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plot_filename = f"comparison_lc_{loss_key}.png"
            plt.savefig(plot_filename)
            print(f"Saved '{plot_filename}'")
            plt.show()
        else:
            plt.close() # Close empty plot

# This allows you to run "python experiment_tracker.py" to generate plots
if __name__ == "__main__":
    plot_comparison()