import pandas as pd
import numpy as np
import joblib
from .feature_engineering import fit_ecm
from . import config

def predict_soh(file_path: str, model_name: str, soc_value: float) -> float:
    """
    Predicts the SoH for a single EIS file using a specified trained model pipeline.

    Args:
        file_path: The full path to the new EIS .csv file.
        model_name: The model to use for prediction ('rf', 'xgb', 'svm').
        soc_value: The State of Charge at which the EIS was measured.

    Returns:
        The predicted SoH value as a float, or None on failure.
    """
    print(f"Loading data from {file_path} at SoC={soc_value:.2f}...")
    try:
        df = pd.read_csv(file_path, header=None, names=['Freq', 'Re', 'Im'], delimiter=',')
        
        freq = df['Freq'].values
        z_complex = df['Re'].values + 1j * df['Im'].values
        
        print("Extracting features by fitting ECM...")
        features_dict = fit_ecm(freq, z_complex)
        
        if any(np.isnan(val) for val in features_dict.values()):
            raise ValueError("ECM fitting failed for the provided data.")

        features_df = pd.DataFrame([features_dict])
        features_df['SoC'] = soc_value
        features_df['is_extreme_soc'] = ((features_df['SoC'] <= 0.1) | (features_df['SoC'] >= 0.9)).astype(int)
        features_df['Cycle'] = 0  # Placeholder, as the model was trained with this column
        
        # Load the trained model pipeline
        model_path = config.get_model_path(model_name)
        pipeline = joblib.load(model_path)
        
        # Ensure feature order matches what the model was trained on
        # The pipeline object has the final estimator which has this attribute
        model_in_pipeline = pipeline.steps[-1][1]
        if hasattr(model_in_pipeline, 'feature_names_in_'):
             features_for_prediction = features_df[model_in_pipeline.feature_names_in_]
        else: # for models like SVR that might not store it
            # Manually get columns from training data if needed, but pipeline handles this.
            # Here we assume the dataframe columns are in the right order.
             features_for_prediction = features_df 

        print("Making prediction...")
        # The pipeline automatically handles scaling and then predicts
        prediction = pipeline.predict(features_for_prediction)
        
        return prediction[0]
        
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return None

