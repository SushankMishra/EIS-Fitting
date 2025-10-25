import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np
from . import config

def train_model(model_name: str):
    """
    Loads features and trains a specified machine learning model.
    Handles feature scaling via a pipeline, which is crucial for models like SVM.

    Args:
        model_name: The key for the model configuration ('rf', 'xgb', or 'svm').
    """
    # Load feature data
    df = pd.read_csv(config.FEATURE_PATH)
    
    # Feature Engineering for extreme SoC conditions
    df['is_extreme_soc'] = ((df['SoC'] <= 0.1) | (df['SoC'] >= 0.9)).astype(int)
    
    # Define features (X) and target (y)
    y = df[config.TARGET_VARIABLE]
    X = df.drop(columns=[config.TARGET_VARIABLE])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # --- Model Pipeline Setup ---
    model_config = config.MODEL_CONFIGS[model_name]
    model_class = model_config['model']
    model_params = model_config['params']
    
    pipeline_steps = []
    
    # Add a scaler to the pipeline if the model requires it (e.g., SVM)
    if model_config['requires_scaling']:
        pipeline_steps.append(('scaler', StandardScaler()))
    
    # Add the model to the pipeline
    pipeline_steps.append((model_name, model_class(**model_params)))
    
    pipeline = Pipeline(pipeline_steps)
    
    # Train the pipeline
    print(f"Training {model_name.upper()} model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Model Evaluation ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"R-squared: {r2:.4f}")
    
    # Save the trained pipeline
    model_path = config.get_model_path(model_name)
    joblib.dump(pipeline, model_path)
    print(f"\nPipeline saved successfully to {model_path}")

