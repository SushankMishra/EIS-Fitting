import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

# --- Project Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "LiPO 1")
DISCHARGE_DATA_PATH = os.path.join(RAW_DATA_PATH, "Discharge_curve")
EIS_DATA_PATH = os.path.join(RAW_DATA_PATH, "EIS_Charge_discharge")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed")
MODEL_PATH = os.path.join(BASE_DIR, "models")

# --- File Paths ---
FEATURE_PATH = os.path.join(PROCESSED_DATA_PATH, "features.csv")

def get_model_path(model_name: str) -> str:
    """Generates the file path for a given model name."""
    return os.path.join(MODEL_PATH, f"soh_model_{model_name}.joblib")

# --- Feature Engineering ---
CIRCUIT_STRING = 'R0-p(R1,CPE1)-p(R2,CPE2)'
INITIAL_GUESS = [0.01, 0.01, 1e-5, 0.9, 0.05, 1e-4, 0.8]

# --- Model Training ---
TARGET_VARIABLE = 'SoH'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- Model Configurations ---
MODEL_CONFIGS = {
    'rf': {
        'model': RandomForestRegressor,
        'params': {'n_estimators': 150, 'max_depth': 20, 'min_samples_leaf': 2, 'random_state': RANDOM_STATE},
        'requires_scaling': False
    },
    'xgb': {
        'model': XGBRegressor,
        'params': {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 5, 'random_state': RANDOM_STATE, 'objective': 'reg:squarederror'},
        'requires_scaling': False
    },
    'svm': {
        'model': SVR,
        'params': {'C': 1.0, 'epsilon': 0.01, 'kernel': 'rbf'},
        'requires_scaling': True  # SVM is sensitive to feature scaling
    }
}

