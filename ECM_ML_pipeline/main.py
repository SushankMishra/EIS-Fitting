import os
import pandas as pd
import numpy as np
import mlflow
from src.data_loader import load_eis_data
from src.feature_engineering import extract_features
from src.model_training import train_model
from src.model_inference import predict_parameters
from src.utils import mean_percentage_absolute_error
from mlflow.models import infer_signature 
from sklearn.model_selection import train_test_split  
from urllib.parse import urlparse
# from src.utils import preprocess_targets
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Load full dataset (replace with actual target values if available)
raw_df = load_eis_data('ECM_ML_pipeline/data', soc_value=10, batch=1)
X = extract_features(raw_df)

# Simulated targets for illustration (replace with actual known parameters)
# y = pd.DataFrame([[0.05, 50, 1e-5]], columns=['R_ohmic', 'R_ct', 'C_dl'])  # dummy
result_df = pd.read_csv('ECM_ML_pipeline/data/fitted_parameters.csv')  # Load actual target parameters
y = result_df[['R0', 'R2', 'CPE2_0']]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
params = {
    'n_estimators': [100, 200],
    'max_depth': [5,10,None],
    'min_samples_split': [2,5],
    'min_samples_leaf': [1,2]
}
signature = infer_signature(x_test, y_test)

# Inference
with mlflow.start_run() as run:
    mlflow.set_tag("mlflow.runName", "EIS Paramter Estimation")
    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_params(params)
    model_path = 'rf_model.pkl'
    model = train_model(x_train, y_train, model_path, model_type='random_forest',params=params)
    best_model = model.best_estimator_
    y_pred = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mpae = mean_percentage_absolute_error(y_test, y_pred)
    mlflow.log_param("best_n_estimators", model.best_params_['n_estimators'])
    mlflow.log_param("best_max_depth", model.best_params_['max_depth'])
    mlflow.log_param("best_min_samples_split", model.best_params_['min_samples_split'])
    mlflow.log_param("best_min_samples_leaf", model.best_params_['min_samples_leaf'])
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mpae", mpae)
    mlflow.log_metric("r2", r2)
    mlflow.set_tracking_uri(uri = "http://127.0.0.1:5000")
    mlflow.sklearn.log_model(best_model,"model",registered_model_name="Best Randomforest Model")
    


