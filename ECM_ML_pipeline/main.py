import os
import pandas as pd
import numpy as np
import mlflow
from src.data_loader import load_eis_data
from src.feature_engineering import process_eis_features
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
raw_df = load_eis_data('ECM_ML_pipeline/data', soc_value=50)
X = process_eis_features('/Users/sushankmishra/Desktop/MTP_Materials/EIS Fitting/ECM_ML_pipeline/data')

# Simulated targets for illustration (replace with actual known parameters)
# y = pd.DataFrame([[0.05, 50, 1e-5]], columns=['R_ohmic', 'R_ct', 'C_dl'])  # dummy
result_df = pd.read_csv('ECM_ML_pipeline/data/fitted_params_filewise.csv')  # Load actual target parameters
y = result_df[['R1','CPE1_T','CPE1_P','R2','Wo3_R','Wo3_T','W1_R']]
x_train, x_test, y_train, y_test = X[:17],X[17:], y[:17], y[17:]


# Train
params = {
    'n_estimators': [100, 200],
    'max_depth': [5,10,None],
    'min_samples_split': [2,5],
    'min_samples_leaf': [1,2]
}
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
    signature = infer_signature(x_test, y_test)
    mlflow.log_param("best_n_estimators", model.best_params_['n_estimators'])
    mlflow.log_param("best_max_depth", model.best_params_['max_depth'])
    mlflow.log_param("best_min_samples_split", model.best_params_['min_samples_split'])
    mlflow.log_param("best_min_samples_leaf", model.best_params_['min_samples_leaf'])
    print("True Parameters:", y_test.iloc[0].values)
    print("Estimated Parameters:", y_pred[0])
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mpae", mpae)
    mlflow.log_metric("r2", r2)
    mlflow.set_tracking_uri(uri = "http://127.0.0.1:5000")
    mlflow.sklearn.log_model(best_model,"model",registered_model_name="Best Randomforest Model")
    


