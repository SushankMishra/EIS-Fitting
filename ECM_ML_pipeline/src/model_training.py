import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

def hyperparameter_tuning(model_type,params, x_train, y_train):
    # Placeholder for hyperparameter tuning logic
    # This function should return the best parameters after tuning
    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    elif model_type == 'xgboost':
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

    elif model_type == 'linear':
        model = LinearRegression()

    elif model_type == 'plsr':
        # Adjust n_components as needed
        model = PLSRegression(n_components=min(x_train.shape[1], y_train.shape[1] if len(y_train.shape) > 1 else 1))

    elif model_type == 'pcr':
        # PCA followed by linear regression
        pca_components = min(X.shape[1], 5)  # You can adjust this
        model = make_pipeline(PCA(n_components=pca_components,**params), LinearRegression())

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    grid_search = GridSearchCV(estimator=model, param_grid=params, cv=3,n_jobs=-1,verbose=2, scoring='neg_mean_squared_error')
    grid_search.fit(x_train, y_train)
    return grid_search

def train_model(X, y, model_path, model_type,params=None):
    # Select model based on model_type
    # Fit the selected model
    model = hyperparameter_tuning(model_type, params, X, y)
    # Save it
    joblib.dump(model, model_path)
    return model