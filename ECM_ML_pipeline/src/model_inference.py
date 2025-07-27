import joblib

def predict_parameters(X_new, model_path):
    model = joblib.load(model_path)
    return model.predict(X_new)