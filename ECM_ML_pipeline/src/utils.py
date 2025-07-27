import numpy as np
def mean_percentage_absolute_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100