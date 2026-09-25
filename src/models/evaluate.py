import sys

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.exception import CustomException


# Evaluate prediction metrics
def evaluate_preds(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
    try:
        y_true = np.expm1(y_true_log)
        y_pred = np.clip(np.expm1(y_pred_log), a_min=0, a_max=None)

        return {
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": float(r2_score(y_true, y_pred))
        }
    except Exception as err:
        raise CustomException(err, sys) from err