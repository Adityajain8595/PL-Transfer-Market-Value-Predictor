import sys

import mlflow

from src.utils.exception import CustomException
from src.utils.logger import logger


# Register champion model
def register_model(model_uri: str, model_name: str):
    try:
        logger.info(f"Registering model from: {model_uri}...")
        reg_model = mlflow.register_model(model_uri=model_uri, name=model_name)
        logger.info(f"Registered '{model_name}' version {reg_model.version}.")
        return reg_model
    except Exception as err:
        logger.error(f"Failed to register: {model_name}")
        raise CustomException(err, sys) from err