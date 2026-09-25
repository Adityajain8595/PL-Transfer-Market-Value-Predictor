import subprocess
import sys

from src.utils.exception import CustomException
from src.utils.logger import logger


# DagsHub DVC remote synchronization
def sync_dvc(action: str = "pull") -> None:
    try:
        logger.info(f"Running DVC {action}...")
        cmd = ["dvc", action]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(res.stdout)
        logger.info(f"DVC {action} completed successfully.")
    except Exception as err:
        logger.error(f"DVC sync failed for {action}.")
        raise CustomException(err, sys) from err

if __name__ == "__main__":
    sync_dvc("pull")