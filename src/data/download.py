import os
import sys
import zipfile
from pathlib import Path

import yaml
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

from src.utils.exception import CustomException
from src.utils.logger import logger
from src.utils.paths import CONFIG_DIR, RAW_DIR

load_dotenv()
CONFIG_PATH = CONFIG_DIR / "data_config.yaml"

# Download Kaggle datasets
def download_data(config_path: Path = CONFIG_PATH) -> None:
    try:
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration missing: {config_path}")

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        raw_dir = RAW_DIR
        raw_dir.mkdir(parents=True, exist_ok=True)
        req_files = set(cfg["raw_data"]["required_files"])
        dataset_name = "davidcariboo/player-scores"

        # Kaggle API authentication
        logger.info("Authenticating with Kaggle API...")
        api = KaggleApi()
        api.authenticate()

        # Fetch remote archive
        logger.info(f"Downloading dataset '{dataset_name}' to {raw_dir}...")
        api.dataset_download_files(dataset=dataset_name, path=str(raw_dir), unzip=False)

        zip_files = list(raw_dir.glob("*.zip"))
        if not zip_files:
            raise FileNotFoundError(f"No zip found in {raw_dir}")

        # Extract targeted files
        for zip_path in zip_files:
            logger.info(f"Extracting archive: {zip_path.name}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                archive_files = zip_ref.namelist()
                target_files = [f for f in archive_files if Path(f).name in req_files]
                for file_name in target_files:
                    zip_ref.extract(file_name, path=raw_dir)
            os.remove(zip_path)

        csv_files = list(raw_dir.glob("*.csv"))
        logger.info(f"Extracted {len(csv_files)} files into {raw_dir}.")

    except Exception as err:
        logger.error("Data download failed.")
        raise CustomException(err, sys) from err

if __name__ == "__main__":
    download_data()