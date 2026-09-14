import os
from dotenv import load_dotenv
from pathlib import Path
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

DATASET_NAME = "davidcariboo/player-scores"
RAW_DATA_DIR = Path("data/raw")

load_dotenv()
KAGGLE_API_TOKEN = os.getenv("KAGGLE_API_TOKEN")

def download_kaggle_dataset(dataset_name: str = DATASET_NAME, output_dir: Path = RAW_DATA_DIR) -> None:
    """Authenticates with Kaggle API, downloads the dataset archive, and extracts all CSV tables into data/raw/."""
        
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Initializing Kaggle API Client...")
    api = KaggleApi()
    api.authenticate()
    
    print(f"[INFO] Downloading dataset: {dataset_name} to {output_dir}...")
    api.dataset_download_files(
        dataset=dataset_name, 
        path=str(output_dir), 
        unzip=False
    )
    
    zip_files = list(output_dir.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No zip archive found in {output_dir} after download.")

    for zip_path in zip_files:
        print(f"[INFO] Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        os.remove(zip_path)
        print(f"[INFO] Removed archive {zip_path.name}")
    
    csv_files = list(output_dir.glob("*.csv"))
    print(f"\n[SUCCESS] Successfully downloaded and extracted {len(csv_files)} CSV tables:")
    for f in sorted(csv_files):
        file_size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name:<25} ({file_size_mb:.2f} MB)")

if __name__ == "__main__":
    download_kaggle_dataset()
