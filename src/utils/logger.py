import logging
from datetime import datetime, timezone

from src.utils.paths import LOGS_DIR

# Central logging configuration
LOGS_DIR.mkdir(parents=True, exist_ok=True)

stamp = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H_%M_%S")
log_path = LOGS_DIR / f"{stamp}.log"

logging.basicConfig(
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("PL_TransferMarket")
