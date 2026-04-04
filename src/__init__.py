from datetime import datetime
import os

from .utils import Logger

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logs_root = os.environ.get("HVIT_LOG_ROOT", "./logs")
checkpoints_root = os.environ.get("HVIT_CHECKPOINT_ROOT", "./checkpoints")

logs_dir = os.path.join(logs_root, current_time)
os.makedirs(logs_dir, exist_ok=True)
logger = Logger(logs_dir)

checkpoints_dir = os.path.join(checkpoints_root, current_time)
os.makedirs(checkpoints_dir, exist_ok=True)
