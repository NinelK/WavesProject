import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from TorchLogger import runserver
from src import LOG_DIR

log_dir_path = os.path.join(LOG_DIR)
runserver(log_dir_path)