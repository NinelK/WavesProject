import os
import logging

logging.getLogger().setLevel(logging.INFO) # Pass down the tree
h = logging.StreamHandler()
h.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
h.setLevel(level=logging.INFO)
# No default handler (some modules won't see logger otherwise)
logging.getLogger().addHandler(h)

REPOSITORY_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
logging.info("Autodiscovery repository dir: " + REPOSITORY_DIR)

storage_dir = os.environ.get("SCRATCH", "")
if os.path.exists(os.path.join(storage_dir, "WavesProject")):
	storage_dir = os.path.join(storage_dir, "WavesProject")
else:
	storage_dir = REPOSITORY_DIR

DATA_DIR = os.path.join(storage_dir, "dataset")
CENTRES_DIR = os.path.join(storage_dir, "dataset_centres")
MODELS_DIR = os.path.join(storage_dir, "models")
LOG_DIR = os.path.join(storage_dir, "log")
logging.info(DATA_DIR)

if not os.path.exists(DATA_DIR):
	os.mkdir(DATA_DIR)
if not os.path.exists(CENTRES_DIR):
	os.mkdir(CENTRES_DIR)
if not os.path.exists(MODELS_DIR):
	os.mkdir(MODELS_DIR)
if not os.path.exists(LOG_DIR):
	os.mkdir(LOG_DIR)
