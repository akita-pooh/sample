import lightgbm as lgb

from experiment_tools.set_up import start_experiment
from utils.cfg_diff import get_config, get_diff


default_filename = "../config/default/LightGBM.json"
exp_filename = "../config/experiment/LightGBM.json"

_, cfg, default_str, exp_str = get_config(default_filename, exp_filename)

logger = start_experiment(cfg)
logger = get_diff(default_str, exp_str, logger)

device = "cpu"
logger.info(f"device: {device}")
