import json, os, shutil, gc
from pathlib import Path
import Levenshtein
from jiwer import wer, compute_measures
import sys
import logging
from colorlog import ColoredFormatter

def setup_logger():
    """
    Sets up a project-wide logger that outputs to both console and a file.
    """
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO) # Set the minimum level to capture

    # Prevent the logger from being configured multiple times
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # --- File Handler ---
    # This handler writes all logs (INFO and above) to a file.
    log_file_path = "benchmark.log"
    file_handler = logging.FileHandler(log_file_path, mode='w') # 'w' for overwrite each run
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # --- Console Handler (with colors) ---
    # This handler prints logs to the console with colors for different levels.
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = ColoredFormatter(
        "%(log_color)s%(asctime)s [%(levelname)-8s]%(reset)s %(blue)s%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Initial log to confirm setup
    logging.getLogger(__name__).info(f"Logger configured. Log file at: {log_file_path}")


def load_config(config_path="config.json"):
    """
    Loads the configuration file. It looks for the file relative
    to the script's own location, making it independent of the current
    working directory. Includes diagnostics.
    """
    try:
        script_dir = Path(__file__).resolve().parent # This will be the utils.py's parent
        # We need to go one level up to find config.json if utils.py is in a subfolder
        # Assuming utils.py is in the root, this is fine. If utils.py moves to a subfolder,
        # this path needs adjustment (e.g., script_dir.parent / config_path)
        
        # For this project structure, utils.py is at the root, so script_dir is the project root.
        config_file_path = script_dir / config_path
        
        if config_file_path.exists():
            # print(f"[Info] Loading configuration from: {config_file_path}") # Suppress for cleaner output
            with open(config_file_path, 'r') as f:
                return json.load(f)
        else:
            print(f"[Warning] Configuration file not found at '{config_file_path}'. Using defaults.", file=sys.stderr)
            return {}
    except Exception as e:
        print(f"[Error] Failed to load or parse config file. Error: {e}", file=sys.stderr)
        return {}

def read_gemini_ground_truth(gemini_path):
    p = Path(gemini_path)
    if p.is_file():
        if p.suffix.lower() == '.json':
            data = json.loads(p.read_text(encoding='utf-8'))
            return {int(k): v for k, v in data.items()}
        else:
            txt = p.read_text(encoding='utf-8')
            if '\f' in txt:
                pages = txt.split('\f')
                return {i+1: pages[i].strip() for i in range(len(pages))}
            else:
                return {1: txt.strip()}
    elif p.is_dir():
        out = {}
        for f in p.iterdir():
            name = f.stem
            try:
                idx = int(name.split('_')[-1])
            except:
                try:
                    idx = int(name)
                except:
                    continue
            out[idx] = f.read_text(encoding='utf-8').strip()
        return dict(sorted(out.items()))
    else:
        raise FileNotFoundError(str(p))

def compute_cer(gt, hyp):
    if gt is None or len(gt) == 0:
        return None, None
    dist = Levenshtein.distance(gt, hyp)
    cer = dist / len(gt)
    return cer, dist

def compute_word_measures(gt, hyp):
    measures = compute_measures(gt, hyp)
    w = wer(gt, hyp)
    return w, measures
