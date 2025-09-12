# utils.py

import json
import os
import sys
from pathlib import Path
import logging
from colorlog import ColoredFormatter

import Levenshtein
from jiwer import wer, compute_measures
from rapidfuzz.fuzz import ratio as fuzz_ratio

# --- Imports for Advanced Metrics ---
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# ======================================================================================
#  Logger and Config Setup
# ======================================================================================
def setup_logger():
    """
    Sets up a project-wide logger that outputs to both console and a file.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    log_file_path = "benchmark.log"
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

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

    logging.getLogger(__name__).info(f"Logger configured. Log file at: {log_file_path}")

def load_config(config_path="config.json"):
    """
    Loads the configuration file relative to this script's location.
    """
    try:
        script_dir = Path(__file__).resolve().parent
        config_file_path = script_dir / config_path
        if config_file_path.exists():
            with open(config_file_path, 'r') as f:
                return json.load(f)
        else:
            logging.warning(f"Configuration file not found at '{config_file_path}'. Using defaults.")
            return {}
    except Exception as e:
        logging.error(f"Failed to load or parse config file.", exc_info=True)
        return {}

# ======================================================================================
#  Ground Truth Reader
# ======================================================================================
def read_gemini_ground_truth(gemini_path):
    """
    Reads ground truth data from a JSON file, a text file, or a directory of text files.
    """
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

# ======================================================================================
#  Comprehensive Metrics Engine (without Cosine Similarity)
# ======================================================================================
def compute_all_metrics(gt: str, hyp: str) -> dict:
    """
    Calculates a comprehensive suite of OCR accuracy and similarity metrics.

    Args:
        gt: The ground truth text.
        hyp: The hypothesis (OCR output) text.

    Returns:
        A dictionary containing all calculated metrics.
    """
    # Handle empty ground truth case
    if not gt or not gt.strip():
        return {
            'cer': None, 'wer': None, 'char_acc': None, 'word_acc': None,
            'levenshtein_dist': None, 'fuzz_ratio': None, 'bleu': None,
            'rougeL_f1': None, 'substitutions': None, 'deletions': None, 'insertions': None
        }

    # 1. Edit Distance Based Metrics
    lev_dist = Levenshtein.distance(gt, hyp)
    cer = lev_dist / len(gt) if len(gt) > 0 else 1.0
    
    word_measures = compute_measures(gt, hyp)
    wer_val = word_measures['wer']

    # 2. Accuracy Scores
    char_acc = 1.0 - cer
    word_acc = 1.0 - wer_val

    # 3. String Similarity Scores
    fuzz = fuzz_ratio(gt, hyp) / 100.0  # Normalize to 0-1 range

    # 4. NLP-based Metrics (BLEU and ROUGE)
    gt_tokens = [gt.split()]
    hyp_tokens = hyp.split()
    
    # BLEU Score
    chencherry = SmoothingFunction()
    bleu = sentence_bleu(gt_tokens, hyp_tokens, smoothing_function=chencherry.method1)
    
    # ROUGE-L Score
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(gt, hyp)
    rouge_l_f1 = rouge_scores['rougeL'].fmeasure

    return {
        'cer': cer,
        'wer': wer_val,
        'char_acc': char_acc,
        'word_acc': word_acc,
        'levenshtein_dist': lev_dist,
        'fuzz_ratio': fuzz,
        'bleu': bleu,
        'rougeL_f1': rouge_l_f1,
        'substitutions': word_measures['substitutions'],
        'deletions': word_measures['deletions'],
        'insertions': word_measures['insertions'],
    }