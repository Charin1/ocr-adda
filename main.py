# main.py - OCR benchmark harness (Final Version with Logging and Dynamic Importing)

import argparse
import shutil
import tempfile
import gc
import json
import csv
import sys
from pathlib import Path
from pdf2image import convert_from_path
import importlib
import logging

# --- Local project utilities ---
from utils import (
    read_gemini_ground_truth,
    compute_cer,
    compute_word_measures,
    load_config,
    setup_logger  # Import our logger setup function
)

# Get a logger for this module. The setup will be done in cli().
logger = logging.getLogger(__name__)

# ======================================================================================
#  Runner Configuration
# ======================================================================================
# Stores the full import path to each runner class as a string.
# The script will only import a path if its corresponding flag is true in config.json.
RUNNER_MAP = {
    'tesseract': 'runners.tesseract_runner.TesseractRunner',
    'easyocr': 'runners.easyocr_runner.EasyOCRRunner',
    'paddle_ppstructure': 'runners.paddle_runner.PaddleStructureRunner',
    'gemini_1_5_flash': 'runners.api_runner.GeminiRunner',
    'florence2_base': 'runners.local_llm_runner.FlorenceRunner'
}

# ======================================================================================
#  Helper Functions
# ======================================================================================
def save_text(path, text):
    """Saves text content to a file, creating parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')

def resolve_input_path(cmd_path, default_dir="sample_data"):
    """
    Resolves the input path. If the direct path doesn't exist,
    it tries to find the file in a default directory.
    """
    path = Path(cmd_path)
    if path.exists():
        return path
    fallback_path = Path(default_dir) / path.name
    if fallback_path.exists():
        logger.info(f"Path '{cmd_path}' not found. Using fallback: '{fallback_path}'")
        return fallback_path
    raise FileNotFoundError(f"Input file not found at '{cmd_path}' or in '{default_dir}'.")

# ======================================================================================
#  Core Processing Logic
# ======================================================================================
def process(pdf_path, gemini_gt_path, out_dir, dpi, poppler_path, models_to_run):
    """
    Main function to run the OCR benchmark process.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved in: {out_dir.resolve()}")
    try:
        pdf_path = resolve_input_path(pdf_path)
        gemini_gt_path = resolve_input_path(gemini_gt_path)
    except FileNotFoundError as e:
        logger.critical(f"Input file not found: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info(f"Loading ground truth from: {gemini_gt_path}")
    gemini_gt = read_gemini_ground_truth(gemini_gt_path)

    logger.info(f"Converting PDF '{pdf_path.name}' to images at {dpi} DPI...")
    with tempfile.TemporaryDirectory() as tmpdir:
        page_images = convert_from_path(
            str(pdf_path), dpi=dpi, output_folder=tmpdir,
            fmt='png', poppler_path=poppler_path, paths_only=True
        )
        total_pages = len(page_images)
        logger.info(f"Successfully converted {total_pages} pages.")

        runners = []
        logger.info("Initializing enabled OCR models...")
        for name in models_to_run:
            if name in RUNNER_MAP:
                try:
                    class_path = RUNNER_MAP[name]
                    module_path, class_name = class_path.rsplit('.', 1)
                    
                    logger.info(f"  -> Dynamically importing '{class_name}'...")
                    module = importlib.import_module(module_path)
                    runner_class = getattr(module, class_name)
                    
                    instance = runner_class(['en']) if name == 'easyocr' else runner_class()
                    runners.append(instance)
                    logger.info(f"  [SUCCESS] Initialized '{name}'")
                except Exception as e:
                    logger.error(f"  [FAILURE] Could not initialize '{name}'.", exc_info=True)
            else:
                logger.warning(f"  [SKIPPED] Model '{name}' from config not found in RUNNER_MAP.")
        
        if not runners:
            logger.critical("No models were successfully initialized. Exiting.")
            return

        summary_rows = []
        logger.info("Starting OCR benchmark...")
        for runner in runners:
            model_out_dir = out_dir / runner.name
            model_out_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"--- Running Model: {runner.name} ---")
            for page_idx, img_path in enumerate(page_images, start=1):
                logger.info(f"  -> Processing page {page_idx}/{total_pages}...")
                try:
                    page_text = runner.run_image(str(img_path))
                except Exception as e:
                    logger.error(f"Model '{runner.name}' failed on page {page_idx}.", exc_info=True)
                    page_text = f"[OCR_ERROR: {e}]"
                
                save_text(model_out_dir / f"page_{page_idx}.txt", page_text)
                gt = gemini_gt.get(page_idx, '')
                hyp = page_text or ''
                cer, _ = compute_cer(gt, hyp)
                w, measures = compute_word_measures(gt, hyp) if gt else (None, {})
                char_acc = (1.0 - cer) if cer is not None else None
                row = {'model': runner.name, 'page': page_idx, 'gt_len_chars': len(gt), 'hyp_len_chars': len(hyp), 'cer': cer, 'wer': w, 'char_acc': char_acc, 'substitutions': measures.get('substitutions'), 'deletions': measures.get('deletions'), 'insertions': measures.get('insertions')}
                summary_rows.append(row)
                del page_text, hyp, gt
                gc.collect()
            
            logger.info(f"--- Finished Model: {runner.name}. Releasing resources. ---")
            del runner
            gc.collect()
            try:
                import torch
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except (ImportError, AttributeError):
                pass

    if summary_rows:
        logger.info("Writing summary files...")
        csv_path = out_dir / 'summary.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
            fields = list(summary_rows[0].keys())
            writer = csv.DictWriter(cf, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary_rows)
        (out_dir / 'summary.json').write_text(json.dumps(summary_rows, indent=2), encoding='utf-8')
        logger.info(f"Summary written to {csv_path.name} and summary.json")

    logger.info(f"Benchmark finished. Results are in {out_dir.resolve()}")

# ======================================================================================
#  Command-Line Interface
# ======================================================================================
def cli():
    """Defines the CLI. Model selection is driven ENTIRELY by config.json."""
    # Set up the logger as the very first action.
    setup_logger()

    config = load_config()
    benchmark_config = config.get("benchmark_run", {})
    
    enable_models = benchmark_config.get("enable_models", {})
    models_to_run = [model_name for model_name, is_enabled in enable_models.items() if is_enabled]

    if not models_to_run:
        logger.critical("No models are enabled in 'enable_models' of config.json. Exiting.")
        sys.exit(1)
    
    logger.info(f"Models enabled in config.json: {', '.join(models_to_run)}")

    parser = argparse.ArgumentParser(description="A modular OCR benchmarking tool driven by config.json.")
    
    parser.add_argument('--pdf', default=benchmark_config.get("default_pdf"))
    parser.add_argument('--gemini_gt', default=benchmark_config.get("default_gemini_gt"))
    parser.add_argument('--out_dir', default=benchmark_config.get("default_out_dir", "results"))
    parser.add_argument('--dpi', type=int, default=benchmark_config.get("dpi", 150))
    parser.add_argument('--poppler_path', default=benchmark_config.get("poppler_path"))
    
    args = parser.parse_args()

    process(
        pdf_path=args.pdf,
        gemini_gt_path=args.gemini_gt,
        out_dir=args.out_dir,
        dpi=args.dpi,
        poppler_path=args.poppler_path,
        models_to_run=models_to_run
    )

if __name__ == '__main__':
    cli()