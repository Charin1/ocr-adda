# main.py

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
from dotenv import load_dotenv

from utils import (
    read_gemini_ground_truth,
    load_config,
    setup_logger,
    compute_all_metrics
)

logger = logging.getLogger(__name__)

# ======================================================================================
#  Runner Configuration
# ======================================================================================
RUNNER_MAP = {
    'tesseract': 'runners.tesseract_runner.TesseractRunner',
    'easyocr': 'runners.easyocr_runner.EasyOCRRunner',
    'paddle_ppstructure': 'runners.paddle_runner.PaddleStructureRunner',
    'gemini_api': 'runners.api_runner.GeminiRunner',
    'florence2_base': 'runners.local_llm_runner.FlorenceRunner',
    'granite_docling': 'runners.granite_runner.GraniteRunner'
}

# ======================================================================================
#  Helper Functions
# ======================================================================================
def save_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')

def resolve_input_path(cmd_path, default_dir="sample_data"):
    path = Path(cmd_path)
    if path.exists(): return path
    fallback_path = Path(default_dir) / path.name
    if fallback_path.exists():
        logger.info(f"Path '{cmd_path}' not found. Using fallback: '{fallback_path}'")
        return fallback_path
    raise FileNotFoundError(f"Input file not found at '{cmd_path}' or in '{default_dir}'.")

# ======================================================================================
#  Core Processing Logic
# ======================================================================================
def process(pdf_path, gemini_gt_path, out_dir, dpi, poppler_path, models_to_run_config):
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
        for model_config in models_to_run_config:
            runner_key = model_config["runner"]
            params = model_config.get("params", {})
            
            if runner_key in RUNNER_MAP:
                try:
                    class_path = RUNNER_MAP[runner_key]
                    module_path, class_name = class_path.rsplit('.', 1)
                    
                    logger.info(f"  -> Dynamically importing '{class_name}' for '{model_config['name']}'...")
                    module = importlib.import_module(module_path)
                    runner_class = getattr(module, class_name)
                    
                    if runner_key == 'easyocr':
                        instance = runner_class(['en'])
                    else:
                        instance = runner_class(**params)
                    
                    runners.append(instance)
                    logger.info(f"  [SUCCESS] Initialized '{model_config['name']}'")
                except Exception as e:
                    logger.error(f"  [FAILURE] Could not initialize '{model_config['name']}'.", exc_info=True)
            else:
                logger.warning(f"  [SKIPPED] Runner key '{runner_key}' not found in RUNNER_MAP.")
        
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
                
                # Call the new comprehensive metrics engine
                metrics = compute_all_metrics(gt, hyp)
                
                row = {
                    'model': runner.name,
                    'page': page_idx,
                    'gt_len_chars': len(gt),
                    'hyp_len_chars': len(hyp)
                }
                row.update(metrics)
                
                summary_rows.append(row)
                
                del page_text, hyp, gt, metrics
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
    setup_logger()
    load_dotenv()

    config = load_config()
    benchmark_config = config.get("benchmark_run", {})
    
    models_config = benchmark_config.get("models", {})
    models_to_run_config = []
    for model_name, config_data in models_config.items():
        if config_data.get("enabled", False):
            config_data['name'] = model_name
            models_to_run_config.append(config_data)

    if not models_to_run_config:
        logger.critical("No models are enabled in the 'models' section of config.json. Exiting.")
        sys.exit(1)
    
    enabled_model_names = [cfg['name'] for cfg in models_to_run_config]
    logger.info(f"Models enabled in config.json: {', '.join(enabled_model_names)}")

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
        models_to_run_config=models_to_run_config
    )

if __name__ == '__main__':
    cli()