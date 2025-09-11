# create_ground_truth.py

import argparse
import json
import os
import sys
from pathlib import Path

# --- The correct import for the library you want to use ---
from google import genai
from dotenv import load_dotenv
from pdf2image import convert_from_path
from tqdm import tqdm

# A specific prompt to instruct the model to perform a perfect transcription.
OCR_PROMPT = """
Your task is to perform a perfect, high-fidelity OCR transcription of this document image.
Transcribe the text exactly as it appears.
Preserve all original line breaks, spacing, and formatting.
Do not add any commentary, summarization, or explanation. Output only the transcribed text.
"""

def load_config(config_path="config.json"):
    """
    Loads the configuration file. It looks for the file relative
    to the script's own location, making it independent of the current
    working directory.
    """
    try:
        # Get the absolute path to the directory containing this script
        script_dir = Path(__file__).resolve().parent
        
        # Join the script's directory with the config file name
        config_file_path = script_dir / config_path
        
        if config_file_path.exists():
            print(f"[Info] Loading configuration from: {config_file_path}")
            with open(config_file_path, 'r') as f:
                return json.load(f)
        else:
            # This warning is more helpful as it shows the full path it checked
            print(f"[Warning] Configuration file not found at '{config_file_path}'. Using defaults.")
            return {}
    except Exception as e:
        print(f"[Error] Failed to load or parse config file. Error: {e}", file=sys.stderr)
        return {}

def create_ground_truth(pdf_path: Path, output_json_path: Path, model_name: str, dpi: int, poppler_path: str = None):
    """
    Generates a ground truth JSON file from a PDF using the genai.Client.
    """
    print(f"Starting ground truth generation for '{pdf_path.name}'...")
    print(f"Using model: {model_name}")

    # --- 1. Initialize the GenAI Client ---
    try:
        client = genai.Client()
        if not os.getenv("GOOGLE_API_KEY"):
             raise ValueError("GOOGLE_API_KEY environment variable not set.")
    except Exception as e:
        print(f"[Error] Failed to initialize Google Gen AI Client: {e}", file=sys.stderr)
        sys.exit(1)

    # --- 2. Convert PDF to a list of PIL Images ---
    print(f"Converting PDF to images at {dpi} DPI (this may take a moment)...")
    try:
        page_images = convert_from_path(pdf_path=str(pdf_path), dpi=dpi, poppler_path=poppler_path)
        print(f"Successfully converted {len(page_images)} pages.")
    except Exception as e:
        print(f"[Error] Failed to convert PDF. Is Poppler installed? Error: {e}", file=sys.stderr)
        sys.exit(1)

    # --- 3. Process each page with the client ---
    ground_truth_data = {}
    for i, page_image in enumerate(tqdm(page_images, desc="Transcribing pages")):
        page_num = i + 1
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[OCR_PROMPT, page_image]
            )
            transcribed_text = response.text.strip()
            ground_truth_data[str(page_num)] = transcribed_text
        except Exception as e:
            print(f"\n[Warning] Failed to process page {page_num}. Error: {e}")
            ground_truth_data[str(page_num)] = f"[ERROR: Could not transcribe page {page_num}]"

    # --- 4. Save the results to a JSON file ---
    print(f"\nSaving ground truth data to '{output_json_path}'...")
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(ground_truth_data, f, indent=2, ensure_ascii=False)
    print("Ground truth generation complete!")


def cli():
    """Command-line interface that prioritizes CLI args > config file > defaults."""
    config = load_config()
    gt_config = config.get("ground_truth_generation", {})

    parser = argparse.ArgumentParser(
        description="Create a ground truth JSON file from a PDF using the Gemini API. "
                    "Arguments can be provided via the command line or set as defaults in config.json."
    )
    
    parser.add_argument("--pdf", help="Path to the input PDF file.", default=gt_config.get("default_pdf"))
    parser.add_argument("--output_json", help="Path to save the output ground truth JSON file.", default=gt_config.get("default_output_json"))
    parser.add_argument("--model_name", help="Name of the Gemini model to use.", default=gt_config.get("model_name", "gemini-1.5-pro-latest"))
    parser.add_argument("--dpi", type=int, help="DPI for converting PDF pages.", default=gt_config.get("dpi", 300))
    parser.add_argument("--poppler_path", help="Optional path to Poppler binaries.", default=gt_config.get("poppler_path"))
    
    args = parser.parse_args()

    if not args.pdf:
        print("[Error] An input PDF must be specified via --pdf or in config.json.", file=sys.stderr)
        sys.exit(1)
    if not args.output_json:
        print("[Error] An output JSON path must be specified via --output_json or in config.json.", file=sys.stderr)
        sys.exit(1)

    load_dotenv()
    
    create_ground_truth(
        pdf_path=Path(args.pdf),
        output_json_path=Path(args.output_json),
        model_name=args.model_name,
        dpi=args.dpi,
        poppler_path=args.poppler_path
    )

if __name__ == "__main__":
    cli()