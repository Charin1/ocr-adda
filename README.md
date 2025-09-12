# Advanced OCR Benchmark Tool

![status](https://img.shields.io/badge/status-in_development-yellow)

⚠️ This project is under active development and is **not production-ready**.

An extensible, memory-efficient benchmarking tool designed to compare the performance of traditional OCR engines against modern cloud-based and local Large Language Models (LLMs). This project is optimized for resource-constrained environments and provides a robust framework for OCR evaluation.

## Key Features

- **Config-Driven Model Selection**: Easily enable or disable models via a central `config.json` file to manage memory and control benchmark scope.
- **Memory Efficient**: Utilizes dynamic importing to only load a model's heavy libraries (like PyTorch/Transformers) if that model is enabled.
- **Sequential Processing**: Runs models one-by-one, releasing memory after each run to prevent system crashes.
- **Extensible Runner Architecture**: Easily add new OCR engines or LLMs by creating a simple runner class.
- **Automated Ground Truth Generation**: Includes a helper script to generate high-quality ground truth files from a PDF using a powerful Gemini model.
- **Professional Logging**: All operations are logged to both the console (with colors) and a `benchmark.log` file for easy debugging and analysis.
- **Detailed Metrics**: Automatically calculates and saves Character Error Rate (CER), Word Error Rate (WER), and other key accuracy metrics to a `summary.csv`.

## Project Structure

The project is organized to separate concerns, making it easy to manage and extend.

```
ocr_benchmark_project/
├── .env                  # For storing your API key (must be created manually)
├── .gitignore            # Specifies files for Git to ignore
├── config.json           # The master control file for paths and model toggles
├── main.py               # The main script to run the benchmark
├── requirements.txt      # A list of all Python dependencies
├── create_ground_truth.py # A helper script to automate GT creation
├── runners/
│   ├── __init__.py
│   ├── api_runner.py         # Gemini 1.5 Flash runner
│   ├── easyocr_runner.py     # EasyOCR runner
│   ├── local_llm_runner.py   # Florence-2 (local LLM) runner
│   ├── paddle_runner.py      # PaddleOCR runner
│   └── tesseract_runner.py   # Tesseract runner
└── utils.py                # Helper functions for metrics, config, and logging
```

---

## Setup and Installation

Follow these steps to set up and run the project on a macOS system.

### Step 1: Prerequisites (System Dependencies)

Ensure you have [Homebrew](https://brew.sh/) installed. Then, install the required system packages:

```bash
brew install poppler tesseract
```

### Step 2: Clone the Repository

```bash
git clone <your-repository-url>
cd ocr_benchmark_project
```

### Step 3: Set Up the Python Environment

It is highly recommended to use a virtual environment.

```bash
# Create and activate the virtual environment
python3 -m venv venv
source venv/bin/activate

# Install all required Python packages
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure the Project

You must create and configure two files before running the application.

**1. Create the `.env` file for your API Key:**
   - Create a new file named `.env` in the project root.
   - Add your Google Gemini API key to it:
     ```
     GOOGLE_API_KEY="your-google-api-key-here"
     ```

**2. Configure `config.json`:**
   - This is the master control file. Open `config.json` and edit the paths and model flags.
   - **Set the `poppler_path`**: This is critical. Find your Poppler `bin` directory by running `brew --prefix poppler` and add `/bin` to the end of the result.
   - **Enable Models**: Set the flags in `enable_models` to `true` for the models you wish to run.

   ```json
   {
     "ground_truth_generation": {
       "poppler_path": "/opt/homebrew/opt/poppler/bin"
     },
     "benchmark_run": {
       "poppler_path": "/opt/homebrew/opt/poppler/bin",
       "enable_models": {
         "tesseract": true,
         "easyocr": true,
         "paddle_ppstructure": false,
         "gemini_1_5_flash": true,
         "florence2_base": false
       }
     }
   }
   ```

## How to Run

The recommended workflow is a two-step process: first, generate a high-quality ground truth file, then run the benchmark against it.

### Step 1: Generate the Ground Truth (Optional but Recommended)

Use the provided script to automatically transcribe your PDF using Gemini 1.5 Pro. This creates a reliable ground truth for your benchmark.

```bash
# This command uses the default paths set in config.json
python create_ground_truth.py

# Or, specify your own files
python create_ground_truth.py --pdf "path/to/my_document.pdf" --output_json "path/to/my_document_gt.json"
```
This will create a new `.json` file containing the transcribed text for each page.

### Step 2: Run the Benchmark

Run the main script. The models that are executed are controlled **only** by the `enable_models` flags in `config.json`.

```bash
# Run the benchmark using the default PDF and GT paths from config.json
python main.py

# Or, override the paths for a specific run
python main.py --pdf "path/to/my_document.pdf" --gemini_gt "path/to/my_document_gt.json" --out_dir "my_document_results"
```

## Understanding the Output

The primary output is `summary.csv`, which contains a rich set of metrics for analysis:

### Core Metrics
- **`model`**: The name of the OCR model that was run.
- **`page`**: The page number of the PDF.
- **`gt_len_chars`**: The total number of characters in the ground truth text.
- **`hyp_len_chars`**: The total number of characters in the model's output text.

### Accuracy & Error Metrics
- **`cer` (Character Error Rate)**: The percentage of characters that were incorrectly transcribed. **Lower is better.**
- **`wer` (Word Error Rate)**: The percentage of words that were incorrect. **Lower is better.**
- **`char_acc` (Character Accuracy)**: `1 - CER`. The percentage of characters that were correct. **Higher is better.**
- **`word_acc` (Word Accuracy)**: `1 - WER`. The percentage of words that were correct. **Higher is better.**
- **`levenshtein_dist`**: The raw number of character edits (insertions, deletions, substitutions) needed to change the hypothesis to the ground truth.

### Similarity Metrics
- **`fuzz_ratio`**: A string similarity score from 0.0 to 1.0 based on Levenshtein distance. **Higher is better.**
- **`bleu` (Bilingual Evaluation Understudy)**: A score from 0.0 to 1.0 that measures n-gram precision. Originally for translation, it's useful for measuring sentence structure similarity. **Higher is better.**
- **`rougeL_f1` (Recall-Oriented Understudy for Gisting Evaluation)**: The F1-score for the longest common subsequence. It's good at capturing sentence-level structural similarity. **Higher is better.**

## Status
This project is still in its early stages. Expect breaking changes.